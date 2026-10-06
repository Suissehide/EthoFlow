"""RMSE par keypoint : lequel des 12 points tire la moyenne vers le bas ?

`evaluate_network` sort une RMSE globale. Elle dit si le modèle est bon,
pas *où* il échoue. Or l'écart typique entre `rmse` et `rmse_pcutoff` —
par exemple 125 px contre 4.4 px — vient rarement de tous les keypoints à
parts égales : quelques-uns sont invisibles une partie du temps (les
pattes qui passent sous le corps en bottom-view) et le modèle les place
n'importe où, avec une confiance basse.

Ce script répond aux deux questions qui décident de la suite :

  · **Quels keypoints** sont mal prédits → ceux sur lesquels extraire des
    frames supplémentaires à l'étape B.6.
  · **Quelle fraction des prédictions dépasse le seuil de confiance** →
    combien de points le nettoyage de l'étape 6b va devoir interpoler.
    Un keypoint à 40 % au-dessus du seuil est utilisable ; à 5 %, il vaut
    mieux le retirer des features VAME (`filter_keypoints.py`).

Usage :
    # Menu des modèles trouvés sous D:/EthoFlow/models
    python scripts/dlc_model-training/keypoint_errors.py

    # Sur un modèle précis
    python scripts/dlc_model-training/keypoint_errors.py \\
        --model-dir D:/EthoFlow/models/bottomview_129_IR

    # Autre seuil de confiance, et export CSV
    python scripts/dlc_model-training/keypoint_errors.py \\
        --model-dir <...> --pcutoff 0.3 --out erreurs.csv

Lecture des colonnes :
    n           instances labellisées de ce keypoint
    %>seuil     part des prédictions au-dessus du seuil de confiance
    rmse        RMSE sur toutes les instances (px)
    rmse_seuil  RMSE sur les seules instances confiantes (px)
    médiane     erreur médiane, insensible aux valeurs extrêmes (px)

`rmse` très supérieure à `rmse_seuil` sur un keypoint = le modèle sait
qu'il ne sait pas. C'est le comportement souhaitable : ces points seront
écartés par le cutoff en aval. L'inverse — `rmse_seuil` élevée — est le
vrai problème : le modèle se trompe **en étant confiant**, et aucun
filtrage ne le rattrapera.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from interactive import DEFAULT_MODELS_ROOT, prompt, prompt_existing_path  # noqa: E402


DEFAULT_PCUTOFF = 0.6

# Au-delà, une prédiction n'est plus imprécise : elle est ailleurs. 25 px
# ≈ 8 % d'une souris de 300 px, bien au-dessus du bruit de clic (2-3 px).
SEUIL_GROSSIER_PX = 25.0


def bodyparts_of(df: pd.DataFrame) -> list[str]:
    """Noms de keypoints d'un DataFrame DLC, quel que soit le nb de niveaux."""
    names = list(df.columns.names or [])
    if "bodyparts" in names:
        return list(dict.fromkeys(df.columns.get_level_values("bodyparts")))
    # Repli : avant-dernier niveau (…, bodyparts, coords)
    return list(dict.fromkeys(c[-2] for c in df.columns))


def column_for(df: pd.DataFrame, bodypart: str, coord: str):
    """Première colonne (bodypart, coord) du DataFrame, ou None."""
    for col in df.columns:
        if col[-2] == bodypart and col[-1] == coord:
            return col
    return None


def normalise_index(df: pd.DataFrame) -> pd.DataFrame:
    """Ramène l'index à une chaîne unique par image.

    DLC indexe tantôt par un MultiIndex ('labeled-data', 'video', 'img.png'),
    tantôt par le chemin en une seule chaîne, selon la version et selon
    qu'on lit la vérité terrain ou les prédictions. On normalise pour
    pouvoir aligner les deux.
    """
    df = df.copy()
    if isinstance(df.index, pd.MultiIndex):
        df.index = ["/".join(str(p) for p in idx) for idx in df.index]
    else:
        df.index = [str(i).replace("\\", "/") for i in df.index]
    return df


def per_keypoint_errors(df_gt: pd.DataFrame, df_pred: pd.DataFrame,
                         pcutoff: float = DEFAULT_PCUTOFF) -> pd.DataFrame:
    """Erreur de prédiction par keypoint.

    Args:
        df_gt: vérité terrain (colonnes …/bodyparts/{x,y})
        df_pred: prédictions (colonnes …/bodyparts/{x,y,likelihood})
        pcutoff: seuil de confiance

    Returns:
        Un DataFrame trié par rmse_seuil décroissante, une ligne par
        keypoint plus une ligne TOTAL.
    """
    gt, pred = normalise_index(df_gt), normalise_index(df_pred)
    communes = [i for i in gt.index if i in set(pred.index)]
    if not communes:
        raise ValueError(
            "Aucune image commune entre vérité terrain et prédictions. "
            "Les index ne se recoupent pas — vérifie que les prédictions "
            "viennent bien de ce projet."
        )
    gt, pred = gt.loc[communes], pred.loc[communes]

    lignes = []
    err_tout, err_conf = [], []
    n_tout = n_conf = 0

    for bp in bodyparts_of(gt):
        cgx, cgy = column_for(gt, bp, "x"), column_for(gt, bp, "y")
        cpx, cpy = column_for(pred, bp, "x"), column_for(pred, bp, "y")
        if None in (cgx, cgy, cpx, cpy):
            continue  # keypoint absent d'un des deux côtés
        cpl = column_for(pred, bp, "likelihood")

        gx = gt[cgx].to_numpy(float)
        gy = gt[cgy].to_numpy(float)
        px = pred[cpx].to_numpy(float)
        py = pred[cpy].to_numpy(float)
        lik = (pred[cpl].to_numpy(float) if cpl is not None
               else np.ones_like(px))

        # Une instance non labellisée n'est pas une erreur du modèle.
        valide = ~(np.isnan(gx) | np.isnan(gy) | np.isnan(px) | np.isnan(py))
        if not valide.any():
            continue
        err = np.hypot(gx[valide] - px[valide], gy[valide] - py[valide])
        confiant = lik[valide] >= pcutoff

        lignes.append({
            "keypoint": bp,
            "n": int(valide.sum()),
            "pct_au_dessus_seuil": 100.0 * confiant.mean(),
            "rmse": float(np.sqrt(np.mean(err ** 2))),
            "rmse_seuil": (float(np.sqrt(np.mean(err[confiant] ** 2)))
                           if confiant.any() else float("nan")),
            "mediane": float(np.median(err)),
            # La RMSE est dominée par quelques frames aberrantes : un point
            # juste à 1.5 px dans 95 % des cas affiche 70 px de RMSE si 5 %
            # des frames sont à 300 px. Les deux colonnes ci-dessous
            # séparent ce qui est systématique (médiane des prédictions
            # confiantes) de ce qui est ponctuel (part d'erreurs grossières).
            "mediane_seuil": (float(np.median(err[confiant]))
                              if confiant.any() else float("nan")),
            "pct_grossieres": 100.0 * float((err > SEUIL_GROSSIER_PX).mean()),
        })
        err_tout.append(err)
        err_conf.append(err[confiant])
        n_tout += int(valide.sum())
        n_conf += int(confiant.sum())

    if not lignes:
        raise ValueError("Aucun keypoint comparable entre les deux fichiers.")

    df = pd.DataFrame(lignes).sort_values(
        "rmse_seuil", ascending=False, na_position="first"
    ).reset_index(drop=True)

    tout = np.concatenate(err_tout)
    conf = np.concatenate([e for e in err_conf if e.size])
    total = {
        "keypoint": "TOTAL",
        "n": n_tout,
        "pct_au_dessus_seuil": 100.0 * n_conf / max(n_tout, 1),
        "rmse": float(np.sqrt(np.mean(tout ** 2))),
        "rmse_seuil": (float(np.sqrt(np.mean(conf ** 2)))
                       if conf.size else float("nan")),
        "mediane": float(np.median(tout)),
        "mediane_seuil": float(np.median(conf)) if conf.size else float("nan"),
        "pct_grossieres": 100.0 * float((tout > SEUIL_GROSSIER_PX).mean()),
    }
    return pd.concat([df, pd.DataFrame([total])], ignore_index=True)


def _aligner(df_gt: pd.DataFrame, df_pred: pd.DataFrame):
    gt, pred = normalise_index(df_gt), normalise_index(df_pred)
    communes = [i for i in gt.index if i in set(pred.index)]
    return gt.loc[communes], pred.loc[communes]


def _xy(df: pd.DataFrame, bp: str):
    cx, cy = column_for(df, bp, "x"), column_for(df, bp, "y")
    if cx is None or cy is None:
        return None
    return df[cx].to_numpy(float), df[cy].to_numpy(float)


def diagnostic_confusions(df_gt: pd.DataFrame, df_pred: pd.DataFrame
                           ) -> dict:
    """Où tombent réellement les prédictions fausses ?

    Deux confusions typiques de la vue de dessous, qu'une RMSE ne
    distingue pas d'une simple imprécision :

      · **Côté** — la patte droite est prédite à l'emplacement de la
        gauche (ou l'inverse). On compare la prédiction de chaque membre
        d'une paire à la vérité terrain de son JUMEAU : si elle en est
        plus proche que de sa propre cible, le réseau a confondu les
        deux.

      · **Retournement tête/queue** — sur certaines frames, le nez est
        prédit à l'arrière et la queue à l'avant. Erreur d'environ une
        longueur de corps, souvent avec une confiance élevée, sur une
        poignée de frames : c'est elle qui gonfle la RMSE de nose, chin
        et tail_tip alors que leur médiane reste à 1-3 px.
    """
    gt, pred = _aligner(df_gt, df_pred)
    out: dict = {"paires": [], "retournement": None}

    # --- confusion de côté ---
    bps = bodyparts_of(gt)
    for g in [b for b in bps if str(b).endswith("_left")]:
        d = str(g)[: -len("_left")] + "_right"
        if d not in bps:
            continue
        gg, gd, pg, pd_ = _xy(gt, g), _xy(gt, d), _xy(pred, g), _xy(pred, d)
        if None in (gg, gd, pg, pd_):
            continue
        for nom, p, propre, jumeau in ((d, pd_, gd, gg), (g, pg, gg, gd)):
            vers_propre = np.hypot(p[0] - propre[0], p[1] - propre[1])
            vers_jumeau = np.hypot(p[0] - jumeau[0], p[1] - jumeau[1])
            ok = ~(np.isnan(vers_propre) | np.isnan(vers_jumeau))
            if not ok.any():
                continue
            out["paires"].append({
                "keypoint": nom,
                "pct_sur_jumeau": 100.0 * float(
                    (vers_jumeau[ok] < vers_propre[ok]).mean()),
                "mediane_vers_jumeau": float(np.median(vers_jumeau[ok])),
            })

    # --- retournement tête/queue ---
    nez_gt, base_gt = _xy(gt, "nose"), _xy(gt, "tail_base")
    nez_pr = _xy(pred, "nose")
    if None not in (nez_gt, base_gt, nez_pr):
        vers_nez = np.hypot(nez_pr[0] - nez_gt[0], nez_pr[1] - nez_gt[1])
        vers_base = np.hypot(nez_pr[0] - base_gt[0], nez_pr[1] - base_gt[1])
        ok = ~(np.isnan(vers_nez) | np.isnan(vers_base))
        retourne = ok & (vers_base < vers_nez)
        out["retournement"] = {
            "n": int(retourne.sum()),
            "total": int(ok.sum()),
            "frames": [str(i) for i in np.asarray(gt.index)[retourne]],
        }
    return out


def dossiers_evaluation(model_dir: Path) -> list[Path]:
    """Dossiers de résultats d'évaluation, quel que soit le backend.

    DLC suffixe ses dossiers selon le backend : `evaluation-results` pour
    TensorFlow, `evaluation-results-pytorch` pour le backend torch de
    DLC 3.x. On ne peut pas deviner lequel existe, donc on prend les deux.
    """
    return [d for d in sorted(model_dir.glob("evaluation-results*"))
            if d.is_dir()]


def find_predictions(model_dir: Path) -> Path | None:
    """Fichier de prédictions le plus récent, tous dossiers d'éval confondus.

    On écarte les `*-results.h5` : ce sont les tableaux de métriques
    agrégées (une ligne par snapshot), pas les prédictions par image.
    """
    candidats = []
    for root in dossiers_evaluation(model_dir):
        candidats += [p for p in root.rglob("*.h5")
                      if not p.stem.lower().endswith("-results")]
    if not candidats:
        return None
    return max(candidats, key=lambda p: p.stat().st_mtime)


def diagnostic_absence(model_dir: Path) -> str:
    """Message d'erreur qui montre ce qui existe réellement sur le disque."""
    dossiers = dossiers_evaluation(model_dir)
    if not dossiers:
        return (f"   Aucun dossier evaluation-results* dans {model_dir}.\n"
                f"   L'évaluation n'a jamais tourné.")
    lignes = [f"   Dossiers trouvés : "
              f"{', '.join(d.name for d in dossiers)}"]
    for d in dossiers:
        fichiers = sorted(p.name for p in d.rglob("*") if p.is_file())
        if not fichiers:
            lignes.append(f"     {d.name}/ : vide")
            continue
        lignes.append(f"     {d.name}/ contient :")
        lignes += [f"       · {f}" for f in fichiers[:8]]
        if len(fichiers) > 8:
            lignes.append(f"       … et {len(fichiers) - 8} autre(s)")
    lignes.append("   Si un .h5 de prédictions figure ci-dessus, passe-le "
                  "directement avec --predictions.")
    return "\n".join(lignes)


def load_ground_truth(model_dir: Path) -> pd.DataFrame:
    """Concatène tous les CollectedData_*.h5 de labeled-data/."""
    labeled = model_dir / "labeled-data"
    fichiers = sorted(labeled.glob("*/CollectedData_*.h5")) if labeled.exists() else []
    if not fichiers:
        raise FileNotFoundError(
            f"Aucun CollectedData_*.h5 dans {labeled} — le projet n'a pas "
            f"de frames labellisées."
        )
    return pd.concat([pd.read_hdf(f) for f in fichiers])


def choisir_modele(model_dir: Path | None) -> Path:
    """Résout le modèle depuis l'argument, ou propose un menu."""
    if model_dir is not None:
        return Path(model_dir).resolve()
    modeles = []
    if DEFAULT_MODELS_ROOT.exists():
        modeles = sorted(d for d in DEFAULT_MODELS_ROOT.iterdir()
                         if d.is_dir() and (d / "config.yaml").exists())
    if modeles:
        print(f"Modèles DLC trouvés dans {DEFAULT_MODELS_ROOT} :")
        for i, m in enumerate(modeles, start=1):
            print(f"  {i}. {m.name}")
        print(f"  {len(modeles) + 1}. (autre chemin)")
        choix = prompt("Modèle", default="1")
        if choix.isdigit() and 1 <= int(choix) <= len(modeles):
            return modeles[int(choix) - 1]
    return prompt_existing_path("Dossier du projet DLC", must_exist=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model-dir", type=Path, default=None,
                        help="Dossier du projet DLC. Demandé si absent.")
    parser.add_argument("--predictions", type=Path, default=None,
                        help="Fichier .h5 de prédictions à comparer "
                             "(défaut : le plus récent dans "
                             "evaluation-results/).")
    parser.add_argument("--pcutoff", type=float, default=DEFAULT_PCUTOFF,
                        help=f"Seuil de confiance (défaut : {DEFAULT_PCUTOFF}, "
                             f"celui de l'évaluation DLC).")
    parser.add_argument("--out", type=Path, default=None,
                        help="Écrit le tableau en CSV à ce chemin.")
    args = parser.parse_args()

    model_dir = choisir_modele(args.model_dir)
    pred_path = args.predictions or find_predictions(model_dir)
    if pred_path is None:
        print(f"❌ Aucun fichier de prédictions trouvé pour {model_dir.name}.",
              file=sys.stderr)
        print(diagnostic_absence(model_dir), file=sys.stderr)
        print(f"\n   Si l'évaluation n'a pas tourné :\n"
              f"     python scripts/dlc_model-training/02_train.py "
              f"--config-dir {model_dir} --eval-only", file=sys.stderr)
        sys.exit(1)

    print(f"Modèle       : {model_dir.name}")
    print(f"Prédictions  : {pred_path.name}")
    print(f"Seuil        : {args.pcutoff}")
    print()

    df_gt, df_pred = load_ground_truth(model_dir), pd.read_hdf(pred_path)
    df = per_keypoint_errors(df_gt, df_pred, args.pcutoff)

    largeur = max(len(str(k)) for k in df["keypoint"])
    entete = (f"{'keypoint'.ljust(largeur)}  {'n':>5}  {'%>seuil':>8}  "
              f"{'médiane':>8}  {'méd.seuil':>9}  {'%>25px':>7}  {'rmse':>7}")
    print(entete)
    print("-" * len(entete))
    for _, r in df.iterrows():
        if r["keypoint"] == "TOTAL":
            print("-" * len(entete))
        print(f"{str(r['keypoint']).ljust(largeur)}  {int(r['n']):>5}  "
              f"{r['pct_au_dessus_seuil']:>7.1f}%  {r['mediane']:>8.2f}  "
              f"{r['mediane_seuil']:>9.2f}  {r['pct_grossieres']:>6.1f}%  "
              f"{r['rmse']:>7.1f}")
    print()
    print("médiane   : erreur typique (px), insensible aux frames aberrantes")
    print("méd.seuil : idem, sur les seules prédictions au-dessus du seuil")
    print("%>25px    : part des prédictions franchement ailleurs")
    print("rmse      : dominée par ces quelques frames — à lire en dernier")

    # Lecture assistée : on ne commente que ce qui mérite une action.
    print()
    corps = df[df["keypoint"] != "TOTAL"]
    systematiques = corps[corps["mediane_seuil"] > 10]
    ponctuels = corps[(corps["pct_grossieres"] > 3)
                      & ~corps["keypoint"].isin(systematiques["keypoint"])]
    peu_visibles = corps[corps["pct_au_dessus_seuil"] < 50]

    if len(systematiques):
        print("❌ Faux de façon systématique (même confiant) : "
              + ", ".join(systematiques["keypoint"]))
        print("   La majorité des prédictions confiantes tombe à côté. Le "
              "cutoff ne rattrapera rien.")
    if len(ponctuels):
        print("⚠  Justes en général, mais grossièrement faux sur quelques "
              "frames : " + ", ".join(
                  f"{r.keypoint} ({r.pct_grossieres:.0f} %)"
                  for r in ponctuels.itertuples()))
    if len(peu_visibles):
        print("ℹ  Souvent sous le seuil : " + ", ".join(
            f"{r.keypoint} ({r.pct_au_dessus_seuil:.0f} %)"
            for r in peu_visibles.itertuples()))
        print("   Si leur médiane reste basse, la position est bonne et c'est "
              "seulement la confiance\n   qui est faible : baisser le seuil "
              "en aval les récupère.")

    diag = diagnostic_confusions(df_gt, df_pred)
    confus = [p for p in diag["paires"] if p["pct_sur_jumeau"] > 30]
    if confus:
        print()
        print("❌ Confusion gauche/droite :")
        for p in confus:
            print(f"   {p['keypoint']} est prédit sur son symétrique dans "
                  f"{p['pct_sur_jumeau']:.0f} % des frames "
                  f"(médiane {p['mediane_vers_jumeau']:.1f} px de lui).")
        print("   Le réseau trouve UNE patte et y pose les deux marqueurs.")
    ret = diag["retournement"]
    if ret and ret["n"]:
        print()
        print(f"⚠  Retournement tête/queue : {ret['n']} frame(s) sur "
              f"{ret['total']} — le nez prédit plus près de la base de la "
              f"queue que du nez.")
        print("   Ce sont elles qui gonflent la RMSE de nose, chin et "
              "tail_tip. À regarder :")
        for f in ret["frames"][:8]:
            print(f"     · {f}")
        if ret["n"] > 8:
            print(f"     … et {ret['n'] - 8} autre(s)")

    if not (len(systematiques) or len(ponctuels) or confus
            or (ret and ret["n"])):
        print("✅ Aucun keypoint problématique.")

    if args.out:
        df.to_csv(args.out, index=False)
        print(f"\n✓ Tableau écrit : {args.out}")


if __name__ == "__main__":
    main()
