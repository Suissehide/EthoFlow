"""Distribution des confiances d'un .h5 d'inférence, par keypoint.

À lancer quand la vidéo annotée est vide ou clairsemée : « j'ai baissé le
pcutoff à 0.3 et je n'ai quasiment aucun point ». Ce script dit *à quel
point* le modèle est incertain, et sur quels keypoints.

La question qu'il tranche : le modèle **généralise-t-il à cette vidéo** ?
Une RMSE de 4 px à l'évaluation ne le dit pas — elle est mesurée sur les
frames labellisées, donc sur les vidéos qui ont servi à l'entraînement.
Un modèle excellent sur son training set peut être perdu sur une vidéo
tournée avec un autre éclairage, une autre caméra ou un autre zoom.

Lecture :

  · médiane > 0.6 sur la plupart des keypoints → le modèle est à l'aise,
    la vidéo ressemble au training set.
  · médiane entre 0.05 et 0.3 sur TOUS les keypoints → le modèle ne
    reconnaît pas cette vidéo. Ce n'est pas un problème de seuil : il n'y
    a rien à récupérer en baissant le pcutoff, il faut labelliser des
    frames de CETTE vidéo (étape B.6) ou vérifier qu'elle vient bien du
    même setup.
  · médiane haute sauf sur 2-3 keypoints → comportement normal, ces
    points sont occlus une partie du temps.

Note sur `05_refine_outliers.py` : la méthode `uncertain` cherche les
frames sous `p_bound` (0.01 par défaut). Si les confiances de ta vidéo
vivent entre 0.02 et 0.3, elle trouve **0 outlier** — non pas parce que
tout va bien, mais parce que le seuil est plus bas que ton pire cas.
C'est exactement la situation que ce script rend visible.

Usage :
    # Cherche le .h5 le plus récent dans result-videos/
    python scripts/dlc_model-training/inspect_predictions.py \\
        --model-dir D:/EthoFlow/models/bottomview_129_IR

    # Sur un fichier précis
    python scripts/dlc_model-training/inspect_predictions.py \\
        --h5 D:/EthoFlow/models/bottomview_129_IR/result-videos/1/1DLC_...h5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from interactive import DEFAULT_MODELS_ROOT, prompt, prompt_existing_path  # noqa: E402


SEUILS = (0.1, 0.3, 0.6, 0.9)


def likelihoods(df: pd.DataFrame) -> pd.DataFrame:
    """Extrait les colonnes de likelihood, indexées par nom de keypoint."""
    names = list(df.columns.names or [])
    if "coords" in names:
        lik = df.xs("likelihood", level="coords", axis=1)
    else:
        lik = df.loc[:, [c for c in df.columns if c[-1] == "likelihood"]]
        lik.columns = pd.MultiIndex.from_tuples(
            [c[:-1] for c in lik.columns])
    if isinstance(lik.columns, pd.MultiIndex):
        niveau = ("bodyparts" if "bodyparts" in (lik.columns.names or [])
                  else lik.columns.nlevels - 1)
        lik.columns = lik.columns.get_level_values(niveau)
    return lik.astype(float)


def resume(lik: pd.DataFrame, seuils=SEUILS) -> pd.DataFrame:
    """Une ligne par keypoint : médiane + % au-dessus de chaque seuil."""
    out = pd.DataFrame({"mediane": lik.median()})
    for s in seuils:
        out[f">{s}"] = (lik > s).mean() * 100
    return out.sort_values("mediane", ascending=False)


def trouver_h5(model_dir: Path) -> Path | None:
    """.h5 d'inférence le plus récent sous result-videos/."""
    root = model_dir / "result-videos"
    if not root.exists():
        return None
    candidats = [p for p in root.rglob("*.h5")
                 if "filtered" not in p.stem and "results" not in p.stem]
    return max(candidats, key=lambda p: p.stat().st_mtime) if candidats else None


def choisir_modele(model_dir: Path | None) -> Path:
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
        choix = prompt("Modèle", default="1")
        if choix.isdigit() and 1 <= int(choix) <= len(modeles):
            return modeles[int(choix) - 1]
    return prompt_existing_path("Dossier du projet DLC", must_exist=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model-dir", type=Path, default=None,
                        help="Dossier du projet DLC. Demandé si absent.")
    parser.add_argument("--h5", type=Path, default=None,
                        help="Fichier de prédictions à inspecter (défaut : "
                             "le plus récent dans result-videos/).")
    args = parser.parse_args()

    h5 = args.h5
    if h5 is None:
        h5 = trouver_h5(choisir_modele(args.model_dir))
    if h5 is None:
        print("❌ Aucun .h5 d'inférence trouvé. Lance d'abord 03_apply.py.",
              file=sys.stderr)
        sys.exit(1)

    lik = likelihoods(pd.read_hdf(h5))
    tab = resume(lik)

    print(f"Fichier : {Path(h5).name}")
    print(f"Frames  : {len(lik)}")
    print()
    print(tab.round(2).to_string())

    print()
    verdict(tab, lik)


def paires_gauche_droite(keypoints) -> list[tuple[str, str]]:
    """Apparie les keypoints symétriques (front_paw_left/right, etc.)."""
    paires = []
    for k in keypoints:
        s = str(k)
        if s.endswith("_left"):
            jumeau = s[: -len("_left")] + "_right"
            if jumeau in set(str(x) for x in keypoints):
                paires.append((s, jumeau))
    return paires


def verdict(tab: pd.DataFrame, lik: pd.DataFrame) -> None:
    """Commente le tableau — par groupe de keypoints, pas globalement.

    Une médiane globale n'a pas de sens ici : un modèle peut être bon sur
    l'axe du corps et aveugle sur la queue. Les actions à prendre ne sont
    pas les mêmes, donc on les sépare.
    """
    med_col, seuil_col = "mediane", ">0.3"

    # Asymétrie gauche/droite : l'anatomie est symétrique, donc un écart
    # marqué entre les deux côtés vient des labels, pas de l'animal.
    asym = []
    for g, d in paires_gauche_droite(tab.index):
        if g not in tab.index or d not in tab.index:
            continue
        a, b = tab.loc[g, seuil_col], tab.loc[d, seuil_col]
        haut, bas = max(a, b), min(a, b)
        if haut > 10 and haut > 2 * max(bas, 0.5):
            asym.append((g, a, d, b))
    concernes_asym = {k for g, _, d, _ in asym for k in (g, d)}

    morts = tab[(tab[med_col] < 0.15) | (tab[seuil_col] < 5)]
    faibles = tab[(tab[med_col] >= 0.15) & (tab[med_col] < 0.45)
                  & (tab[seuil_col] >= 5)]
    bons = tab[tab[med_col] >= 0.45]
    # Un keypoint muet dont le symétrique fonctionne n'est pas invisible :
    # c'est un problème d'annotation. Le retirer masquerait la cause.
    a_retirer = [k for k in morts.index if str(k) not in concernes_asym]

    if len(bons):
        print(f"✅ Fiables ({len(bons)}) : {', '.join(map(str, bons.index))}")
    if len(faibles):
        print(f"⚠  Incertains ({len(faibles)}) : "
              f"{', '.join(map(str, faibles.index))}")
        print("   Récupérables : ajoute des frames dans les situations où "
              "ils échouent (étape B.6).")
    if len(morts):
        print(f"❌ Jamais détectés ({len(morts)}) : "
              f"{', '.join(map(str, morts.index))}")
    if a_retirer:
        print(f"   Invisibles sur cette vue (aucun symétrique qui "
              f"fonctionne) : {', '.join(map(str, a_retirer))}")
        print("   Aucun seuil ne les récupère — retire-les des features "
              "VAME :")
        print("     python scripts/filter_keypoints.py")

    if asym:
        print()
        print("⚠  Asymétrie gauche/droite :")
        for g, a, d, b in asym:
            print(f"     {g} {a:.0f}% au-dessus de 0.3  vs  {d} {b:.0f}%")
        print("   L'animal est symétrique : cet écart vient des "
              "annotations, pas de la souris.")
        print("   Lance l'audit L/R avant d'ajouter quoi que ce soit au "
              "training set :")
        print("     python scripts/dlc_model-training/06_check_labels.py")

    # Conséquence directe sur l'étape 6b — le seuil par défaut y est 0.70.
    print()
    utilisables = int(((lik > 0.7).mean() * 100 > 20).sum())
    if utilisables < len(tab) / 2:
        print(f"ℹ  Pour l'étape 6b : seuls {utilisables} keypoint(s) sur "
              f"{len(tab)} dépassent 0.70 plus de 20 % du temps.")
        print("   Le seuil par défaut de prepare_vame_input_custom.py (0.70) "
              "mettrait presque tout en NaN.")
        print("   Baisse-le (--likelihood-threshold 0.3) OU réduis le jeu de "
              "keypoints aux plus fiables.")


if __name__ == "__main__":
    main()
