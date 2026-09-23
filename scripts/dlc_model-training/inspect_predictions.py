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

    # Verdict : on compare la médiane GLOBALE, pas keypoint par keypoint —
    # un modèle qui ne reconnaît pas la vidéo est perdu partout à la fois.
    med = float(lik.stack().median())
    print()
    if med > 0.6:
        print(f"✅ Confiance médiane {med:.2f} — le modèle reconnaît cette "
              f"vidéo.")
    elif med > 0.3:
        print(f"⚠  Confiance médiane {med:.2f} — utilisable, mais le modèle "
              f"hésite.")
        print("   Ajoute des frames de cette vidéo au training set (B.6).")
    else:
        print(f"❌ Confiance médiane {med:.2f} — le modèle ne reconnaît pas "
              f"cette vidéo.")
        print("   Baisser le pcutoff n'y changera rien : il n'y a pas de "
              "bonne prédiction cachée sous le seuil.")
        print("   Vérifie d'abord que cette vidéo vient du MÊME setup que "
              "tes frames labellisées")
        print("   (caméra, objectif, éclairage IR, hauteur, résolution). "
              "Si oui, il faut labelliser")
        print("   des frames de cette vidéo — le training set ne couvre pas "
              "ce qu'elle montre.")


if __name__ == "__main__":
    main()
