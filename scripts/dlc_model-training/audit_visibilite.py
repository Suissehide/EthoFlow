"""Repère les labels posés sur une partie du corps qu'on ne voit pas.

En vue de dessous rétro-éclairée, la souris est une silhouette noire sans
détail interne. Un membre n'est visible que lorsqu'il **dépasse du
contour**. Une patte arrière repliée sous le ventre est fondue dans la
masse : la labelliser revient à deviner sa position.

Ces labels devinés sont nocifs. Le réseau ne peut pas apprendre une cible
qui ne correspond à rien dans l'image ; il se rabat sur la seule chose qui
ressemble à une patte — celle qui dépasse — et y pose les deux marqueurs.
C'est exactement le symptôme mesuré sur bottomview_129_IR : les deux
pattes arrière prédites au même pixel, sur la patte visible.

Mesure : pour chaque label, sa **profondeur** dans la silhouette, en
pixels jusqu'au fond le plus proche (0 = sur le bord ou dehors). Une patte
visible a une profondeur faible ; un label à 20 px de profondeur est en
plein dans la masse noire, donc deviné.

Par défaut le script ne modifie rien : il affiche la répartition et écrit
la liste des labels suspects. Avec `--appliquer`, il retire ces labels
(les met à « non annoté ») après avoir sauvegardé chaque fichier.

Usage :
    # Constat seulement
    python scripts/dlc_model-training/audit_visibilite.py \\
        --model-dir D:/EthoFlow/models/bottomview_129_IR

    # Retirer les labels de pattes arrière enfouis à plus de 15 px
    python scripts/dlc_model-training/audit_visibilite.py \\
        --model-dir <...> --appliquer

    # Autres keypoints, autre seuil
    python scripts/dlc_model-training/audit_visibilite.py \\
        --model-dir <...> --keypoints hind_paw_left hind_paw_right \\
        front_paw_left front_paw_right --profondeur 12
"""
from __future__ import annotations

import argparse
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from keypoint_errors import choisir_modele  # noqa: E402

KEYPOINTS_DEFAUT = ["hind_paw_left", "hind_paw_right"]
PROFONDEUR_DEFAUT = 15.0


def carte_profondeur(gris: np.ndarray) -> np.ndarray | None:
    """Profondeur de chaque pixel dans la silhouette (0 hors silhouette).

    Otsu sépare la souris sombre du plancher clair ; on garde la plus
    grande composante (les coins et repères de l'arène peuvent aussi être
    sombres), puis une transformée de distance donne, pour chaque pixel
    de la souris, la distance au pixel de fond le plus proche.
    """
    import cv2

    _, masque = cv2.threshold(gris, 0, 255,
                              cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    n, etiq, stats, _ = cv2.connectedComponentsWithStats(masque, 8)
    if n <= 1:
        return None
    plus_grande = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    souris = (etiq == plus_grande).astype(np.uint8)
    return cv2.distanceTransform(souris, cv2.DIST_L2, 5)


def chemin_image(model_dir: Path, index) -> Path:
    """Chemin du PNG d'une ligne de CollectedData, quel que soit le format d'index."""
    if isinstance(index, tuple):
        return model_dir.joinpath(*[str(p) for p in index])
    return model_dir / str(index).replace("\\", "/")


def profondeurs(df: pd.DataFrame, model_dir: Path,
                keypoints: list[str]) -> pd.DataFrame:
    """Une ligne par (frame, keypoint) labellisé : sa profondeur."""
    import cv2

    scorer = df.columns.get_level_values(0)[0]
    lignes = []
    for idx in df.index:
        img_path = chemin_image(model_dir, idx)
        gris = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if gris is None:
            continue
        carte = carte_profondeur(gris)
        if carte is None:
            continue
        for bp in keypoints:
            if (scorer, bp, "x") not in df.columns:
                continue
            x, y = df.at[idx, (scorer, bp, "x")], df.at[idx, (scorer, bp, "y")]
            if pd.isna(x) or pd.isna(y):
                continue
            xi, yi = int(round(x)), int(round(y))
            if not (0 <= yi < carte.shape[0] and 0 <= xi < carte.shape[1]):
                prof = 0.0
            else:
                prof = float(carte[yi, xi])
            lignes.append({"index": idx, "image": str(img_path),
                           "keypoint": bp, "profondeur": prof})
    return pd.DataFrame(lignes)


def resume(prof: pd.DataFrame, seuil: float) -> None:
    print(f"{'keypoint':<18} {'labels':>7} {'médiane':>8} "
          f"{'enfouis >' + str(int(seuil)) + 'px':>15}")
    for bp, g in prof.groupby("keypoint", sort=False):
        enfouis = (g["profondeur"] > seuil).sum()
        print(f"{bp:<18} {len(g):>7} {g['profondeur'].median():>7.1f}  "
              f"{enfouis:>6} ({100 * enfouis / len(g):4.1f} %)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model-dir", type=Path, default=None,
                        help="Dossier du projet DLC. Demandé si absent.")
    parser.add_argument("--keypoints", nargs="+", default=KEYPOINTS_DEFAUT,
                        help="Keypoints à auditer (défaut : pattes arrière).")
    parser.add_argument("--profondeur", type=float, default=PROFONDEUR_DEFAUT,
                        help=f"Profondeur (px) au-delà de laquelle un label "
                             f"est jugé deviné (défaut {PROFONDEUR_DEFAUT}).")
    parser.add_argument("--appliquer", action="store_true",
                        help="Retire les labels enfouis (sauvegarde d'abord "
                             "chaque CollectedData modifié).")
    args = parser.parse_args()

    try:
        import cv2  # noqa: F401
    except ImportError:
        print("❌ OpenCV requis : conda activate dlc", file=sys.stderr)
        sys.exit(1)

    model_dir = choisir_modele(args.model_dir)
    fichiers = sorted((model_dir / "labeled-data").glob("*/CollectedData_*.h5"))
    if not fichiers:
        print(f"❌ Aucun CollectedData dans {model_dir / 'labeled-data'}",
              file=sys.stderr)
        sys.exit(1)

    print(f"Modèle    : {model_dir.name}")
    print(f"Keypoints : {', '.join(args.keypoints)}")
    print(f"Seuil     : profondeur > {args.profondeur:.0f} px = deviné\n")

    toutes, par_fichier = [], {}
    for h5 in fichiers:
        df = pd.read_hdf(h5)
        p = profondeurs(df, model_dir, args.keypoints)
        if len(p):
            toutes.append(p)
            par_fichier[h5] = (df, p)

    if not toutes:
        print("❌ Aucune image lisible — vérifie que les PNG de labeled-data "
              "sont présents.", file=sys.stderr)
        sys.exit(1)
    tout = pd.concat(toutes, ignore_index=True)
    resume(tout, args.profondeur)

    suspects = tout[tout["profondeur"] > args.profondeur]
    liste = model_dir / "labels_enfouis.csv"
    suspects.sort_values("profondeur", ascending=False).to_csv(liste, index=False)
    print(f"\n✓ Liste des {len(suspects)} label(s) suspect(s) : {liste}")
    print("  Triée par profondeur décroissante : les premiers sont les plus "
          "sûrement devinés.")

    if not args.appliquer:
        print("\nRien n'a été modifié. Vérifie quelques images de la liste "
              "(les plus profondes en tête),\npuis relance avec --appliquer "
              "pour retirer ces labels.")
        return

    horodatage = datetime.now().strftime("%Y%m%d-%H%M%S")
    n_retires = 0
    for h5, (df, p) in par_fichier.items():
        a_retirer = p[p["profondeur"] > args.profondeur]
        if a_retirer.empty:
            continue
        sauvegarde = h5.with_name(f"{h5.stem}.avant-audit-{horodatage}.h5")
        shutil.copy2(h5, sauvegarde)
        scorer = df.columns.get_level_values(0)[0]
        for _, r in a_retirer.iterrows():
            for c in ("x", "y"):
                df.at[r["index"], (scorer, r["keypoint"], c)] = np.nan
        df.to_hdf(h5, key="df_with_missing", mode="w")
        df.to_csv(h5.with_suffix(".csv"))
        n_retires += len(a_retirer)
        print(f"  {h5.parent.name}/{h5.name} : {len(a_retirer)} label(s) "
              f"retiré(s) — sauvegarde {sauvegarde.name}")

    print(f"\n✅ {n_retires} label(s) retiré(s).")
    print("   Vérifie la couverture puis réentraîne :")
    print("     python scripts/dlc_model-training/06_check_labels.py "
          "--skip-images")
    print("     python scripts/dlc_model-training/02_train.py --reset "
          "--crop-size 768 --batch-size 3")


if __name__ == "__main__":
    main()
