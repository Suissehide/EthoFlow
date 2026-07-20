"""Audit géométrique des labels pour détecter les inversions L/R des pattes.

Pourquoi ce script existe :

    En bottom-view IR, distinguer left/right paw est ambigu quand la souris
    pivote — "left" peut se référer à "côté gauche de l'écran" ou "côté gauche
    de la souris" (anatomique), et ces deux conventions s'inversent selon
    l'orientation. Même 10-15 % de frames mal labellisées suffisent pour que
    le modèle apprenne une position moyenne entre L et R et finisse par coller
    les deux markers sur la même patte au moment de l'inférence.

    Le symptôme final : à pcutoff bas, hind_paw_left et hind_paw_right
    apparaissent sur la même patte. Likelihood ~0.3 sur les hind paws. Le
    modèle SAIT localiser (rmse_pcutoff = 4 px à 0.7) mais NE SAIT PAS choisir
    un côté.

Comment on diagnostique :

    1. dlc.check_labels génère les images annotées (markers superposés sur les
       PNG) dans <PROJECT>/labeled-data/<video>_labeled/.

    2. Pour chaque frame labellisée, on calcule un produit vectoriel 2D :
       l'axe corps va de tail_base à nose, et chaque paw est soit "à gauche"
       soit "à droite" de cet axe. Le signe du cross-product (ax * py - ay * px)
       indique de quel côté.

    3. On NE postule PAS la "bonne" convention. À la place : pour chaque paw,
       on regarde le signe DOMINANT (la mode statistique sur 360 frames). Si
       l'expérimentateur a été cohérent à 90 %, les 10 % de frames du côté
       minoritaire sont les outliers à inspecter — pas besoin de savoir si la
       "vraie" convention est anatomique ou écran.

    4. Score de suspicion par frame = nombre de paws (0-4) du mauvais côté
       par rapport à la mode. Sorted desc → top des frames à vérifier.

Pré-requis :
    - Toutes les frames labellisées (config.yaml à jour)
    - conda activate dlc

Workflow :
    1. python scripts/dlc_model-training/06_check_labels.py
    2. Inspecte les top frames suspectes via les PNG dans
       labeled-data/<video>_labeled/.
    3. Re-labellise les erreurs via Project Manager.
    4. Nettoie les caches et re-train.
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _config import CONFIG, PROJECT_DIR  # noqa: E402


PAWS = [
    "front_paw_left",
    "front_paw_right",
    "hind_paw_left",
    "hind_paw_right",
]


def cross_sign_vec(
    paw_x: np.ndarray,
    paw_y: np.ndarray,
    tail_x: np.ndarray,
    tail_y: np.ndarray,
    nose_x: np.ndarray,
    nose_y: np.ndarray,
) -> np.ndarray:
    """Signe du produit vectoriel 2D (axe tail→nose) × (vecteur tail→paw).

    Retourne +1, -1, 0, ou NaN par frame. Le signe en soi n'a pas de
    sémantique fixée (dépend du sens y-down de l'image) — on s'en sert juste
    comme indicateur "côté A / côté B" et on calibre via la mode dans main().
    """
    ax = nose_x - tail_x
    ay = nose_y - tail_y
    px = paw_x - tail_x
    py = paw_y - tail_y
    cross = ax * py - ay * px
    # np.sign propage NaN — c'est ce qu'on veut
    result = np.sign(cross)
    # Frames pile-poil sur l'axe (norme ~0) → side indéterminée
    result = np.where(np.abs(cross) < 1e-6, 0, result)
    return result


def main() -> None:
    # ------------------------------------------------------------------
    # 1) Génère les images annotées (markers superposés)
    # ------------------------------------------------------------------
    import deeplabcut as dlc

    print("Génération des images annotées via dlc.check_labels...")
    print("  Sortie : <PROJECT>/labeled-data/<video>_labeled/\n")
    try:
        dlc.check_labels(CONFIG, visualizeindividuals=False)
    except TypeError:
        # Signature plus ancienne
        dlc.check_labels(CONFIG)
    print("✅ Images générées.\n")

    # ------------------------------------------------------------------
    # 2) Charge tous les CollectedData et calcule les sides par paw
    # ------------------------------------------------------------------
    all_rows: list[dict] = []

    for vdir in sorted((PROJECT_DIR / "labeled-data").iterdir()):
        if not vdir.is_dir() or vdir.name.endswith("_labeled"):
            continue
        # DLC nomme le fichier de labels manuels CollectedData_<EXPERIMENTER>.h5
        # → cherche celui qui matche notre config, sinon le premier disponible
        # (utile si tu as repris un projet DLC créé par quelqu'un d'autre).
        from _config import EXPERIMENTER
        h5 = vdir / f"CollectedData_{EXPERIMENTER}.h5"
        if not h5.exists():
            fallback = sorted(vdir.glob("CollectedData_*.h5"))
            if not fallback:
                continue
            h5 = fallback[0]

        df = pd.read_hdf(h5)
        scorer = df.columns.get_level_values("scorer")[0]

        def col(bp: str, c: str) -> np.ndarray:
            return df[(scorer, bp, c)].values

        tail_x, tail_y = col("tail_base", "x"), col("tail_base", "y")
        nose_x, nose_y = col("nose", "x"), col("nose", "y")

        sides_per_paw = {
            paw: cross_sign_vec(
                col(paw, "x"), col(paw, "y"),
                tail_x, tail_y, nose_x, nose_y,
            )
            for paw in PAWS
        }

        for i, idx in enumerate(df.index):
            frame_name = idx[-1] if isinstance(idx, tuple) else str(idx)
            row = {"video": vdir.name, "frame": frame_name}
            for paw in PAWS:
                row[paw] = sides_per_paw[paw][i]
            all_rows.append(row)

    df_all = pd.DataFrame(all_rows)
    print(f"Frames labellisées chargées : {len(df_all)}\n")

    # ------------------------------------------------------------------
    # 3) Calibration : pour chaque paw, side dominante = mode statistique
    # ------------------------------------------------------------------
    print("=== Calibration : side dominante par paw ===")
    expected: dict[str, int] = {}
    for paw in PAWS:
        signs = df_all[paw].dropna()
        signs = signs[signs != 0]
        if len(signs) == 0:
            print(f"  {paw}: aucune frame avec tail_base + nose + {paw} labellisés")
            continue
        counts = Counter(signs.astype(int))
        most, n_most = counts.most_common(1)[0]
        expected[paw] = int(most)
        match_pct = n_most / len(signs) * 100
        n_minority = len(signs) - n_most
        print(
            f"  {paw}: side dominante = {int(most):+d}  "
            f"({n_most}/{len(signs)} = {match_pct:.0f}%)  "
            f"→ {n_minority} frame(s) du côté minoritaire"
        )
    print()

    # ------------------------------------------------------------------
    # 4) Score par frame = nb de paws du mauvais côté
    # ------------------------------------------------------------------
    score = np.zeros(len(df_all), dtype=int)
    wrong_paws_per_frame: list[list[str]] = [[] for _ in range(len(df_all))]
    for paw, exp_side in expected.items():
        sides = df_all[paw].values
        wrong_mask = (~np.isnan(sides)) & (sides != 0) & (sides != exp_side)
        score += wrong_mask.astype(int)
        for i, w in enumerate(wrong_mask):
            if w:
                wrong_paws_per_frame[i].append(paw)
    df_all["score"] = score
    df_all["wrong_paws"] = [", ".join(p) for p in wrong_paws_per_frame]

    # ------------------------------------------------------------------
    # 5) Stats
    # ------------------------------------------------------------------
    print("=== Distribution des scores de suspicion ===")
    counts = df_all["score"].value_counts().sort_index()
    for s, c in counts.items():
        marker = "  ← suspect" if s >= 1 else ""
        print(f"  score {s}: {c} frames{marker}")
    print()

    n_susp_1 = (df_all["score"] >= 1).sum()
    n_susp_2 = (df_all["score"] >= 2).sum()
    print(
        f"Frames avec >= 1 paw du mauvais côté : {n_susp_1} "
        f"({n_susp_1 / len(df_all) * 100:.1f}%)"
    )
    print(
        f"Frames avec >= 2 paws du mauvais côté : {n_susp_2} "
        f"({n_susp_2 / len(df_all) * 100:.1f}%)"
    )
    print()

    # ------------------------------------------------------------------
    # 6) Top 30 frames à inspecter
    # ------------------------------------------------------------------
    df_susp = df_all.sort_values("score", ascending=False)
    cols_export = ["video", "frame", "score", "wrong_paws"]
    print("=== Top 30 frames les plus suspectes ===")
    if (df_susp["score"] > 0).any():
        print(df_susp[df_susp["score"] > 0][cols_export].head(30).to_string(index=False))
    else:
        print("  Aucune frame suspecte. Si le problème L/R persiste à\n"
              "  l'inférence, la cause est ailleurs (ambiguïté visuelle\n"
              "  intrinsèque sur les hind paws, pas un drift de labels).")
    print()

    # ------------------------------------------------------------------
    # 7) CSV export
    # ------------------------------------------------------------------
    out_csv = PROJECT_DIR / "label_audit_suspicion.csv"
    df_susp[cols_export].to_csv(out_csv, index=False)
    print(f"✅ CSV complet : {out_csv}")

    print(
        "\n=== Étapes suivantes ===\n"
        "1. Pour les frames score >= 1, ouvre l'image annotée :\n"
        "     <PROJECT>/labeled-data/<video>_labeled/<frame>.png\n"
        "   (générée par check_labels juste avant)\n"
        "\n"
        "2. Vérifie visuellement contre TA convention L/R. Le script ne sait\n"
        "   pas quelle convention est la bonne — il signale juste les outliers.\n"
        "\n"
        "3. Pour les frames effectivement inversées :\n"
        "     python -m deeplabcut\n"
        "   → Project Manager → onglet Label frames → corrige.\n"
        "\n"
        "4. Nettoie les caches AVANT de re-train (sinon DLC réutilise\n"
        "   l'ancien dataset) :\n"
        "     Remove-Item -Recurse -Force\n"
        "       \"<PROJECT_DIR>\\training-datasets\"\n"
        "     Remove-Item -Recurse -Force\n"
        "       \"<PROJECT_DIR>\\dlc-models-pytorch\"\n"
        "\n"
        "5. python scripts\\dlc_model-training\\02_train.py\n"
    )


if __name__ == "__main__":
    main()
