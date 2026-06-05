"""Applique le modèle DLC bottom-view entraîné sur une ou plusieurs vidéos
et produit la vidéo annotée pour inspection visuelle.

Pré-requis :
    - 02_train.py terminé avec succès (best_model présent)
    - conda activate dlc

Pour brancher ce modèle sur le pipeline EthoFlow ensuite :
    1. Renseigne `dlc_project_config: <chemin>/config.yaml` dans
       `configs/pipeline_config.yaml`
    2. Lance `python scripts/run_dlc_inference.py <session> --mode custom`

Ce script-ci est utile en standalone pour :
    - Vérifier visuellement la qualité du modèle après chaque retrain
    - Inférer sur des vidéos qui ne sont pas (encore) dans une session
      EthoFlow propre
"""
from __future__ import annotations

import deeplabcut as dlc


# ----------------------------------------------------------------------
# À ÉDITER
# ----------------------------------------------------------------------

CONFIG = "/chemin/absolu/vers/souris-bottomview-Leo-2026-06-XX/config.yaml"

# Une ou plusieurs vidéos à analyser. Chemins absolus de préférence.
VIDEOS = [
    "/chemin/vers/ta_video_pilote.mp4",
    # "/chemin/vers/autre_video.mp4",
]


# ----------------------------------------------------------------------
# Pipeline
# ----------------------------------------------------------------------

def main() -> None:
    print(f"Inférence DLC sur {len(VIDEOS)} vidéo(s)...")
    dlc.analyze_videos(CONFIG, VIDEOS, save_as_csv=True)
    print("✅ Inférence terminée — .h5 et .csv écrits à côté des vidéos sources.\n")

    print("Génération des vidéos annotées (pour inspection visuelle)...")
    dlc.create_labeled_video(CONFIG, VIDEOS)
    print("✅ Vidéos annotées générées.\n")

    print(
        "À vérifier sur la vidéo annotée :\n"
        "  - Les 12 keypoints suivent visuellement la souris en posture neutre\n"
        "  - Pendant les rearings, les pattes avant disparaissent ou ont une\n"
        "    likelihood basse (c'est ATTENDU et c'est ce qu'on exploitera plus\n"
        "    tard pour la détection automatique du rearing)\n"
        "  - Pas de jitter excessif sur les pattes arrière et tail_base\n"
        "    (anchors pour l'égocentrage VAME downstream)\n"
    )


if __name__ == "__main__":
    main()
