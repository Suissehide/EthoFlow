"""Applique le modèle DLC bottom-view entraîné sur une ou plusieurs vidéos.

Outputs (h5, csv, vidéo annotée) organisés par vidéo source dans :
    <PROJECT_DIR>/result-videos/<nom_video_sans_extension>/

Avantages de cette organisation :
    - Pas de mélange avec d'autres fichiers à côté de la vidéo source
    - Pas d'écrasement entre runs sur la même vidéo
    - Tous les outputs du projet centralisés au même endroit

Pré-requis :
    - 02_train.py terminé avec succès (best_model.pkl écrit)
    - VIDEOS_TO_ANALYZE renseigné dans `_config.py`
    - conda activate dlc
"""
from __future__ import annotations

import sys
from pathlib import Path

import deeplabcut as dlc

# Import du config centralisé
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _config import (  # noqa: E402
    CONFIG,
    LABELED_VIDEO_PCUTOFF,
    MAKE_LABELED_VIDEO,
    RESULTS_DIR,
    VIDEOS_TO_ANALYZE,
)


def main() -> None:
    if not VIDEOS_TO_ANALYZE:
        print("⚠ VIDEOS_TO_ANALYZE est vide dans _config.py")
        return

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Résultats dans : {RESULTS_DIR}")
    print(f"  pcutoff vidéo annotée : {LABELED_VIDEO_PCUTOFF}")
    print(f"  vidéo annotée         : {'oui' if MAKE_LABELED_VIDEO else 'non'}\n")

    for video in VIDEOS_TO_ANALYZE:
        if not video.exists():
            print(f"⚠ Vidéo introuvable, skip : {video}")
            continue

        # Un sous-dossier dédié par vidéo, nommé par le stem
        # (ex: "970.mp4" → result-videos/970/)
        out_dir = RESULTS_DIR / video.stem
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"→ {video.name}")
        print(f"   sortie : {out_dir.relative_to(RESULTS_DIR.parent)}")

        # Inférence : produit le .h5 et le .csv dans out_dir
        # snapshot_index=-1 force le dernier snapshot par numéro d'epoch
        # (= snapshot-100.pt) plutôt que le "best" tracké pendant training,
        # qui peut être un snapshot intermédiaire pas optimal sur le test set.
        # Cf. notre eval post-training : snapshot-100 bat snapshot-best-090
        # de loin (rmse_pcutoff 4.04 vs 7.21).
        dlc.analyze_videos(
            CONFIG,
            [str(video)],
            save_as_csv=True,
            destfolder=str(out_dir),
            snapshot_index=-1,
        )

        # Vidéo annotée pour inspection visuelle (optionnelle)
        if MAKE_LABELED_VIDEO:
            dlc.create_labeled_video(
                CONFIG,
                [str(video)],
                destfolder=str(out_dir),
                pcutoff=LABELED_VIDEO_PCUTOFF,
                draw_skeleton=True,
            )
        print(f"   ✅ Terminé\n")

    print(
        "À vérifier sur la vidéo annotée :\n"
        "  - Les 12 keypoints suivent visuellement la souris en posture neutre\n"
        "  - Pendant les rearings, les pattes avant ont une likelihood basse\n"
        "    (c'est ATTENDU)\n"
        "  - Pas de jitter excessif sur tail_base et les hanches\n"
        "    (anchors VAME downstream)\n"
    )


if __name__ == "__main__":
    main()
