"""Régénère la vidéo annotée à un pcutoff différent, SANS relancer l'inférence.

L'inférence DLC (`analyze_videos`) est lente parce qu'elle passe le modèle
sur toutes les frames de la vidéo. Une fois faite, elle produit un `.h5` de
prédictions avec les scores de confiance par keypoint et par frame.

`create_labeled_video` lit ce `.h5` et redessine simplement les keypoints
au-dessus de chaque frame, ne gardant que ceux dont la likelihood >= pcutoff.
C'est rapide (~30 s à 1 min pour une vidéo de 20 min).

Donc si tu veux comparer plusieurs seuils de confiance pour le QC visuel,
inutile de relancer `03_apply.py` à chaque fois — ce script-ci suffit.

DLC encode le pcutoff dans le nom du fichier sorti (ex: `..._p10_labeled.mp4`
pour pcutoff=0.1), donc les versions à plusieurs seuils cohabitent dans le
même dossier sans s'écraser.

Usage :
    python scripts/dlc_bottomview/create_labeled_video.py --pcutoff 0.1
    python scripts/dlc_bottomview/create_labeled_video.py --pcutoff 0.4
    python scripts/dlc_bottomview/create_labeled_video.py --pcutoff 0.6

    # Vidéo spécifique (sinon toutes les VIDEOS_TO_ANALYZE de _config.py) :
    python scripts/dlc_bottomview/create_labeled_video.py \
        --pcutoff 0.1 --video "D:/chemin/autre_video.mp4"

Pré-requis :
    - 03_apply.py a déjà tourné et produit des .h5 dans
      <PROJECT_DIR>/result-videos/<nom_video>/
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import deeplabcut as dlc

# Import du config centralisé
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _config import CONFIG, RESULTS_DIR, VIDEOS_TO_ANALYZE  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Régénère la vidéo annotée à un pcutoff différent "
                    "(réutilise le .h5 existant, pas de nouvelle inférence)."
    )
    parser.add_argument(
        "--pcutoff",
        type=float,
        default=0.1,
        help="Seuil de confiance pour l'affichage (défaut: 0.1 = debug). "
             "Valeurs utiles : 0.1 (debug, tout afficher), 0.4 (intermédiaire), "
             "0.6 (final, propre).",
    )
    parser.add_argument(
        "--video",
        type=str,
        default=None,
        help="Chemin vers une vidéo spécifique. Si omis, traite toutes les "
             "VIDEOS_TO_ANALYZE de _config.py.",
    )
    args = parser.parse_args()

    if args.video:
        videos = [Path(args.video)]
    else:
        videos = list(VIDEOS_TO_ANALYZE)

    if not videos:
        print("⚠ Aucune vidéo à traiter (VIDEOS_TO_ANALYZE vide).")
        sys.exit(1)

    print(f"Régénération vidéo annotée à pcutoff={args.pcutoff}")
    print(f"({len(videos)} vidéo(s) à traiter)\n")

    for video in videos:
        if not video.exists():
            print(f"⚠ skip (introuvable) : {video}")
            continue

        # Le .h5 attendu est dans <RESULTS_DIR>/<stem>/, produit par 03_apply.py
        out_dir = RESULTS_DIR / video.stem
        h5_files = list(out_dir.glob("*.h5"))
        if not h5_files:
            print(
                f"⚠ skip : pas de .h5 dans {out_dir}\n"
                f"   → Lance d'abord `03_apply.py` pour produire les prédictions."
            )
            continue

        print(f"→ {video.name}")
        print(f"   .h5 réutilisé : {h5_files[0].name}")

        dlc.create_labeled_video(
            CONFIG,
            [str(video)],
            destfolder=str(out_dir),
            pcutoff=args.pcutoff,
            draw_skeleton=True,
        )

        # Affiche le nom du fichier généré (le plus récent dans out_dir)
        new_videos = sorted(
            out_dir.glob("*_labeled.mp4"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if new_videos:
            print(f"   ✅ {new_videos[0].name}\n")


if __name__ == "__main__":
    main()
