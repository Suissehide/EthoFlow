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

import argparse
import sys
from pathlib import Path

# Insère le dossier du script en tête de sys.path pour trouver _load_config
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _load_config import add_config_dir_arg, completer_videos, load_config  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    add_config_dir_arg(parser)
    parser.add_argument(
        "--no-labeled-video", action="store_true",
        help="N'écrit pas de vidéo annotée : seulement les prédictions .h5 "
             "et .csv. Plus rapide et bien plus léger. Pour en produire une "
             "seule ensuite : create_labeled_video.py --video <chemin>.",
    )
    parser.add_argument(
        "--videos", nargs="+", default=None, metavar="CHEMIN",
        help="Analyse uniquement ces vidéos, au lieu de toutes celles "
             "déclarées dans _config.py.",
    )
    args = parser.parse_args()
    load_config(args)

    import deeplabcut as dlc  # noqa: E402
    from _config import (  # noqa: E402
        CONFIG, LABELED_VIDEO_PCUTOFF, MAKE_LABELED_VIDEO,
        PROJECT_DIR, RESULTS_DIR, VIDEOS_TO_ANALYZE,
    )

    videos = ([Path(v) for v in args.videos] if args.videos
              else completer_videos(VIDEOS_TO_ANALYZE, "VIDEOS_TO_ANALYZE"))
    faire_video = MAKE_LABELED_VIDEO and not args.no_labeled_video
    if not videos:
        print("⚠ Aucune vidéo à analyser : VIDEOS_TO_ANALYZE est vide dans\n"
              "   _config.py, et aucune autre liste n'en déclare.")
        return

    # Date du modèle : toute prédiction antérieure vient d'un modèle
    # précédent, même si son nom de fichier est identique.
    snapshots = sorted((PROJECT_DIR / "dlc-models-pytorch").rglob("snapshot-*.pt"),
                       key=lambda p: p.stat().st_mtime) \
        if (PROJECT_DIR / "dlc-models-pytorch").exists() else []
    snapshot_mtime = snapshots[-1].stat().st_mtime if snapshots else 0.0
    if snapshots:
        import datetime as _dt
        quand = _dt.datetime.fromtimestamp(snapshot_mtime).strftime(
            "%d/%m %H:%M")
        print(f"Modèle daté du {quand} ({snapshots[-1].name})")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Résultats dans : {RESULTS_DIR}")
    print(f"  pcutoff vidéo annotée : {LABELED_VIDEO_PCUTOFF}")
    print(f"  vidéo annotée         : {'oui' if faire_video else 'non'}\n")

    for video in videos:
        if not video.exists():
            print(f"⚠ Vidéo introuvable, skip : {video}")
            continue

        # Un sous-dossier dédié par vidéo, nommé par le stem
        # (ex: "970.mp4" → result-videos/970/)
        out_dir = RESULTS_DIR / video.stem
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"→ {video.name}")
        print(f"   sortie : {out_dir.relative_to(RESULTS_DIR.parent)}")

        # Piège silencieux : DLC nomme ses sorties d'après le snapshot
        # (`<video>DLC_..._snapshot_best-170.h5`) et saute les vidéos
        # « déjà analysées ». Après un réentraînement, si le nouveau
        # meilleur snapshot retombe sur le même numéro d'epoch — ce qui
        # arrive très bien —, le nom de fichier est identique et DLC
        # conserve les prédictions de l'ANCIEN modèle. On croit alors
        # mesurer le nouveau modèle et on lit les chiffres du précédent.
        perimes = [p for p in out_dir.glob("*.h5")
                   if p.stat().st_mtime < snapshot_mtime]
        if perimes:
            print(f"   ⚠ {len(perimes)} prédiction(s) antérieure(s) au "
                  f"modèle actuel — supprimées pour forcer le recalcul :")
            for p in perimes:
                print(f"       · {p.name}")
                p.unlink()
                for jumeau in out_dir.glob(p.stem + ".*"):
                    jumeau.unlink()

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
        if faire_video:
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
