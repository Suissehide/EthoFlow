"""Crée les entrées de session pour les vidéos bottom-view.

Le pipeline ethoflow s'attend à ce que chaque vidéo ait son propre dossier
data/raw/<session_id>/metadata.yaml pointant vers son source_video. Pour
des dossiers contenant des vidéos déjà tournées (pas issues du sync Excel
des sessions topview), ce script génère les entrées en lot.

Pour bottom-view, pas de bloc `arenes:` — c'est un seul animal par vidéo,
DLC voit la souris entière, pas besoin de splitter. La metadata est
minimaliste.

Convention de nommage des sessions :
    <PREFIX>-<stem-de-la-video>
    e.g. : BV-970, BV-971, ...

Usage :
    python scripts/dlc_bottomview/create_bottomview_sessions.py \\
        --in-dir "E:\\data\\bottom_view\\08062026" \\
        --in-dir "E:\\data\\bottom_view\\autre-dossier" \\
        --prefix BV

    # Dry-run pour voir ce qui serait créé :
    python scripts/dlc_bottomview/create_bottomview_sessions.py \\
        --in-dir "E:\\data\\bottom_view\\08062026" \\
        --dry-run
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent.parent
RAW_DIR = ROOT / "data" / "raw"


def build_metadata(session_id: str, source_video: Path) -> dict:
    """Schéma minimaliste pour bottom-view single-animal."""
    return {
        "session_id": session_id,
        "project": "BottomView",
        "source_video": str(source_video),
        "notes": "Bottom-view IR, single animal per video.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--in-dir", action="append", required=True, type=Path,
        help="Dossier contenant les .mp4. Répétable pour plusieurs dossiers.",
    )
    parser.add_argument(
        "--prefix", default="BV",
        help="Préfixe du session_id (défaut: BV → session BV-970 pour 970.mp4)",
    )
    parser.add_argument(
        "--video-ext", default="mp4",
        help="Extension à matcher (défaut: mp4)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="N'écrit rien, affiche juste ce qui serait créé",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Écrase les metadata existants (sinon skip)",
    )
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    all_videos: list[Path] = []
    for d in args.in_dir:
        if not d.exists():
            print(f"⚠ Dossier introuvable, skip : {d}")
            continue
        videos = sorted(d.glob(f"*.{args.video_ext}"))
        print(f"  {d}: {len(videos)} vidéo(s)")
        all_videos.extend(videos)

    if not all_videos:
        print("\n❌ Aucune vidéo trouvée.")
        sys.exit(1)

    print(f"\nTotal : {len(all_videos)} vidéos\n")

    n_created = 0
    n_skipped = 0
    for video in all_videos:
        session_id = f"{args.prefix}-{video.stem}"
        session_dir = RAW_DIR / session_id
        metadata_path = session_dir / "metadata.yaml"

        if metadata_path.exists() and not args.overwrite:
            print(f"  {session_id}: déjà existant, skip")
            n_skipped += 1
            continue

        metadata = build_metadata(session_id, video)

        if args.dry_run:
            print(f"  [dry] {session_id} → {video.name}")
        else:
            session_dir.mkdir(parents=True, exist_ok=True)
            with open(metadata_path, "w") as f:
                yaml.safe_dump(metadata, f, allow_unicode=True, sort_keys=False)
            print(f"  ✓ {session_id} → {video.name}")
        n_created += 1

    print(
        f"\n{'(dry-run)' if args.dry_run else ''}"
        f"{n_created} session(s) créée(s), {n_skipped} existante(s) skip"
    )
    if not args.dry_run:
        print(
            "\nÉtape suivante :\n"
            "  python scripts/run_dlc_inference.py --all --mode custom\n"
            "  (s'assurer que pipeline_config.yaml pointe vers le bon\n"
            "  dlc_project_config et que snapshot-best-090.pt est dans le\n"
            "  dossier train du projet DLC pour que DLC le prenne par défaut)"
        )


if __name__ == "__main__":
    main()
