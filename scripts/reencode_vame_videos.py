"""Réencode les vidéos d'un projet VAME en H.264 yuv420p pour qu'OpenCV
puisse les ouvrir.

Pourquoi : `vame.motif_videos` (et d'autres étapes downstream) utilisent
`cv2.VideoCapture` qui galère avec certaines combinaisons codec/pixfmt
sur Windows (typiquement les MP4 produits par Ethovision qui sortent en
yuv422p ou avec un codec non-libx264). Symptôme :

    ValueError: Video capture could not be opened. Ensure the video file
    is valid.\\n D:\\...\\data\\raw\\BV-1001.mp4

Alors que la vidéo s'ouvre très bien dans VLC. La cause est un cv2
buildé sans support pour ce codec/pixfmt précis. Le fix : réencoder en
H.264 + yuv420p (plus universel possible) — ffmpeg le fait sans perte
visible si on garde le bitrate haut.

Stratégie :
    1. Pour chaque BV-*.mp4 du dossier data/raw/ du projet VAME, lance
       ffmpeg vers <fichier>_h264.mp4 avec le codec/pixfmt compatible.
    2. Vérifie la sortie (cv2.VideoCapture peut l'ouvrir).
    3. Remplace l'original : <fichier>.mp4.bak + rename <fichier>_h264.mp4 → <fichier>.mp4.

Si tu veux conserver les originaux ailleurs : passe `--keep-original`,
le script garde le .bak permanent.

Pré-requis :
    - ffmpeg dans le PATH (test : `ffmpeg -version`)
    - cv2 (déjà dans l'env vame normalement)
    - conda activate vame

Usage :
    python scripts/reencode_vame_videos.py \\
        --project-dir D:/ethoflow/projects/bottomview-MCC-2026-06

    # Dry-run pour voir la liste sans rien faire
    python scripts/reencode_vame_videos.py \\
        --project-dir <...> --dry-run

    # Préserve les .mp4.bak (sinon supprimés après vérif)
    python scripts/reencode_vame_videos.py \\
        --project-dir <...> --keep-original
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import (  # noqa: E402
    add_project_dir_arg,
    resolve_project,
    vame_dir,
)


def test_opencv_can_open(video_path: Path) -> bool:
    """Vérifie qu'OpenCV peut bien ouvrir et lire au moins une frame."""
    try:
        import cv2
    except ImportError:
        print("⚠️ cv2 non installé, skip de la vérif post-réencodage", file=sys.stderr)
        return True
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        return False
    ok, _ = cap.read()
    cap.release()
    return ok


def reencode(src: Path, dst: Path) -> bool:
    """Lance ffmpeg pour réencoder src → dst (H.264 + yuv420p).

    Returns True si ffmpeg renvoie 0 et le fichier existe.
    """
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-y",                      # overwrite sans demander
        "-i", str(src),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-preset", "fast",         # bon compromis vitesse/taille
        "-crf", "18",              # quasi-sans-perte visuel
        "-an",                     # pas d'audio
        str(dst),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ❌ ffmpeg : {result.stderr.strip()[:300]}", file=sys.stderr)
        return False
    return dst.exists() and dst.stat().st_size > 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    add_project_dir_arg(parser, required=True)
    parser.add_argument(
        "--dry-run", action="store_true",
        help="N'effectue rien, liste juste les vidéos qui seraient réencodées.",
    )
    parser.add_argument(
        "--keep-original", action="store_true",
        help="Conserve les originaux en .mp4.bak (sinon supprimés après vérif).",
    )
    parser.add_argument(
        "--skip-test", action="store_true",
        help="Skip la vérif cv2.VideoCapture post-réencodage.",
    )
    args = parser.parse_args()

    project = resolve_project(args)
    vame_raw = vame_dir(project) / "data" / "raw"
    if not vame_raw.exists():
        print(f"❌ Dossier introuvable : {vame_raw}", file=sys.stderr)
        sys.exit(1)

    videos = sorted(vame_raw.glob("*.mp4"))
    # Exclut les éventuels _h264.mp4 et .bak résiduels d'un run précédent
    videos = [v for v in videos if "_h264" not in v.stem and not v.name.endswith(".bak")]
    if not videos:
        print(f"⚠ Aucune vidéo .mp4 trouvée dans {vame_raw}")
        sys.exit(0)

    print(f"Dossier        : {vame_raw}")
    print(f"Vidéos cibles  : {len(videos)}")
    print(f"keep-original  : {args.keep_original}")
    print()

    if args.dry_run:
        for v in videos:
            print(f"  [dry] reencode {v.name}")
        sys.exit(0)

    n_ok = n_fail = 0
    for i, src in enumerate(videos, 1):
        print(f"[{i}/{len(videos)}] {src.name}")
        tmp = src.with_name(f"{src.stem}_h264.mp4")
        ok = reencode(src, tmp)
        if not ok:
            print(f"  ⚠ skip (ffmpeg failed)")
            n_fail += 1
            if tmp.exists():
                tmp.unlink()
            continue

        # Vérif post-encodage : cv2 doit pouvoir ouvrir
        if not args.skip_test and not test_opencv_can_open(tmp):
            print(f"  ❌ cv2 ne peut toujours pas ouvrir {tmp.name}, abandon")
            n_fail += 1
            tmp.unlink()
            continue

        # Swap : src → src.bak, tmp → src
        bak = src.with_suffix(".mp4.bak")
        if bak.exists():
            bak.unlink()
        shutil.move(str(src), str(bak))
        shutil.move(str(tmp), str(src))
        if not args.keep_original:
            bak.unlink()

        print(f"  ✓ ok"
              + (f" (original conservé en {bak.name})" if args.keep_original else ""))
        n_ok += 1

    print(f"\n✅ {n_ok}/{len(videos)} vidéos réencodées, {n_fail} échec(s).")
    if n_ok > 0:
        print(
            f"\nTu peux maintenant relancer :\n"
            f"  python scripts/run_vame.py --project-dir {project} motif-videos"
        )


if __name__ == "__main__":
    main()
