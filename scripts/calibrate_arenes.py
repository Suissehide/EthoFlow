"""
Calibration interactive des 4 arènes.

Affiche une frame extraite d'une vidéo source, te demande de dessiner les
4 rectangles d'arène dans l'ordre A1, A2, A3, A4, puis sauvegarde les
coordonnées dans `configs/pipeline_config.yaml` sous `default_arenes_coords`.

Ces coords sont utilisées comme fallback par `crop_arenes.py` et
`assign_arenas.py` quand le `metadata.yaml` d'une session a `coords: null`.

Usage:
    # depuis une vidéo arbitraire
    python scripts/calibrate_arenes.py /chemin/vers/video.mp4

    # ou depuis une session déjà synchronisée
    python scripts/calibrate_arenes.py --session OF-M1-20251010-V01

    # spécifier la frame à utiliser (défaut : milieu de vidéo)
    python scripts/calibrate_arenes.py video.mp4 --frame 500

Pré-requis :
    pip install opencv-python pyyaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import yaml

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
CONFIG_PATH = ROOT / "configs" / "pipeline_config.yaml"
N_ARENAS = 4


def resolve_video(args) -> Path:
    if args.session:
        meta_path = RAW_DIR / args.session / "metadata.yaml"
        if not meta_path.exists():
            raise FileNotFoundError(f"Metadata absent : {meta_path}")
        with open(meta_path) as f:
            meta = yaml.safe_load(f)
        source = meta.get("source_video")
        if not source:
            raise ValueError("Pas de `source_video` dans le metadata")
        return Path(source)
    if args.video:
        return Path(args.video)
    raise ValueError("Préciser --session ou un chemin vidéo en argument")


def extract_frame(video_path: Path, frame_idx: int | None) -> "cv2.Mat":
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Impossible d'ouvrir : {video_path}")
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_idx is None:
        frame_idx = n // 2
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Lecture frame {frame_idx} échouée")
    print(f"Frame extraite : {frame_idx}/{n}, taille {frame.shape[1]}×{frame.shape[0]}")
    return frame


def select_arenas(frame) -> list[tuple[int, int, int, int]]:
    print(
        "\n→ Dessine 4 rectangles à la souris (un par arène, dans l'ordre A1, A2, A3, A4).\n"
        "  ENTREE / ESPACE après chaque rectangle.\n"
        "  ESC pour annuler.\n"
    )
    rois = cv2.selectROIs("Arènes (A1, A2, A3, A4)", frame,
                          showCrosshair=True, fromCenter=False)
    cv2.destroyAllWindows()
    return [tuple(int(v) for v in roi) for roi in rois]


def save_coords(coords: dict[str, list[int]]) -> None:
    config = {}
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH) as f:
            config = yaml.safe_load(f) or {}
    config["default_arenes_coords"] = coords
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_PATH, "w") as f:
        yaml.dump(config, f, sort_keys=False, allow_unicode=True)
    print(f"\n✅ Coords sauvées dans {CONFIG_PATH}")
    for k, v in coords.items():
        print(f"   {k}: {v}")


def main():
    parser = argparse.ArgumentParser(description="Calibration interactive des 4 arènes")
    parser.add_argument("video", nargs="?", help="Chemin vers une vidéo source")
    parser.add_argument("--session", help="Session ID (utilise sa source_video)")
    parser.add_argument("--frame", type=int, default=None,
                        help="Numéro de frame à afficher (défaut : milieu)")
    args = parser.parse_args()

    try:
        video_path = resolve_video(args)
    except (FileNotFoundError, ValueError) as e:
        print(f"❌ {e}", file=sys.stderr)
        sys.exit(1)

    if not video_path.exists():
        print(f"❌ Vidéo introuvable : {video_path}", file=sys.stderr)
        sys.exit(1)

    frame = extract_frame(video_path, args.frame)
    rois = select_arenas(frame)

    if len(rois) != N_ARENAS:
        print(
            f"❌ {len(rois)} rectangle(s) sélectionné(s), il en faut {N_ARENAS}.",
            file=sys.stderr,
        )
        sys.exit(1)

    coords = {f"A{i+1}": list(roi) for i, roi in enumerate(rois)}
    save_coords(coords)


if __name__ == "__main__":
    main()
