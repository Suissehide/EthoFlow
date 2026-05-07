"""
Calibration interactive des 4 arènes.

Affiche une frame extraite d'une vidéo source, te demande de dessiner les
4 rectangles d'arène dans l'ordre A1, A2, A3, A4, puis sauvegarde les
coordonnées dans `configs/pipeline_config.yaml` sous `default_arenes_coords`.

Ces coords sont utilisées comme fallback par `crop_arenes.py` et
`assign_arenas.py` quand le `metadata.yaml` d'une session a `coords: null`.

Contrôles :
    Clic-glisser     : dessine un rectangle
    U / Backspace    : annule le dernier rectangle
    R                : reset complet
    ENTREE           : valide quand 4 rectangles sont posés
    ESC              : quitte sans rien sauver

Les rectangles déjà dessinés restent visibles avec leur label.
Une bannière en haut indique le compteur (« A2 (1/4) »).

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


WINDOW = "Calibration arenes — clic-glisser, U=annuler, ENTREE=valider, ESC=quitter"

# Couleurs BGR pour chaque arène (visuellement distinctes)
ARENA_COLORS = [
    (0, 255, 0),     # A1 vert
    (0, 200, 255),   # A2 orange
    (255, 100, 100), # A3 bleu clair
    (255, 0, 255),   # A4 magenta
]


def _render(frame, rects, drag, n_target):
    """Construit l'image à afficher : frame + rectangles + bannière statut."""
    display = frame.copy()

    # Rectangles déjà validés
    for i, (x, y, w, h) in enumerate(rects):
        color = ARENA_COLORS[i % len(ARENA_COLORS)]
        cv2.rectangle(display, (x, y), (x + w, y + h), color, 2)
        label = f"A{i + 1}"
        # Fond noir derrière le label pour la lisibilité
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        cv2.rectangle(display, (x + 4, y + 4), (x + 8 + tw, y + 12 + th), (0, 0, 0), -1)
        cv2.putText(display, label, (x + 6, y + 8 + th),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # Rectangle en cours de tracé
    if drag is not None:
        (x0, y0), (x1, y1) = drag
        next_color = ARENA_COLORS[len(rects) % len(ARENA_COLORS)]
        cv2.rectangle(display, (x0, y0), (x1, y1), next_color, 1, cv2.LINE_AA)

    # Bannière statut en haut
    n = len(rects)
    if n < n_target:
        msg = f"Dessine A{n + 1}  ({n}/{n_target})"
        bg = (40, 40, 40)
        fg = ARENA_COLORS[n % len(ARENA_COLORS)]
    else:
        msg = f"OK ({n}/{n_target}) — ENTREE pour valider, U pour annuler le dernier"
        bg = (0, 80, 0)
        fg = (200, 255, 200)

    h, w = display.shape[:2]
    banner_h = 40
    cv2.rectangle(display, (0, 0), (w, banner_h), bg, -1)
    cv2.putText(display, msg, (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, fg, 2, cv2.LINE_AA)
    return display


def select_arenas(frame, n_target: int = 4) -> list[tuple[int, int, int, int]]:
    """Loop interactive : retourne la liste des n_target rectangles, ou [] si annulation."""
    rects: list[tuple[int, int, int, int]] = []
    state = {"drawing": False, "start": None, "current": None}

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            state["drawing"] = True
            state["start"] = (x, y)
            state["current"] = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and state["drawing"]:
            state["current"] = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and state["drawing"]:
            state["drawing"] = False
            x0, y0 = state["start"]
            x1, y1 = (x, y)
            rx, ry = min(x0, x1), min(y0, y1)
            rw, rh = abs(x1 - x0), abs(y1 - y0)
            if rw > 5 and rh > 5 and len(rects) < n_target:
                rects.append((rx, ry, rw, rh))
                print(f"  + A{len(rects)} = ({rx}, {ry}, {rw}, {rh})")
            state["start"] = None
            state["current"] = None

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW, on_mouse)

    print("\n→ Clic-glisser pour dessiner chaque arène dans l'ordre A1 → A2 → A3 → A4.")
    print("  U : annuler le dernier   |   R : tout effacer   |   ENTREE : valider   |   ESC : quitter\n")

    while True:
        drag = (state["start"], state["current"]) if state["drawing"] else None
        cv2.imshow(WINDOW, _render(frame, rects, drag, n_target))
        key = cv2.waitKey(20) & 0xFF

        if key == 27:  # ESC
            print("Calibration annulée.")
            rects = []
            break
        if key in (13, 10) and len(rects) == n_target:  # ENTREE / Return
            break
        if key in (ord("u"), ord("U"), 8):  # U / Backspace
            if rects:
                last = rects.pop()
                print(f"  - annulé : {last}")
        if key in (ord("r"), ord("R")):
            if rects:
                print("  ↺  reset")
                rects.clear()

    cv2.destroyAllWindows()
    return rects


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
    rois = select_arenas(frame, n_target=N_ARENAS)

    if len(rois) != N_ARENAS:
        print(
            f"❌ {len(rois)} rectangle(s) sélectionné(s), il en faut {N_ARENAS}. "
            f"Aucune sauvegarde.",
            file=sys.stderr,
        )
        sys.exit(1)

    coords = {f"A{i+1}": [int(v) for v in roi] for i, roi in enumerate(rois)}
    save_coords(coords)


if __name__ == "__main__":
    main()
