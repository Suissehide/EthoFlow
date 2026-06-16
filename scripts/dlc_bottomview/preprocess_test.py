"""Test rapide du preprocessing background-subtraction + CLAHE.

Sort une vidéo side-by-side (original | preprocessed) pour décider visuellement
si on lance le pipeline complet sur tout le projet.

Méthode :
    1. Background = projection temporelle sur N frames aléatoires de la vidéo
       (par défaut percentile 90 — plus robuste que la médiane pour une souris
       qui peut stationner dans une zone).
    2. Pour chaque frame du sample : foreground = |frame - background|
       (la souris devient brillante sur fond sombre).
    3. CLAHE par-dessus pour booster les détails locaux (pattes notamment).
    4. Sauve une vidéo side-by-side + une vidéo preprocessed seule + l'image
       background pour inspection.

Pourquoi un test avant le pipeline complet :
    - Permet de tweaker les paramètres (méthode de BG, CLAHE) sur un petit
      sample sans engloutir 30 min de processing.
    - On voit en 30 s de vidéo si la souris ressort vraiment mieux et si les
      parasites du décor (bords d'arène, reflets IR) disparaissent.

Usage :
    python scripts/dlc_bottomview/preprocess_test.py \\
        --video "E:\\data\\bottom_view\\08062026\\970.mp4"

    # tweak params :
    python scripts/dlc_bottomview/preprocess_test.py \\
        --video "E:\\data\\bottom_view\\08062026\\970.mp4" \\
        --start 120 --duration 60 \\
        --bg-method median --clip-limit 3.0

    # Output par défaut : preprocess_test/970_sidebyside.mp4
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np


def compute_background(
    video_path: Path,
    n_samples: int,
    method: str,
) -> np.ndarray:
    """Background empty-arena via projection temporelle.

    Pour une souris SOMBRE sur fond BRILLANT (notre cas), les méthodes :
      - "median" : médiane par pixel — robuste si la souris bouge bien
      - "percentile_90" : 90th percentile — encore plus robuste quand la
        souris stationne dans une zone (le pixel reste sombre 50 % du
        temps mais le top 10 % des valeurs reste bien 'background')
      - "max" : maximum — agressif mais sensible aux flashes IR
    """
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total == 0:
        raise RuntimeError(f"Cannot read frames from {video_path}")

    rng = np.random.default_rng(42)
    indices = sorted(rng.choice(total, min(n_samples, total), replace=False))

    frames: list[np.ndarray] = []
    for i in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ret, frame = cap.read()
        if ret:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    cap.release()

    if not frames:
        raise RuntimeError(f"No frames could be read from {video_path}")

    arr = np.stack(frames, axis=0)
    print(f"  Background: stack shape {arr.shape}, method '{method}'")

    if method == "median":
        bg = np.median(arr, axis=0)
    elif method == "percentile_90":
        bg = np.percentile(arr, 90, axis=0)
    elif method == "max":
        bg = arr.max(axis=0)
    else:
        raise ValueError(f"unknown method: {method}")

    return bg.astype(np.uint8)


def process_frame(
    gray: np.ndarray,
    bg: np.ndarray,
    clahe: cv2.CLAHE,
) -> np.ndarray:
    """frame brute -> frame pré-traitée (subtraction + CLAHE)."""
    diff = cv2.absdiff(gray, bg)
    return clahe.apply(diff)


def make_side_by_side(
    orig_bgr: np.ndarray,
    processed_gray: np.ndarray,
) -> np.ndarray:
    """Concatène original (BGR) et processed (gris) côte à côte avec labels."""
    proc_bgr = cv2.cvtColor(processed_gray, cv2.COLOR_GRAY2BGR)

    label_color = (0, 255, 255)  # jaune vif
    cv2.putText(orig_bgr, "ORIGINAL", (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, label_color, 2)
    cv2.putText(proc_bgr, "PREPROCESSED", (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, label_color, 2)

    return np.hstack([orig_bgr, proc_bgr])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test du preprocessing background-subtraction + CLAHE.",
    )
    parser.add_argument("--video", required=True, type=Path,
                        help="Vidéo source")
    parser.add_argument("--out-dir", type=Path,
                        default=Path("preprocess_test"),
                        help="Dossier de sortie (défaut: preprocess_test/)")
    parser.add_argument("--start", type=float, default=60.0,
                        help="Début du sample en secondes (défaut: 60s, "
                        "évite les premières frames souvent atypiques)")
    parser.add_argument("--duration", type=float, default=30.0,
                        help="Durée du sample en secondes (défaut: 30s)")
    parser.add_argument("--bg-samples", type=int, default=200,
                        help="Nb de frames pour calculer le background")
    parser.add_argument("--bg-method",
                        choices=["median", "percentile_90", "max"],
                        default="percentile_90",
                        help="Méthode de projection temporelle (défaut: percentile_90)")
    parser.add_argument("--clip-limit", type=float, default=2.0,
                        help="CLAHE clipLimit (défaut: 2.0). Plus haut = plus "
                        "de contraste mais aussi plus de bruit.")
    parser.add_argument("--tile-size", type=int, default=8,
                        help="CLAHE tileGridSize (carré, défaut: 8)")
    args = parser.parse_args()

    if not args.video.exists():
        print(f"❌ Vidéo introuvable : {args.video}")
        sys.exit(1)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # ----- 1) Background -----
    print(f"Calcul du background depuis {args.video.name}")
    print(f"  méthode: {args.bg_method}, n samples: {args.bg_samples}")
    bg = compute_background(args.video, args.bg_samples, args.bg_method)
    bg_path = args.out_dir / f"{args.video.stem}_background.png"
    cv2.imwrite(str(bg_path), bg)
    print(f"  ✅ Background sauvegardé : {bg_path}\n")

    # ----- 2) Process le sample -----
    cap = cv2.VideoCapture(str(args.video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    start_frame = int(args.start * fps)
    n_frames = int(args.duration * fps)
    n_frames = min(n_frames, total - start_frame)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    side_path = args.out_dir / f"{args.video.stem}_sidebyside.mp4"
    side_writer = cv2.VideoWriter(
        str(side_path), fourcc, fps, (width * 2, height), isColor=True,
    )

    proc_path = args.out_dir / f"{args.video.stem}_preprocessed.mp4"
    proc_writer = cv2.VideoWriter(
        str(proc_path), fourcc, fps, (width, height), isColor=False,
    )

    clahe = cv2.createCLAHE(
        clipLimit=args.clip_limit,
        tileGridSize=(args.tile_size, args.tile_size),
    )

    print(f"Processing : start={args.start}s, duration={args.duration}s "
          f"({n_frames} frames)")
    print(f"  CLAHE clipLimit={args.clip_limit}, "
          f"tileGridSize={args.tile_size}x{args.tile_size}\n")

    for i in range(n_frames):
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        enhanced = process_frame(gray, bg, clahe)

        side_writer.write(make_side_by_side(frame.copy(), enhanced))
        proc_writer.write(enhanced)

        if i % 200 == 0 and i > 0:
            print(f"  {i}/{n_frames} ({100 * i / n_frames:.0f}%)")

    cap.release()
    side_writer.release()
    proc_writer.release()

    # ----- 3) Récap -----
    print(f"\n✅ Background image     : {bg_path}")
    print(f"✅ Side-by-side video   : {side_path}")
    print(f"✅ Preprocessed video   : {proc_path}")
    print(
        "\nOuvre le side-by-side dans un lecteur vidéo (VLC, Windows Media...).\n"
        "Si la souris ressort beaucoup mieux ET les bords de l'arène "
        "disparaissent,\n"
        "c'est win → on lance le pipeline complet :\n"
        "  1. Dupliquer le projet DLC pour conserver l'original.\n"
        "  2. Pré-traiter les 6 vidéos dans un dossier <data>_preproc/.\n"
        "  3. Remplacer les PNG dans labeled-data/<video>/ par leurs versions\n"
        "     pré-traitées (les labels x,y restent valides).\n"
        "  4. Update VIDEOS_TO_ANALYZE dans _config.py.\n"
        "  5. Nettoyer caches + re-train (50 epochs suffisent depuis ce point).\n"
    )


if __name__ == "__main__":
    main()
