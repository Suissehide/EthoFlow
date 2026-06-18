"""Pré-traite TOUTES les vidéos du dataset (mode production de preprocess_test_mog2.py).

Pour chaque vidéo dans VIDEOS_TO_ANALYZE :
    1. Warmup MOG2 sur MOG2_WARMUP frames aléatoires (modèle background convergé)
    2. Frozen mode : learning_rate=MOG2_LEARNING_RATE pendant le processing
       → la souris n'est jamais absorbée dans le background, peu importe
       combien de temps elle stationne
    3. Mode --no-mask équivalent : absdiff avec le background adaptatif
       courant (préserve les détails de pattes)
    4. CLAHE par-dessus pour booster le contraste local

Output : <PREPROCESSED_VIDEO_DIR>/<même_nom>.mp4

Pré-requis :
    - PROJECT_DIR dupliqué vers PREPROCESSED_PROJECT_DIR (manuel) :
        xcopy /E /I "E:\\DLC\\souris-bottomview-Leo-2026-06-05" ^
                    "E:\\DLC\\souris-bottomview-Leo-2026-06-05-preproc"
    - PREPROCESSED_VIDEO_DIR créé (le script le créera si absent)
    - VIDEOS_TO_ANALYZE à jour dans _config.py (les 6 vidéos)

Workflow global preprocessing :
    1. python scripts/dlc_bottomview/preprocess_all.py
       → ~30-60 min total (6 vidéos × ~5-10 min/vidéo)
    2. python scripts/dlc_bottomview/migrate_project_to_preproc.py
       → remplace les PNG dans labeled-data/ + update config.yaml
    3. Update _config.py :
         PROJECT_DIR = PREPROCESSED_PROJECT_DIR
         CONFIG = str(PROJECT_DIR / "config.yaml")
         (et VIDEOS_TO_ANALYZE etc. pointent vers les preproc paths)
    4. Nettoyer caches du nouveau projet + lancer 02_train.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _config import (  # noqa: E402
    CLAHE_CLIP_LIMIT,
    CLAHE_TILE_SIZE,
    MOG2_HISTORY,
    MOG2_LEARNING_RATE,
    MOG2_VAR_THRESHOLD,
    MOG2_WARMUP,
    PREPROCESSED_VIDEO_DIR,
    VIDEOS_TO_ANALYZE,
)


def warmup_mog2(
    fgbg: cv2.BackgroundSubtractorMOG2,
    video_path: Path,
    n_warmup: int,
) -> None:
    """Entraîne MOG2 sur n_warmup frames aléatoires uniformément distribuées."""
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    rng = np.random.default_rng(42)
    indices = sorted(rng.choice(total, min(n_warmup, total), replace=False))
    for i in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ret, frame = cap.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            fgbg.apply(gray)
    cap.release()


def process_video(video_path: Path, out_path: Path) -> None:
    """Pré-traite une vidéo complète : MOG2 frozen + absdiff + CLAHE."""
    print(f"\n→ {video_path.name}")
    t0 = time.time()

    # 1) MOG2 setup + warmup
    fgbg = cv2.createBackgroundSubtractorMOG2(
        history=MOG2_HISTORY,
        varThreshold=MOG2_VAR_THRESHOLD,
        detectShadows=False,
    )
    print(f"  Warmup MOG2 sur {MOG2_WARMUP} frames...")
    warmup_mog2(fgbg, video_path, MOG2_WARMUP)

    # 2) Process toute la vidéo
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        str(out_path), fourcc, fps, (width, height), isColor=False,
    )

    clahe = cv2.createCLAHE(
        clipLimit=CLAHE_CLIP_LIMIT,
        tileGridSize=(CLAHE_TILE_SIZE, CLAHE_TILE_SIZE),
    )

    print(f"  Processing {total} frames...")
    for i in range(total):
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # MOG2 quasi figé après warmup
        fgbg.apply(gray, learningRate=MOG2_LEARNING_RATE)
        bg = fgbg.getBackgroundImage()

        # absdiff + CLAHE
        diff = cv2.absdiff(gray, bg)
        enhanced = clahe.apply(diff)
        writer.write(enhanced)

        if i > 0 and i % 1000 == 0:
            elapsed = time.time() - t0
            eta = elapsed * (total - i) / i
            print(f"    {i}/{total} ({100 * i / total:.0f}%) — "
                  f"ETA {eta:.0f}s")

    cap.release()
    writer.release()
    elapsed = time.time() - t0
    print(f"  ✅ {out_path.name} en {elapsed:.0f}s ({elapsed/60:.1f}min)")


def main() -> None:
    if not VIDEOS_TO_ANALYZE:
        print("⚠ VIDEOS_TO_ANALYZE est vide dans _config.py")
        sys.exit(1)

    # Vérifie que toutes les sources existent
    missing = [v for v in VIDEOS_TO_ANALYZE if not v.exists()]
    if missing:
        print("❌ Vidéos source introuvables :")
        for v in missing:
            print(f"  - {v}")
        sys.exit(1)

    PREPROCESSED_VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output : {PREPROCESSED_VIDEO_DIR}")
    print(
        f"Config MOG2 : history={MOG2_HISTORY}, varThreshold={MOG2_VAR_THRESHOLD}, "
        f"warmup={MOG2_WARMUP}, learning_rate={MOG2_LEARNING_RATE}"
    )
    print(
        f"Config CLAHE : clipLimit={CLAHE_CLIP_LIMIT}, "
        f"tileGridSize={CLAHE_TILE_SIZE}x{CLAHE_TILE_SIZE}"
    )

    t_total = time.time()
    for video in VIDEOS_TO_ANALYZE:
        out_path = PREPROCESSED_VIDEO_DIR / video.name
        if out_path.exists():
            print(f"\n→ {video.name} : déjà pré-traité, skip "
                  f"(supprime {out_path} pour forcer le recalcul)")
            continue
        process_video(video, out_path)

    total_min = (time.time() - t_total) / 60
    print(
        f"\n✅ {len(VIDEOS_TO_ANALYZE)} vidéo(s) pré-traitée(s) en "
        f"{total_min:.1f} min total."
    )
    print(
        "\nÉtape suivante :\n"
        "  python scripts/dlc_bottomview/migrate_project_to_preproc.py\n"
        "→ remplace les PNG dans labeled-data/ du projet dupliqué par leurs\n"
        "  versions pré-traitées, et update les video_sets du config.yaml."
    )


if __name__ == "__main__":
    main()
