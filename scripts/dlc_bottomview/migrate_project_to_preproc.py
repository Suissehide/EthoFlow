"""Migre le projet DLC dupliqué vers les vidéos pré-traitées.

Pré-requis :
    1. preprocess_all.py a tourné — les .mp4 pré-traités sont dans
       PREPROCESSED_VIDEO_DIR
    2. PROJECT_DIR a été dupliqué (manuel xcopy) vers PREPROCESSED_PROJECT_DIR

Ce script fait trois choses dans PREPROCESSED_PROJECT_DIR :

    1. Pour chaque PNG dans labeled-data/<video>/ : extrait la frame
       correspondante du .mp4 pré-traité et écrase le PNG.
       Tes labels (x, y) restent strictement valides parce qu'on a juste
       changé le CONTENU pixel des frames sans toucher aux coordonnées.

    2. Update config.yaml :
         - project_path → PREPROCESSED_PROJECT_DIR
         - video_sets : chaque clé (= path absolu de la vidéo source) est
           remplacée par le path de sa version pré-traitée

    3. Sauvegarde l'ancien config.yaml en config.yaml.bak pour pouvoir
       revenir en arrière.

Après migration, tu mets à jour _config.py pour pointer vers les nouveaux
paths (PROJECT_DIR, CONFIG, VIDEOS_TO_ANALYZE, etc.) puis tu lances :
    Remove-Item -Recurse -Force "<PREPROCESSED_PROJECT_DIR>\\training-datasets"
    Remove-Item -Recurse -Force "<PREPROCESSED_PROJECT_DIR>\\dlc-models-pytorch"
    python scripts\\dlc_bottomview\\02_train.py

Sécurité :
    - Tes PNG originaux ne sont PAS modifiés dans PROJECT_DIR (l'original).
      On opère uniquement sur la copie dans PREPROCESSED_PROJECT_DIR.
    - Le config.yaml original (de la copie) est sauvegardé en .bak avant
      modification.
    - Avant tout, ce script verifie que PREPROCESSED_PROJECT_DIR != PROJECT_DIR.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import cv2
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _config import (  # noqa: E402
    PREPROCESSED_PROJECT_DIR,
    PREPROCESSED_VIDEO_DIR,
    PROJECT_DIR,
)


def replace_pngs_for_video(
    labeled_dir: Path,
    preproc_video: Path,
) -> tuple[int, int]:
    """Remplace tous les PNG de labeled_dir par leurs versions pré-traitées
    extraites de preproc_video.

    Returns: (n_replaced, n_skipped)
    """
    pngs = sorted(labeled_dir.glob("img*.png"))
    if not pngs:
        return 0, 0

    cap = cv2.VideoCapture(str(preproc_video))
    if not cap.isOpened():
        print(f"  ❌ Impossible d'ouvrir {preproc_video}")
        return 0, len(pngs)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    n_replaced = 0
    n_skipped = 0
    for png in pngs:
        # img00170.png → 170
        stem = png.stem
        if not stem.startswith("img"):
            n_skipped += 1
            continue
        try:
            frame_idx = int(stem[3:])
        except ValueError:
            n_skipped += 1
            continue
        if frame_idx >= total_frames:
            print(f"  ⚠ frame {frame_idx} hors vidéo "
                  f"({total_frames} frames), skip")
            n_skipped += 1
            continue

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            n_skipped += 1
            continue

        # Les vidéos pré-traitées sont grayscale ; convertir en BGR pour
        # que le PNG soit cohérent avec ce que DLC attend (cv2 écrit en
        # BGR par défaut).
        if frame.ndim == 2:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        else:
            frame_bgr = frame
        cv2.imwrite(str(png), frame_bgr)
        n_replaced += 1

    cap.release()
    return n_replaced, n_skipped


def update_config_yaml(
    config_path: Path,
    new_project_path: Path,
    preproc_video_dir: Path,
) -> None:
    """Update project_path + video_sets dans config.yaml du projet."""
    # Backup
    backup_path = config_path.with_suffix(".yaml.bak")
    shutil.copy(config_path, backup_path)
    print(f"  Backup → {backup_path.name}")

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # project_path
    old_pp = cfg.get("project_path", "(absent)")
    cfg["project_path"] = str(new_project_path)
    print(f"  project_path : {old_pp}")
    print(f"               → {new_project_path}")

    # video_sets : chaque clé pointe vers la vidéo source → on remplace par
    # la vidéo pré-traitée (même nom de fichier).
    old_video_sets = cfg.get("video_sets", {})
    new_video_sets = {}
    for old_path, attrs in old_video_sets.items():
        old_p = Path(old_path)
        new_p = preproc_video_dir / old_p.name
        new_video_sets[str(new_p)] = attrs
        print(f"  video : {old_p.name}")
        print(f"        → {new_p}")
    cfg["video_sets"] = new_video_sets

    with open(config_path, "w") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)


def main() -> None:
    # Sanity check : on ne veut SURTOUT pas opérer sur le projet original
    if PREPROCESSED_PROJECT_DIR.resolve() == PROJECT_DIR.resolve():
        print("❌ PREPROCESSED_PROJECT_DIR == PROJECT_DIR dans _config.py !")
        print("   On ne va pas écraser ton projet original. Modifie _config.py.")
        sys.exit(1)

    if not PREPROCESSED_PROJECT_DIR.exists():
        print(f"❌ PREPROCESSED_PROJECT_DIR n'existe pas : {PREPROCESSED_PROJECT_DIR}")
        print(
            f"   Duplique d'abord le projet :\n"
            f'   xcopy /E /I "{PROJECT_DIR}" "{PREPROCESSED_PROJECT_DIR}"'
        )
        sys.exit(1)

    if not PREPROCESSED_VIDEO_DIR.exists():
        print(f"❌ Pas de vidéos pré-traitées : {PREPROCESSED_VIDEO_DIR}")
        print("   Lance d'abord : python scripts/dlc_bottomview/preprocess_all.py")
        sys.exit(1)

    print(f"Migration du projet :\n"
          f"  source   : {PROJECT_DIR}  (intact)\n"
          f"  cible    : {PREPROCESSED_PROJECT_DIR}\n"
          f"  vidéos   : {PREPROCESSED_VIDEO_DIR}\n")

    # --- 1) Remplace les PNG dans labeled-data/ ---
    labeled_root = PREPROCESSED_PROJECT_DIR / "labeled-data"
    if not labeled_root.exists():
        print(f"❌ Pas de labeled-data dans {PREPROCESSED_PROJECT_DIR}")
        sys.exit(1)

    print("=== Remplacement des PNG dans labeled-data/ ===")
    total_replaced = 0
    total_skipped = 0
    for vdir in sorted(labeled_root.iterdir()):
        if not vdir.is_dir() or vdir.name.endswith("_labeled"):
            continue
        preproc_video = PREPROCESSED_VIDEO_DIR / f"{vdir.name}.mp4"
        if not preproc_video.exists():
            print(f"  ⚠ {vdir.name}: pas de vidéo pré-traitée correspondante, skip")
            continue
        n_repl, n_skip = replace_pngs_for_video(vdir, preproc_video)
        print(f"  {vdir.name}: {n_repl} PNG remplacés, {n_skip} skip")
        total_replaced += n_repl
        total_skipped += n_skip
    print(f"\nTotal : {total_replaced} PNG remplacés, {total_skipped} skip\n")

    # --- 2) Update config.yaml ---
    print("=== Update config.yaml ===")
    config_path = PREPROCESSED_PROJECT_DIR / "config.yaml"
    if not config_path.exists():
        print(f"❌ Pas de config.yaml dans {PREPROCESSED_PROJECT_DIR}")
        sys.exit(1)
    update_config_yaml(config_path, PREPROCESSED_PROJECT_DIR, PREPROCESSED_VIDEO_DIR)

    print(
        "\n✅ Migration terminée.\n\n"
        "Étapes suivantes :\n"
        "  1. Update _config.py :\n"
        "       PROJECT_DIR = PREPROCESSED_PROJECT_DIR\n"
        "       CONFIG = str(PROJECT_DIR / 'config.yaml')\n"
        "       PILOT_VIDEO = PREPROCESSED_VIDEO_DIR / '970.mp4'\n"
        "       VIDEOS_TO_ANALYZE = [PREPROCESSED_VIDEO_DIR / v.name "
        "for v in VIDEOS_TO_ANALYZE]\n"
        "     (ou simplement édite à la main les paths absolus)\n"
        "\n"
        "  2. Nettoie caches du nouveau projet :\n"
        f'       Remove-Item -Recurse -Force "{PREPROCESSED_PROJECT_DIR}\\training-datasets"\n'
        f'       Remove-Item -Recurse -Force "{PREPROCESSED_PROJECT_DIR}\\dlc-models-pytorch"\n'
        "\n"
        "  3. Lance le training sur le projet preproc :\n"
        "       python scripts\\dlc_bottomview\\02_train.py\n"
    )


if __name__ == "__main__":
    main()
