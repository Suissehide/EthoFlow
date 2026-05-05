"""
Inférence DeepLabCut sur la vidéo source d'une session.

Deux modes :

1. **SuperAnimal multi-animal** (défaut) — lit `source_video` du metadata,
   lance `video_inference_superanimal` directement sur la vidéo entière
   (pas de crop). Sortie : un seul .h5 multi-animal contenant les
   trajectoires des 4 souris.

2. **Modèle custom** — si `dlc_project_config` est défini dans
   `configs/pipeline_config.yaml`, lance `analyze_videos` avec ce modèle.

Usage:
    python scripts/run_dlc_inference.py <session_id>
    python scripts/run_dlc_inference.py <session_id> --mode custom
    python scripts/run_dlc_inference.py <session_id> --video-adapt

Pré-requis :
    - Activer l'env conda 'dlc' avant de lancer
    - DLC 3.x + PyTorch
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
DLC_OUTPUT_DIR = ROOT / "data" / "dlc-output"
PIPELINE_CONFIG = ROOT / "configs" / "pipeline_config.yaml"


def load_session_metadata(session_id: str) -> dict:
    metadata_path = RAW_DIR / session_id / "metadata.yaml"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata absent : {metadata_path}")
    with open(metadata_path) as f:
        return yaml.safe_load(f)


def get_source_video(metadata: dict) -> Path:
    source = metadata.get("source_video")
    if not source:
        raise ValueError("Pas de `source_video` dans le metadata.yaml")
    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"Vidéo source introuvable : {path}")
    return path


def run_superanimal(
    session_id: str,
    superanimal_name: str,
    model_name: str,
    detector_name: str,
    video_adapt: bool,
) -> None:
    try:
        import deeplabcut
    except ImportError:
        print(
            "❌ DeepLabCut non installé. Active l'env conda 'dlc' :\n"
            "   conda activate dlc",
            file=sys.stderr,
        )
        sys.exit(1)

    metadata = load_session_metadata(session_id)
    source = get_source_video(metadata)

    output_dir = DLC_OUTPUT_DIR / session_id
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Vidéo : {source}")
    print(f"SuperAnimal : {superanimal_name}")
    print(f"  modèle    : {model_name}")
    print(f"  détecteur : {detector_name}")
    print(f"  video_adapt = {video_adapt}")

    deeplabcut.video_inference_superanimal(
        [str(source)],
        superanimal_name=superanimal_name,
        model_name=model_name,
        detector_name=detector_name,
        videotype="mp4",
        video_adapt=video_adapt,
        dest_folder=str(output_dir),
    )

    print(f"\n✅ Inférence SuperAnimal terminée : {output_dir}")
    print("   Étape suivante : `python scripts/assign_arenas.py "
          f"{session_id}` pour splitter par arène.")


def run_custom(session_id: str) -> None:
    try:
        import deeplabcut
    except ImportError:
        print("❌ DeepLabCut non installé.", file=sys.stderr)
        sys.exit(1)

    if not PIPELINE_CONFIG.exists():
        raise FileNotFoundError(
            f"Config absente : {PIPELINE_CONFIG}\n"
            f"Copie configs/pipeline_config.yaml.example et adapte-le."
        )
    with open(PIPELINE_CONFIG) as f:
        config = yaml.safe_load(f)
    dlc_project_config = config.get("dlc_project_config")
    if not dlc_project_config:
        raise ValueError("Clé 'dlc_project_config' manquante dans pipeline_config.yaml")

    metadata = load_session_metadata(session_id)
    source = get_source_video(metadata)

    output_dir = DLC_OUTPUT_DIR / session_id
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Modèle DLC custom : {dlc_project_config}")
    print(f"Vidéo : {source}")

    deeplabcut.analyze_videos(
        config=dlc_project_config,
        videos=[str(source)],
        save_as_csv=True,
        destfolder=str(output_dir),
    )
    print(f"\n✅ Inférence custom terminée : {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inférence DLC sur une session.")
    parser.add_argument("session_id")
    parser.add_argument("--mode", choices=["superanimal", "custom"], default="superanimal")
    parser.add_argument("--superanimal-name", default="superanimal_topviewmouse")
    parser.add_argument("--superanimal-model", default="hrnet_w32")
    parser.add_argument("--superanimal-detector", default="fasterrcnn_resnet50_fpn_v2")
    parser.add_argument("--video-adapt", action="store_true",
                        help="Active le fine-tuning court (plus précis, plus lent)")
    args = parser.parse_args()

    try:
        if args.mode == "superanimal":
            run_superanimal(
                args.session_id,
                superanimal_name=args.superanimal_name,
                model_name=args.superanimal_model,
                detector_name=args.superanimal_detector,
                video_adapt=args.video_adapt,
            )
        else:
            run_custom(args.session_id)
    except (FileNotFoundError, ValueError) as e:
        print(f"❌ {e}", file=sys.stderr)
        sys.exit(1)
