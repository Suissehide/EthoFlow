"""
Inférence DeepLabCut sur les vidéos croppées d'une session.

Pour chaque vidéo dans data/cropped/<session_id>/, applique le modèle DLC
configuré dans configs/pipeline_config.yaml et écrit les sorties dans
data/dlc-output/<session_id>/.

Usage:
    python scripts/run_dlc_inference.py <session_id>

Pré-requis :
    - Activer l'env conda 'dlc' avant de lancer
    - Avoir un modèle DLC entraîné (voir docs)
    - Renseigner le chemin du config DLC dans configs/pipeline_config.yaml
"""
import sys
import yaml
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CROPPED_DIR = ROOT / "data" / "cropped"
DLC_OUTPUT_DIR = ROOT / "data" / "dlc-output"
PIPELINE_CONFIG = ROOT / "configs" / "pipeline_config.yaml"


def run_inference(session_id: str) -> None:
    # Import retardé : nécessite l'env 'dlc'
    try:
        import deeplabcut
    except ImportError:
        print(
            "❌ DeepLabCut n'est pas installé dans l'env Python actuel.\n"
            "   Active l'env conda 'dlc' avant de lancer ce script :\n"
            "   conda activate dlc"
        )
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

    cropped_dir = CROPPED_DIR / session_id
    videos = list(cropped_dir.glob("*.mp4"))

    if not videos:
        raise FileNotFoundError(f"Aucune vidéo croppée dans {cropped_dir}")

    output_dir = DLC_OUTPUT_DIR / session_id
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Modèle DLC : {dlc_project_config}")
    print(f"Vidéos à traiter : {[v.name for v in videos]}")
    print(f"Sortie : {output_dir}")

    deeplabcut.analyze_videos(
        config=dlc_project_config,
        videos=[str(v) for v in videos],
        save_as_csv=True,
        destfolder=str(output_dir),
    )
    print(f"\n✅ Inférence DLC terminée : {output_dir}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    run_inference(sys.argv[1])
