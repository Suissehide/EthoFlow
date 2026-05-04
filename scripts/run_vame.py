"""
Analyse VAME sur les sorties DLC d'une session.

Usage:
    python scripts/run_vame.py <session_id>

Pré-requis :
    - Activer l'env conda 'vame' avant de lancer
    - Avoir un projet VAME initialisé (chemin dans configs/pipeline_config.yaml)
    - Sorties DLC présentes dans data/dlc-output/<session_id>/

Squelette à compléter une fois VAME stabilisé sur un projet pilote.
Voir https://github.com/LINCellularNeuroscience/VAME/blob/master/examples/demo.py
"""
import sys
import yaml
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DLC_OUTPUT_DIR = ROOT / "data" / "dlc-output"
VAME_OUTPUT_DIR = ROOT / "data" / "vame-output"
PIPELINE_CONFIG = ROOT / "configs" / "pipeline_config.yaml"


def run_vame(session_id: str) -> None:
    try:
        import vame  # noqa: F401
    except ImportError:
        print(
            "❌ VAME n'est pas installé dans l'env Python actuel.\n"
            "   Active l'env conda 'vame' avant de lancer ce script :\n"
            "   conda activate vame"
        )
        sys.exit(1)

    if not PIPELINE_CONFIG.exists():
        raise FileNotFoundError(f"Config absente : {PIPELINE_CONFIG}")

    with open(PIPELINE_CONFIG) as f:
        config = yaml.safe_load(f)
    vame_project_config = config.get("vame_project_config")
    if not vame_project_config:
        raise ValueError("Clé 'vame_project_config' manquante dans pipeline_config.yaml")

    dlc_dir = DLC_OUTPUT_DIR / session_id
    h5_files = list(dlc_dir.glob("*.h5"))
    if not h5_files:
        raise FileNotFoundError(f"Aucun fichier DLC .h5 dans {dlc_dir}")

    output_dir = VAME_OUTPUT_DIR / session_id
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"VAME projet : {vame_project_config}")
    print(f"Fichiers DLC : {[f.name for f in h5_files]}")

    # TODO : implémenter le pipeline VAME complet une fois validé sur un pilote.
    # Étapes typiques :
    #   vame.csv_to_numpy(config)
    #   vame.egocentric_alignment(config, pose_ref_index=[0, 5])
    #   vame.create_trainset(config)
    #   vame.train_model(config)
    #   vame.evaluate_model(config)
    #   vame.pose_segmentation(config)
    #   vame.motif_videos(config)
    #   vame.community(config)

    print("⚠️  Pipeline VAME : squelette à compléter (voir TODO dans le script)")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    run_vame(sys.argv[1])
