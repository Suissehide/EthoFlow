"""
Tests basiques pour vérifier que la structure du repo est cohérente.
À enrichir au fur et à mesure que les scripts sont implémentés.

Lancement :
    conda activate ethoflow
    pytest tests/
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_data_directories_exist():
    """Les dossiers de données attendus doivent exister."""
    for sub in ["raw", "cropped", "dlc-output", "vame-output", "results"]:
        assert (ROOT / "data" / sub).is_dir(), f"data/{sub} manquant"


def test_scripts_exist():
    """Les scripts du pipeline doivent être présents."""
    scripts = ["crop_arenes.py", "run_dlc_inference.py", "run_vame.py", "run_pipeline.py"]
    for s in scripts:
        assert (ROOT / "scripts" / s).is_file(), f"scripts/{s} manquant"


def test_streamlit_app_exists():
    assert (ROOT / "streamlit_app" / "app.py").is_file()


def test_configs_present():
    assert (ROOT / "configs" / "pipeline_config.yaml.example").is_file()
    assert (ROOT / "configs" / "metadata_template.yaml").is_file()


def test_environments_present():
    for env in ["pipeline", "dlc", "vame"]:
        assert (ROOT / f"environment-{env}.yml").is_file()
