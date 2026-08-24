"""
Tests basiques pour vérifier que la structure du repo est cohérente.
À enrichir au fur et à mesure que les scripts sont implémentés.

Lancement :
    conda run -n ethoflow python -m pytest tests/
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_scripts_exist():
    """Les scripts du pipeline doivent être présents."""
    scripts = ["crop_arenes.py", "run_dlc_inference.py", "run_vame.py", "run_pipeline.py"]
    for s in scripts:
        assert (ROOT / "scripts" / s).is_file(), f"scripts/{s} manquant"


def test_streamlit_app_exists():
    assert (ROOT / "streamlit_app" / "app.py").is_file()


def test_environments_present():
    for env in ["pipeline", "dlc", "vame"]:
        assert (ROOT / f"environment-{env}.yml").is_file()


def test_scripts_partages_presents():
    """Les modules partagés entre CLI et app doivent exister."""
    for s in ("paths.py", "interactive.py", "run_vame.py", "analyze_vame.py"):
        assert (ROOT / "scripts" / s).is_file(), f"scripts/{s} manquant"


def test_streamlit_lib_importable():
    """lib/ doit s'importer sans Streamlit lancé."""
    import lib.project  # noqa: F401
