"""Vérification de la restauration du dernier projet à l'ouverture de l'app.

AppTest exercising the actual restore block in `app.py` lines 44-50:
- Positive: with valid project recorded in prefs, `current_project_path`
  is restored to session_state
- Deleted project: recorded project is deleted, app does NOT recreate it
- Missing/corrupt prefs: app runs with no project, no exception

Isolation : `lib.project.PREFS_PATH` est monkeypatché pour éviter toute
dépendance à `~/.ethoflow/app_prefs.yaml` réel. `interactive.DEFAULT_PROJECTS_ROOT`
est aussi monkeypatché pour éviter de créer des dossiers dans le dépôt.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from lib import project as P

REPO = Path(__file__).resolve().parent.parent
APP_PY = str(REPO / "streamlit_app" / "app.py")


def test_app_restores_valid_last_project(tmp_path, monkeypatch):
    """Avec un projet valide enregistré dans les prefs, l'app restaure
    current_project_path dans le session_state au démarrage (teste le bloc
    de restauration app.py lines 44-50 qui appelle set_current_project)."""
    # Create a valid project directory
    project_path = tmp_path / "test_project"
    (project_path / "data").mkdir(parents=True)

    # Mock PREFS_PATH and save a last_project reference
    prefs_file = tmp_path / "app_prefs.yaml"
    monkeypatch.setattr(P, "PREFS_PATH", prefs_file)
    P.save_prefs({"last_project": str(project_path)})

    # Mock DEFAULT_PROJECTS_ROOT to avoid creating directories in repo
    import interactive
    monkeypatch.setattr(interactive, "DEFAULT_PROJECTS_ROOT", tmp_path / "projects")

    # Run the app
    at = AppTest.from_file(APP_PY)
    at.run()

    # The restore block (app.py lines 44-50) should have set current_project_path
    # in session_state after calling set_current_project(dernier)
    assert "current_project_path" in at.session_state
    assert at.session_state["current_project_path"] == str(project_path)


def test_app_does_not_restore_deleted_project(tmp_path, monkeypatch):
    """Quand un projet enregistré a été supprimé du disque, l'app ne le
    restaure pas (grâce au `if dernier and Path(dernier).is_dir()` ligne 48)
    ET ne le recrée pas (crucial pour éviter la résurrection de projets,
    ruling R10.6)."""
    # Create a project directory, record it in prefs, then delete it
    project_path = tmp_path / "deleted_project"
    (project_path / "data").mkdir(parents=True)

    # Mock PREFS_PATH and save reference to project
    prefs_file = tmp_path / "app_prefs.yaml"
    monkeypatch.setattr(P, "PREFS_PATH", prefs_file)
    P.save_prefs({"last_project": str(project_path)})

    # Delete the project directory
    shutil.rmtree(project_path)
    assert not project_path.exists(), "Setup: project should be deleted"

    # Mock DEFAULT_PROJECTS_ROOT
    import interactive
    monkeypatch.setattr(interactive, "DEFAULT_PROJECTS_ROOT", tmp_path / "projects")

    # Run the app with the deleted project in prefs
    at = AppTest.from_file(APP_PY)
    at.run()

    # Assert 1: no project is open in session_state
    # (the Path.is_dir() check on line 48 prevented set_current_project call)
    assert "current_project_path" not in at.session_state

    # Assert 2: directory is STILL deleted, not recreated by any side effect
    assert not project_path.exists(), \
        "CRITICAL: restore logic should not have recreated the directory"


def test_app_handles_missing_prefs_file(tmp_path, monkeypatch):
    """Quand le fichier de prefs n'existe pas, load_prefs() retourne {},
    l'app démarre normalement sans projet ouvert et sans lever d'exception."""
    # Mock PREFS_PATH to a file that does not exist
    prefs_file = tmp_path / "nonexistent.yaml"
    monkeypatch.setattr(P, "PREFS_PATH", prefs_file)

    # Mock DEFAULT_PROJECTS_ROOT
    import interactive
    monkeypatch.setattr(interactive, "DEFAULT_PROJECTS_ROOT", tmp_path / "projects")

    # Run the app — should not raise
    at = AppTest.from_file(APP_PY)
    at.run()

    # No project should be open (load_prefs() returned {})
    assert "current_project_path" not in at.session_state


def test_app_handles_corrupt_prefs_file(tmp_path, monkeypatch):
    """Quand le fichier de prefs est corrompu (YAML invalide), load_prefs()
    retourne {} sur exception, l'app démarre normalement sans projet ouvert
    et sans lever d'exception."""
    # Mock PREFS_PATH and write genuinely malformed YAML
    prefs_file = tmp_path / "app_prefs.yaml"
    monkeypatch.setattr(P, "PREFS_PATH", prefs_file)
    prefs_file.write_text("{ invalid yaml [[ bad }", encoding="utf-8")

    # Mock DEFAULT_PROJECTS_ROOT
    import interactive
    monkeypatch.setattr(interactive, "DEFAULT_PROJECTS_ROOT", tmp_path / "projects")

    # Run the app — should not raise despite corrupt prefs
    at = AppTest.from_file(APP_PY)
    at.run()

    # No project should be open (load_prefs returns {} on corrupt file)
    assert "current_project_path" not in at.session_state
