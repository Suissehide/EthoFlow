"""Ordre de navigation de la sidebar (ruling P1-bis, Task 23).

Task 13 avait été réduite : cette vérification a été déplacée ici, seule
tâche qui tourne après que toutes les pages existent. L'ordre attendu suit
le parcours pipeline du README : Projet, Données, Vidéos & calibration,
Pose (DLC), Nettoyage, VAME, Motifs, Analyses, Visualisations, puis
Configuration et À propos en pages « système ».

Isolation identique à `tests/test_app_projet.py` : `lib.project.PREFS_PATH`
monkeypatché, jamais de dépendance à `Path.home()` ni à
`DEFAULT_PROJECTS_ROOT` (chemin Windows littéral sur ce runner).
"""
from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest

from lib import project as P

REPO = Path(__file__).resolve().parent.parent
APP_PY = str(REPO / "streamlit_app" / "app.py")

ORDRE_ATTENDU = [
    "Projet",
    "Données",
    "Vidéos & calibration",
    "Pose (DLC)",
    "Nettoyage",
    "VAME",
    "Motifs",
    "Analyses",
    "Visualisations",
    "Configuration",
    "À propos",
]


def _noms_nav_dans_l_ordre(at: AppTest) -> list[str]:
    return [b.label for b in at.button if b.key and b.key.startswith("nav_")]


def test_ordre_sans_projet_ouvert(tmp_path, monkeypatch):
    """Sans projet ouvert, seules les pages toujours visibles apparaissent
    (Projet, Configuration, À propos), dans l'ordre."""
    monkeypatch.setattr(P, "PREFS_PATH", tmp_path / "app_prefs.yaml")
    P.save_prefs({
        "projects_root": str(tmp_path / "projects"),
        "models_root": str(tmp_path / "models"),
    })

    at = AppTest.from_file(APP_PY)
    at.run()
    assert not at.exception, at.exception

    noms = _noms_nav_dans_l_ordre(at)
    assert noms == ["Projet", "Configuration", "À propos"], noms


def test_ordre_avec_projet_ouvert_suit_le_parcours_pipeline(tmp_path, monkeypatch):
    """Une fois un projet ouvert, les 11 pages apparaissent exactement dans
    l'ordre du parcours pipeline du README."""
    monkeypatch.setattr(P, "PREFS_PATH", tmp_path / "app_prefs.yaml")

    projects_root = tmp_path / "projects"
    models_root = tmp_path / "models"
    projet = projects_root / "proj-nav-order"
    (projet / "data").mkdir(parents=True)
    models_root.mkdir()

    P.save_prefs({
        "projects_root": str(projects_root),
        "models_root": str(models_root),
        "last_project": str(projet),
    })

    at = AppTest.from_file(APP_PY)
    at.run()
    assert not at.exception, at.exception

    noms = _noms_nav_dans_l_ordre(at)
    assert noms == ORDRE_ATTENDU, noms
