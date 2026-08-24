"""Tests de lib/config.py — seul module de lib/ qui touche Streamlit.

`st.session_state` fonctionne en mode "bare" (hors `streamlit run`) comme
un simple dict, avec juste un warning : suffisant pour tester ces
fonctions sans passer par `streamlit.testing.v1.AppTest`.
"""
from __future__ import annotations

import shutil

import streamlit as st

from lib import config as C


def test_current_project_efface_si_dossier_disparu(tmp_path):
    """Régression du projet zombie (ruling R10.6b) : `current_project()`
    ne doit jamais continuer à pointer vers un dossier qui n'existe plus
    (supprimé entre deux rendus de page) — sinon le reste de l'app agit
    sur un projet fantôme. Effet de bord assumé : le getter nettoie
    lui-même le session_state quand il détecte l'incohérence."""
    projet = tmp_path / "sera-supprime"
    projet.mkdir()
    st.session_state["current_project_path"] = str(projet)

    assert C.current_project() == projet

    shutil.rmtree(projet)
    assert C.current_project() is None
    assert "current_project_path" not in st.session_state


def test_current_project_none_sans_session_state():
    st.session_state.pop("current_project_path", None)
    assert C.current_project() is None


def test_current_project_ok_si_dossier_present(tmp_path):
    projet = tmp_path / "existe"
    projet.mkdir()
    st.session_state["current_project_path"] = str(projet)
    assert C.current_project() == projet
    # Toujours là : pas d'effacement injustifié.
    assert st.session_state["current_project_path"] == str(projet)
