"""Page Configuration — provisoire.

L'ancienne version référençait une racine de données globale et une racine
de projets VAME globale (`data_root()`, `vame_projects_root()`,
`st.session_state.vame_projects_root`) : ces notions ont disparu avec le
passage à des projets autonomes, chacun avec ses propres dossiers sous
`<projet>/data/`. Cette page se limite donc, pour l'instant, à afficher les
racines de préférence (projets, modèles DLC) et le projet courant. Refonte
complète prévue à la tâche 23.
"""
from __future__ import annotations

import streamlit as st

from lib.config import current_project, models_root, projects_root
from lib.icons import ACCENT, lucide_html, lucide_title


def render() -> None:
    st.title("Configuration")
    st.caption("Chemins de préférence de l'application")

    st.markdown("---")

    st.markdown(lucide_title("database", "Racines"), unsafe_allow_html=True)

    racines = {
        "Projets EthoFlow": projects_root(),
        "Modèles DLC": models_root(),
    }
    for label, chemin in racines.items():
        icone = (
            lucide_html("circle-check", 14, ACCENT)
            if chemin.exists()
            else lucide_html("circle-x", 14, "#ef4444")
        )
        st.markdown(f"{icone} **{label}** — `{chemin}`", unsafe_allow_html=True)

    projet = current_project()
    if projet:
        st.markdown("---")
        st.markdown(f"**Projet courant :** `{projet}`")

    st.markdown("---")
    st.info(
        "Cette page sera étoffée (édition des racines, préférences par "
        "projet) à une tâche ultérieure."
    )
