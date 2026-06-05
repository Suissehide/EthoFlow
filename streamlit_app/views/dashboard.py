"""Page Tableau de bord — sélection/création de projet + inventaire des sessions."""
from __future__ import annotations

import shutil
from pathlib import Path

import streamlit as st

from lib.config import (
    create_project,
    current_project_name,
    list_projects,
    projects_root,
)
from lib.icons import ACCENT, ACCENT_BG, lucide_html, lucide_title
from lib.sessions import list_sessions


# ============================================================
# Project selector / creator
# ============================================================

def _section_project() -> None:
    existing = list_projects()
    current = st.session_state.get("current_project_path")
    just_created = st.session_state.pop("_project_just_created", None)

    # ---- Confirmation de création ----
    if just_created:
        st.success(f"Projet **{just_created}** créé avec succès")

    # Style boutons
    st.markdown(
        """<style>
        .st-key-btn_delete_project button {
            background: transparent !important;
            color: #ef4444 !important;
            border: 1px solid #ef4444 !important;
        }
        .st-key-btn_delete_project button:hover {
            background: rgba(239,68,68,0.1) !important;
        }
        .st-key-btn_create_project button {
            background: #3b82f6 !important;
            color: #fff !important;
            border: none !important;
        }
        .st-key-btn_create_project button:hover {
            background: #2563eb !important;
        }
        .st-key-delete_actions [data-testid="stVerticalBlock"],
        .st-key-delete_actions [data-testid="stVerticalBlock"] > div {
            display: flex !important;
            flex-direction: row !important;
            gap: 0.5rem !important;
            flex-wrap: wrap !important;
        }
        .st-key-delete_actions button {
            width: auto !important;
        }
        </style>""",
        unsafe_allow_html=True,
    )

    col_open, col_create = st.columns(2)

    # ---- Colonne gauche : projet ----
    with col_open:
        st.markdown(lucide_title("folder-open", "Ouvrir un projet"), unsafe_allow_html=True)

        if current:
            st.markdown(f'<p style="font-size:0.8rem;color:#6b7280;margin-bottom:6px"><code>{current}</code></p>', unsafe_allow_html=True)

        if existing:
            names = [p.name for p in existing]
            default_idx = 0
            if current:
                current_name = Path(current).name
                if current_name in names:
                    default_idx = names.index(current_name)

            selected_name = st.selectbox(
                "Sélectionner un projet",
                options=names,
                index=default_idx,
                key="project_selector",
                label_visibility="collapsed",
            )
            if st.button("Supprimer", key="btn_delete_project"):
                st.session_state["_confirm_delete"] = selected_name

            selected_path = projects_root() / selected_name

            if str(selected_path) != current:
                st.session_state.current_project_path = str(selected_path)
                st.rerun()

            # ---- Confirmation de suppression ----
            if st.session_state.get("_confirm_delete"):
                name_to_delete = st.session_state["_confirm_delete"]
                st.warning(
                    f"Supprimer le projet **{name_to_delete}** et toutes ses données ? "
                    "Cette action est irréversible."
                )
                with st.container(key="delete_actions"):
                    if st.button("Annuler", key="btn_cancel_delete"):
                        st.session_state.pop("_confirm_delete", None)
                        st.rerun()
                    if st.button("Oui, supprimer", key="btn_confirm_delete", type="primary"):
                        path_to_delete = projects_root() / name_to_delete
                        if path_to_delete.exists():
                            shutil.rmtree(path_to_delete)
                        if current and Path(current).name == name_to_delete:
                            st.session_state.pop("current_project_path", None)
                        st.session_state.pop("_confirm_delete", None)
                        st.toast(f"Projet '{name_to_delete}' supprimé")
                        st.rerun()
        else:
            st.caption("Aucun projet existant.")

    # ---- Colonne droite : Créer un nouveau projet ----
    with col_create:
        st.markdown(lucide_title("folder-open", "Créer un nouveau projet"), unsafe_allow_html=True)

        new_name = st.text_input(
            "Nom du projet",
            placeholder="ex: pilote-2026",
            key="new_project_name",
        )
        if st.button("Créer le projet", key="btn_create_project", type="primary"):
            if new_name and new_name.strip():
                clean = new_name.strip().replace(" ", "-")
                project_dir = create_project(clean)
                st.session_state.current_project_path = str(project_dir)
                st.session_state["_project_just_created"] = clean
                st.session_state.pop("new_project_name", None)
                st.rerun()
            else:
                st.warning("Donne un nom au projet.")



# ============================================================
# Session overview
# ============================================================

def _section_sessions() -> None:
    project = current_project_name()
    if not project:
        return

    st.markdown(lucide_title("layout-dashboard", "Sessions"), unsafe_allow_html=True)
    st.caption("Statut d'avancement du pipeline")

    df = list_sessions()
    if df.empty:
        st.info(
            "Aucune session dans ce projet. Va dans **Données** "
            "pour importer depuis Excel ou créer des metadata."
        )
        return

    cols = st.columns(7)
    cols[0].metric("Sessions", len(df))
    cols[1].metric("Vidéos OK", int((df["vidéo"] == "OK").sum()))
    cols[2].metric("DLC", int((df["DLC"] == "OK").sum()))
    cols[3].metric("Split arènes", int((df["split"] == "OK").sum()))
    cols[4].metric("Cleanup", int((df["cleanup"] == "OK").sum()))
    cols[5].metric("VAME", int((df["VAME"] == "OK").sum()))
    cols[6].metric("Validity", int((df["validity"] == "OK").sum()))

    st.dataframe(df, use_container_width=True, hide_index=True)


# ============================================================
# Entry
# ============================================================

def render() -> None:
    st.title("Tableau de bord")
    _section_project()
    st.divider()
    _section_sessions()
