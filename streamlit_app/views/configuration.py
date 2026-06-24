"""Page Configuration — chemins des données et projets VAME."""
from __future__ import annotations

from pathlib import Path

import streamlit as st

from lib.config import data_root, DEFAULT_VAME_PROJECTS_ROOT, current_project_name
from lib.icons import lucide_html, lucide_title, ACCENT


def _folder_picker(label: str, default: str, key: str) -> str:
    """Affiche un champ texte + bouton pour naviguer dans les dossiers."""
    col1, col2 = st.columns([4, 1])
    current = col1.text_input(label, value=default, key=f"{key}_input")

    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Parcourir", key=f"{key}_browse"):
            st.session_state[f"{key}_browsing"] = True

    if st.session_state.get(f"{key}_browsing", False):
        selected = _browse_folders(current, key)
        if selected is not None:
            st.session_state[f"{key}_browsing"] = False
            st.session_state[f"{key}_input"] = selected
            st.rerun()

        if st.button("Annuler", key=f"{key}_cancel"):
            st.session_state[f"{key}_browsing"] = False
            st.rerun()

    return current


def _browse_folders(start_path: str, key: str) -> str | None:
    """Navigateur de dossiers intégré."""
    current_dir = Path(start_path)
    if not current_dir.is_dir():
        current_dir = Path.home()

    browse_key = f"{key}_browse_dir"
    if browse_key in st.session_state:
        current_dir = Path(st.session_state[browse_key])

    st.markdown(f"**Dossier actuel :** `{current_dir}`")

    col_up, col_select = st.columns(2)
    with col_up:
        if current_dir.parent != current_dir:
            if st.button("Dossier parent", key=f"{key}_parent"):
                st.session_state[browse_key] = str(current_dir.parent)
                st.rerun()
    with col_select:
        if st.button("Choisir ce dossier", key=f"{key}_select", type="primary"):
            result = str(current_dir)
            if browse_key in st.session_state:
                del st.session_state[browse_key]
            return result

    try:
        subdirs = sorted(
            [d for d in current_dir.iterdir() if d.is_dir() and not d.name.startswith(".")],
            key=lambda d: d.name.lower(),
        )
    except PermissionError:
        st.warning("Accès refusé à ce dossier.")
        return None

    if not subdirs:
        st.info("Aucun sous-dossier.")
    else:
        for subdir in subdirs[:30]:
            if st.button(subdir.name, key=f"{key}_sub_{subdir.name}"):
                st.session_state[browse_key] = str(subdir)
                st.rerun()
        if len(subdirs) > 30:
            st.caption(f"… et {len(subdirs) - 30} autres dossiers")

    return None


def render() -> None:
    st.title("Configuration")
    st.caption("Chemins des données et paramètres du projet")

    st.markdown("---")

    # ---- Section Données ----
    st.markdown(lucide_title("database", "Données"), unsafe_allow_html=True)

    st.markdown(f"**Racine des données :** `{data_root()}`")
    if data_root().exists():
        st.success(f"Le dossier existe — {sum(1 for _ in data_root().iterdir() if _.is_dir())} sous-dossiers")
    else:
        st.error("Ce dossier n'existe pas.")

    st.markdown("---")

    # ---- Section VAME ----
    st.markdown(lucide_title("brain", "Projets VAME"), unsafe_allow_html=True)

    new_root = _folder_picker(
        "Racine des projets VAME",
        st.session_state.vame_projects_root,
        "vame_root",
    )

    if new_root != st.session_state.vame_projects_root:
        st.session_state.vame_projects_root = new_root

    vame_path = Path(st.session_state.vame_projects_root)
    if vame_path.exists():
        projects = [d.name for d in vame_path.iterdir() if d.is_dir() and (d / "config.yaml").exists()]
        if projects:
            st.success(f"{len(projects)} projet(s) VAME détecté(s)")
            for p in projects:
                st.markdown(f"  - `{p}`")
        else:
            st.warning("Aucun projet VAME trouvé dans ce dossier (pas de config.yaml).")
    else:
        st.warning("Le dossier n'existe pas encore. Il sera créé quand vous lancerez VAME.")

    st.markdown("---")

    # ---- Résumé des chemins ----
    st.markdown(lucide_title("clipboard-list", "Résumé des chemins"), unsafe_allow_html=True)
    from lib.config import raw_dir, cropped_dir, dlc_output_dir, vame_dir, SCRIPTS_DIR

    paths = {
        "Données (racine)": data_root(),
        "Sessions brutes": raw_dir(),
        "Vidéos cropped": cropped_dir(),
        "Sorties DLC": dlc_output_dir(),
        "Projets VAME (projet)": vame_dir(),
        "Scripts": SCRIPTS_DIR,
        "Projets VAME (legacy)": vame_path,
    }
    for label, path in paths.items():
        icon = lucide_html("circle-check", 14, ACCENT) if path.exists() else lucide_html("circle-x", 14, "#ef4444")
        st.markdown(f"{icon} **{label}** — `{path}`", unsafe_allow_html=True)
