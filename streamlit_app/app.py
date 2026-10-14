"""EthoFlow — interface Streamlit.

Voir docs/ETHOFLOW.md pour le pipeline complet.

Lancement :
    conda activate ethoflow
    streamlit run streamlit_app/app.py

Sur le LAN : ajouter --server.address=0.0.0.0
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from lib.config import current_project, current_project_name  # noqa: E402
from lib.icons import lucide_data_uri  # noqa: E402
from views import (  # noqa: E402
    about,
    analyses,
    configuration,
    donnees,
    motifs,
    nettoyage,
    pose,
    projet,
    qualite_dlc,
    vame_page,
    videos,
    visualisations,
)

st.set_page_config(
    page_title="EthoFlow",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# Init session state
# ============================================================
if "current_page" not in st.session_state:
    st.session_state.current_page = "Projet"

# Restore the last project at startup, if it still exists
if "current_project_path" not in st.session_state:
    from lib.project import load_prefs
    dernier = load_prefs().get("last_project")
    if dernier and Path(dernier).is_dir():
        from lib.config import set_current_project
        set_current_project(dernier)

has_project = current_project_name() is not None

# ============================================================
# Pages
#
# Les neuf pages du pipeline sont toutes enregistrées : Projet (PAGES) puis
# Données, Vidéos & calibration, Pose (DLC), Nettoyage, VAME, Motifs,
# Analyses, Visualisations (PROJECT_PAGES, visibles seulement si un projet
# est ouvert). Configuration et À propos sont des pages système
# (BOTTOM_PAGES). Pour enregistrer une nouvelle page ici : l'ajouter à
# PAGES (ou PROJECT_PAGES si elle exige un projet ouvert), avec une icône
# Lucide (voir lib/icons.ICONS) et une clé unique.
# ============================================================
PAGES: list[dict] = [
    {"name": "Projet",             "icon": "layout-dashboard", "render": projet.render,         "key": "projet"},
]

# Pages requiring a project
PROJECT_PAGES: list[dict] = [
    {"name": "Données",            "icon": "file-spreadsheet", "render": donnees.render,        "key": "donnees"},
    {"name": "Vidéos & calibration", "icon": "video",           "render": videos.render,          "key": "videos"},
    {"name": "Pose (DLC)",         "icon": "scan-line",        "render": pose.render,            "key": "pose"},
    {"name": "Nettoyage",          "icon": "brush-cleaning",   "render": nettoyage.render,       "key": "nettoyage"},
    {"name": "VAME",               "icon": "waypoints",        "render": vame_page.render,       "key": "vame"},
    {"name": "Motifs",             "icon": "tags",             "render": motifs.render,          "key": "motifs"},
    {"name": "Analyses",           "icon": "chart-column",     "render": analyses.render,        "key": "analyses"},
    {"name": "Visualisations",     "icon": "clapperboard",     "render": visualisations.render,  "key": "visualisations"},
]

# Outils hors des 9 étapes, sous elles dans le rail.
OUTILS_PAGES: list[dict] = [
    {"name": "Qualité DLC",        "icon": "gauge",            "render": qualite_dlc.render,     "key": "qualite"},
]

BOTTOM_PAGES: list[dict] = [
    {"name": "Configuration",      "icon": "settings",         "render": configuration.render,   "key": "config"},
    {"name": "À propos",           "icon": "info",             "render": about.render,           "key": "about"},
]

# Visible pages depend on project state
if has_project:
    _visible_main = PAGES + PROJECT_PAGES
else:
    _visible_main = PAGES

_ALL_PAGES = {p["name"]: p for p in _visible_main + BOTTOM_PAGES
              + (OUTILS_PAGES if has_project else [])}

# If current page requires project but no project → redirect to Projet
if st.session_state.current_page not in _ALL_PAGES:
    st.session_state.current_page = "Projet"

# ============================================================
# Style global (refonte 1c) — thème de base dans .streamlit/config.toml
# ============================================================
st.markdown(f"<style>{(_HERE / 'style.css').read_text(encoding='utf-8')}</style>",
            unsafe_allow_html=True)

# ============================================================
# Rail d'icônes (sidebar Streamlit, 72 px)
#
# Une icône par page ; le nom est l'infobulle (`help`) et le libellé du
# bouton, masqué en CSS — il reste lu par les lecteurs d'écran et par les
# tests. L'icône est passée en variable CSS par bouton (blanche sur fond
# #34322e pour la page active).
# ============================================================
_rail_css = ""
for _p in _visible_main + OUTILS_PAGES + BOTTOM_PAGES:
    _actif = st.session_state.current_page == _p["name"]
    _icone = lucide_data_uri(_p["icon"], "ffffff" if _actif else "8a867e")
    _rail_css += (f'.st-key-nav_{_p["key"]} button {{ --ef-icone: url("{_icone}");'
                  + (" background: #34322e !important;" if _actif else "") + " }\n")
st.sidebar.markdown(f"<style>{_rail_css}</style>", unsafe_allow_html=True)


def _bouton_rail(page: dict) -> None:
    if st.button(page["name"], key=f"nav_{page['key']}", help=page["name"]):
        st.session_state.current_page = page["name"]
        st.rerun()


with st.sidebar:
    st.markdown('<p class="ef-rail-marque">Ef</p>', unsafe_allow_html=True)
    for _p in _visible_main:
        _bouton_rail(_p)
    if has_project:
        with st.container(key="rail_outils"):
            for _p in OUTILS_PAGES:
                _bouton_rail(_p)
    with st.container(key="rail_bas"):
        for _p in BOTTOM_PAGES:
            _bouton_rail(_p)

# ============================================================
# Colonne Activité (projet ouvert uniquement)
# ============================================================
if has_project:
    from lib import etapes as _E
    from lib import runner as _runner
    from views import _activite

    _projet_courant = current_project()
    _job = _runner.current(_projet_courant) if _runner.is_running(_projet_courant) else None
    st.session_state["_etat_pipeline"] = _E.etat_pipeline(
        _projet_courant, _job.script if _job else None)
    _activite.render(_projet_courant, st.session_state["_etat_pipeline"])

# ============================================================
# Dispatch
# ============================================================
_ALL_PAGES[st.session_state.current_page]["render"]()
