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

from lib.config import current_project_name  # noqa: E402
from lib.icons import ACCENT, ACCENT_BG, ACCENT_HEX, lucide_data_uri  # noqa: E402
from views import (  # noqa: E402
    about,
    configuration,
    donnees,
    pose,
    projet,
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

has_project = current_project_name() is not None

# ============================================================
# Pages
#
# Réduites aux trois pages fonctionnelles au début de ce chantier : les
# autres vues (Détails session, Lancer pipeline, Labellisation VAME,
# Résultats) reviennent une par une, chacune ajoutée par la tâche qui
# l'adapte aux modules `lib/` actuels. Pour enregistrer une page ici :
# l'ajouter à PAGES (ou PROJECT_PAGES si elle exige un projet ouvert), avec
# une icône Lucide (voir lib/icons.ICONS) et une clé unique.
# ============================================================
PAGES: list[dict] = [
    {"name": "Projet",             "icon": "layout-dashboard", "render": projet.render,         "key": "projet"},
]

# Pages requiring a project
PROJECT_PAGES: list[dict] = [
    {"name": "Données",            "icon": "file-spreadsheet", "render": donnees.render,        "key": "donnees"},
    {"name": "Pose (DLC)",         "icon": "scan-line",        "render": pose.render,            "key": "pose"},
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

_ALL_PAGES = {p["name"]: p for p in _visible_main + BOTTOM_PAGES}

# If current page requires project but no project → redirect to Projet
if st.session_state.current_page not in _ALL_PAGES:
    st.session_state.current_page = "Projet"

# ============================================================
# Per-button CSS rules
# ============================================================
_btn_css = ""
for p in _visible_main + BOTTOM_PAGES:
    k = p["key"]
    active = st.session_state.current_page == p["name"]
    uri_default = lucide_data_uri(p["icon"], "6b7280")
    uri_active = lucide_data_uri(p["icon"], ACCENT_HEX)
    uri_hover = lucide_data_uri(p["icon"], "d1d5db")

    _btn_css += f"""
    .st-key-nav_{k} button::before {{
        content: "";
        display: inline-block;
        width: 18px; height: 18px;
        min-width: 18px;
        margin-right: 10px;
        background: url("{uri_active if active else uri_default}") no-repeat center/contain;
    }}
    """
    if active:
        _btn_css += f"""
        .st-key-nav_{k} button {{
            background: {ACCENT_BG};
            color: {ACCENT};
            font-weight: 600;
        }}
        """
    else:
        _btn_css += f"""
        .st-key-nav_{k} button {{
            background: transparent;
            color: #9ca3af;
            font-weight: 500;
        }}
        .st-key-nav_{k} button:hover {{
            background: rgba(255,255,255,0.04);
            color: #d1d5db;
        }}
        .st-key-nav_{k} button:hover::before {{
            background-image: url("{uri_hover}");
        }}
        """

# ============================================================
# Sidebar CSS
# ============================================================
st.sidebar.markdown(
    f"""
    <style>
    section[data-testid="stSidebar"] > div:first-child {{
        background: #111827;
        padding-top: 0 !important;
    }}
    /* ---- Hide default sidebar header ---- */
    [data-testid="stSidebarHeader"] {{
        display: none !important;
    }}
    .ef-brand {{
        padding: 0.8rem 1rem 0 1rem;
    }}
    .ef-brand-text {{
        font-size: 1.6rem;
        font-weight: 800;
        color: #f9fafb;
        letter-spacing: -0.03em;
    }}
    .ef-brand-sub {{
        font-size: 0.72rem;
        color: #6b7280;
        padding: 0.1rem 1rem 0.3rem 1rem;
        margin: 0;
    }}
    .ef-project-badge {{
        display: flex;
        align-items: center;
        gap: 8px;
        font-size: 0.82rem;
        font-weight: 600;
        color: #f9fafb;
        background: linear-gradient(135deg, rgba(16,185,129,0.15), rgba(16,185,129,0.08));
        border: 1px solid rgba(16,185,129,0.25);
        padding: 0.45rem 0.75rem;
        margin: 0 0.75rem 0.7rem 0.75rem;
        border-radius: 8px;
    }}
    .ef-project-badge svg {{
        flex-shrink: 0;
    }}
    .ef-no-project {{
        font-size: 0.78rem;
        color: #6b7280;
        font-style: italic;
        padding: 0.3rem 1rem 0.7rem 1rem;
        margin: 0;
    }}
    .ef-section-label {{
        font-size: 0.65rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #4b5563;
        padding: 0.5rem 1rem 0.2rem 1rem;
        margin: 0;
    }}
    .ef-divider {{
        border: none;
        border-top: 1px solid #1f2937;
        margin: 0.35rem 1rem;
    }}
    .st-key-nav_main [data-testid="stVerticalBlock"],
    .st-key-nav_bottom [data-testid="stVerticalBlock"] {{
        gap: 0 !important;
    }}
    section[data-testid="stSidebar"] [class*="st-key-nav_"],
    section[data-testid="stSidebar"] [class*="st-key-nav_"] > div,
    section[data-testid="stSidebar"] [class*="st-key-nav_"] > div > div {{
        margin: 0 !important;
        padding: 0 !important;
        gap: 0 !important;
    }}
    section[data-testid="stSidebar"] [class*="st-key-nav_"] button > div {{
        justify-content: flex-start !important;
    }}
    section[data-testid="stSidebar"] [class*="st-key-nav_"] button {{
        width: 100%;
        display: flex;
        align-items: center;
        text-align: left;
        justify-content: flex-start;
        padding: 0.45rem 0.75rem;
        margin: 0;
        font-size: 0.875rem;
        border: none;
        border-radius: 6px;
        box-shadow: none;
        transition: background 0.12s ease, color 0.12s ease;
        cursor: pointer;
    }}
    {_btn_css}
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# Sidebar — Brand + Project badge
# ============================================================
st.sidebar.markdown(
    '<div class="ef-brand"><span class="ef-brand-text">EthoFlow</span></div>',
    unsafe_allow_html=True,
)
st.sidebar.markdown(
    '<p class="ef-brand-sub">Analyse comportementale automatisée</p>',
    unsafe_allow_html=True,
)

project_name = current_project_name()
if project_name:
    folder_icon = lucide_data_uri("folder-open", ACCENT_HEX)
    st.sidebar.markdown(
        f'<div class="ef-project-badge">'
        f'<img src="{folder_icon}" width="16" height="16" />'
        f'{project_name}</div>',
        unsafe_allow_html=True,
    )
else:
    st.sidebar.markdown(
        '<p class="ef-no-project">Aucun projet sélectionné</p>',
        unsafe_allow_html=True,
    )

# ============================================================
# Sidebar — Navigation
# ============================================================
st.sidebar.markdown('<p class="ef-section-label">Navigation</p>', unsafe_allow_html=True)

with st.sidebar.container(key="nav_main"):
    for page in _visible_main:
        if st.button(page["name"], key=f"nav_{page['key']}", use_container_width=True):
            st.session_state.current_page = page["name"]
            st.rerun()

st.sidebar.markdown('<hr class="ef-divider">', unsafe_allow_html=True)
st.sidebar.markdown('<p class="ef-section-label">Système</p>', unsafe_allow_html=True)

with st.sidebar.container(key="nav_bottom"):
    for page in BOTTOM_PAGES:
        if st.button(page["name"], key=f"nav_{page['key']}", use_container_width=True):
            st.session_state.current_page = page["name"]
            st.rerun()

# ============================================================
# Dispatch
# ============================================================
_ALL_PAGES[st.session_state.current_page]["render"]()
