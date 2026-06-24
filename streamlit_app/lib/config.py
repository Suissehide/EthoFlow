"""Constantes globales et configuration de l'app EthoFlow Streamlit.

Les chemins de données sont dynamiques : ils dépendent du projet courant
stocké dans `st.session_state.current_project_path`.

La résolution structurelle (mapping `<project>/data/raw/` etc.) vient du
module partagé `scripts/paths.py` — source unique de vérité commune entre
les CLI et cette app Streamlit. Les wrappers ci-dessous se contentent de
lire `current_project_path` dans le `session_state` puis de déléguer à
`paths.<fn>(project)`.
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st


# ============================================================
# Chemins statiques du repo
# ============================================================
ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPTS_DIR = ROOT / "scripts"
CONFIG_POINTER = ROOT / ".vame_config_path"

# Import du module partagé paths.py qui vit dans scripts/
# (pas un package — on insère son dossier dans sys.path)
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
import paths as _paths  # noqa: E402

# Racine par défaut des projets EthoFlow
DEFAULT_PROJECTS_ROOT = Path.home() / "ethoflow" / "projects"

# Racine par défaut des projets VAME (legacy, utilisé si pas de projet)
DEFAULT_VAME_PROJECTS_ROOT = Path.home() / "Inserm" / "vame-projects"

# Mapping logique → nom d'environnement conda
CONDA_ENVS: dict[str, str] = {
    "dlc": "dlc",
    "vame": "vame",
    "ethoflow": "ethoflow",
}


# ============================================================
# Chemins dynamiques — basés sur le projet courant
# ============================================================

def _current_project() -> Path:
    """Racine du projet courant, fallback legacy ROOT si rien n'est sélectionné."""
    p = st.session_state.get("current_project_path")
    if p:
        return Path(p)
    # Fallback legacy (si pas de projet sélectionné)
    return ROOT


def data_root() -> Path:
    return _paths.data_dir(_current_project())

def raw_dir() -> Path:
    return _paths.raw_dir(_current_project())

def cropped_dir() -> Path:
    return _paths.cropped_dir(_current_project())

def dlc_output_dir() -> Path:
    return _paths.dlc_output_dir(_current_project())

def vame_dir() -> Path:
    return _paths.vame_dir(_current_project())

def cleaned_h5_path(session_id: str) -> Path:
    return _paths.cleaned_h5_path(_current_project(), session_id)

def results_dir() -> Path:
    return _paths.results_dir(_current_project())

def pipeline_config_path() -> Path:
    return _paths.pipeline_config_path(_current_project())

def current_project_name() -> str | None:
    """Nom du projet courant, ou None."""
    p = st.session_state.get("current_project_path")
    if p:
        return Path(p).name
    return None


# Aliases pour rétrocompatibilité (lecture seule, évaluées à l'import).
# IMPORTANT: utiliser les fonctions ci-dessus dans le nouveau code — ces
# constantes ne suivent pas le projet courant et restent figées sur la
# racine legacy.
DATA_ROOT = _paths.data_dir(ROOT)
RAW_DIR = _paths.raw_dir(ROOT)
CROPPED_DIR = _paths.cropped_dir(ROOT)
DLC_OUTPUT_DIR = _paths.dlc_output_dir(ROOT)
VAME_DIR = _paths.vame_dir(ROOT)


# ============================================================
# Projets EthoFlow
# ============================================================

def projects_root() -> Path:
    """Racine des projets, lue depuis session_state avec fallback."""
    return Path(
        st.session_state.get("projects_root", str(DEFAULT_PROJECTS_ROOT))
    )


def list_projects() -> list[Path]:
    """Liste les dossiers de projets existants."""
    root = projects_root()
    if not root.exists():
        return []
    return sorted(
        d for d in root.iterdir()
        if d.is_dir() and (d / "data").is_dir()
    )


def create_project(name: str) -> Path:
    """Crée un nouveau projet avec la structure de dossiers standard."""
    project_dir = projects_root() / name
    subdirs = ["raw", "cropped", "dlc-output", "vame"]
    for sub in subdirs:
        (project_dir / "data" / sub).mkdir(parents=True, exist_ok=True)
    return project_dir


# ============================================================
# VAME projects root
# ============================================================

def vame_projects_root() -> Path:
    """Racine des projets VAME, lue depuis `st.session_state` avec fallback."""
    return Path(
        st.session_state.get("vame_projects_root", str(DEFAULT_VAME_PROJECTS_ROOT))
    )


# ============================================================
# Vocabulaire éthologique (référence ETHOFLOW.md §6.7)
# ============================================================

ETHOGRAM: dict[str, list[str]] = {
    "Locomotion": [
        "locomotion", "slow locomotion", "fast locomotion", "running",
        "pivoting", "turning", "walking", "trotting", "darting", "circling",
    ],
    "Stationary": [
        "immobility", "freezing", "resting", "crouching",
        "alert immobility", "rest immobility", "vigilance posture", "pause",
    ],
    "Vertical exploration": [
        "rearing supported", "rearing unsupported", "stretch-attend posture",
        "SAP", "half-rear", "elongated stretch",
    ],
    "Sniffing": [
        "sniffing wall", "sniffing floor", "sniffing air", "sniffing (general)",
    ],
    "Grooming": [
        "grooming face", "grooming body", "grooming tail",
        "grooming genital", "scratching", "paw licking",
    ],
    "Exploration": [
        "exploration", "exploration (active)", "exploration (slow)",
        "novelty investigation", "approach", "inspection",
        "head scanning", "wall-following", "nose-poking",
    ],
    "Arena-specific": [
        "thigmotaxis", "center exploration", "corner",
        "transition wall→center", "transition center→wall",
    ],
    "Specific behaviors": [
        "jumping", "digging", "wall climbing",
        "body shake", "arched back", "hunched posture",
    ],
    "Catch-all": [
        "transition", "ambiguous", "artifact", "immobility (imputed)",
    ],
}
