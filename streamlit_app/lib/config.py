"""Constantes globales et configuration de l'app EthoFlow Streamlit.

Les chemins de données sont dynamiques : ils dépendent du projet courant
stocké dans `st.session_state.current_project_path`.

Les constantes statiques (ROOT, SCRIPTS_DIR, etc.) restent calculées depuis
l'emplacement de ce fichier.
"""
from __future__ import annotations

from pathlib import Path

import streamlit as st

# ============================================================
# Chemins statiques du repo
# ============================================================
ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPTS_DIR = ROOT / "scripts"
CONFIG_POINTER = ROOT / ".vame_config_path"

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

def _project_data() -> Path:
    """Racine data/ du projet courant."""
    p = st.session_state.get("current_project_path")
    if p:
        return Path(p) / "data"
    # Fallback legacy (si pas de projet sélectionné)
    return ROOT / "data"


def data_root() -> Path:
    return _project_data()

def raw_dir() -> Path:
    return _project_data() / "raw"

def cropped_dir() -> Path:
    return _project_data() / "cropped"

def dlc_output_dir() -> Path:
    return _project_data() / "dlc-output"

def vame_input_dir() -> Path:
    return _project_data() / "vame-input"

def vame_output_dir() -> Path:
    return _project_data() / "vame-output"

def current_project_name() -> str | None:
    """Nom du projet courant, ou None."""
    p = st.session_state.get("current_project_path")
    if p:
        return Path(p).name
    return None


# Aliases pour rétrocompatibilité (lecture seule, évaluées à l'import)
# IMPORTANT: utiliser les fonctions ci-dessus dans le nouveau code.
DATA_ROOT = ROOT / "data"
RAW_DIR = DATA_ROOT / "raw"
CROPPED_DIR = DATA_ROOT / "cropped"
DLC_OUTPUT_DIR = DATA_ROOT / "dlc-output"
VAME_INPUT_DIR = DATA_ROOT / "vame-input"
VAME_OUTPUT_DIR = DATA_ROOT / "vame-output"


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
    subdirs = ["raw", "cropped", "dlc-output", "vame-input", "vame-output"]
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
