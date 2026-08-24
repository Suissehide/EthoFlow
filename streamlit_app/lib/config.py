"""Branchement du projet courant sur `lib/project.py`, et vocabulaire.

Seul module de `lib/` autorisé à importer Streamlit : il lit le projet
courant dans le `session_state`. Toute la logique testable vit dans
`lib/project.py`.
"""
from __future__ import annotations

from pathlib import Path

import streamlit as st

from lib.motif_labels import categories  # noqa: F401  (ré-export)
from lib.project import (  # noqa: F401  (ré-exports pour les vues)
    SCRIPTS_DIR,
    arena_coords,
    dlc_config_path,
    list_dlc_models,
    list_projects,
    load_prefs,
    models_root,
    project_kind,
    projects_root,
    px_per_cm,
    read_pipeline_config,
    save_prefs,
)

_CLE = "current_project_path"


def current_project() -> Path | None:
    valeur = st.session_state.get(_CLE)
    return Path(valeur) if valeur else None


def current_project_name() -> str | None:
    projet = current_project()
    return projet.name if projet else None


def set_current_project(path: Path | str | None) -> None:
    if path is None:
        st.session_state.pop(_CLE, None)
        return
    st.session_state[_CLE] = str(Path(path))
    prefs = load_prefs()
    prefs["last_project"] = str(Path(path))
    save_prefs(prefs)


def require_project() -> Path:
    """À appeler en tête de toute vue qui a besoin d'un projet."""
    projet = current_project()
    if projet is None:
        st.warning("Ouvre un projet depuis la page **Projet**.")
        st.stop()
    return projet


# Exemples pour aider à remplir le champ `label`, qui est libre. À ne pas
# confondre avec `categories()` : liste fermée écrite dans `category` et
# utilisée par les analyses pour grouper.
VOCABULAIRE_SUGGERE: dict[str, list[str]] = {
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
