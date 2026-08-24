"""Branchement du projet courant sur `lib/project.py`, et vocabulaire.

Seul module de `lib/` autorisé à importer Streamlit : il lit le projet
courant dans le `session_state`. Toute la logique testable vit dans
`lib/project.py`.

Les 22 noms ré-exportés de `lib.project` (SCRIPTS_DIR, arena_coords, etc.)
sont une commodité pour la couche vue : elle importe d'ici plutôt que d'avoir
deux imports. `lib/project.py` en reste propriétaire et lieu de test — aucune
modification ne doit leur être apportée ici.
"""
from __future__ import annotations

from pathlib import Path

import streamlit as st

from lib.motif_labels import categories  # noqa: F401  (ré-export)
from lib.project import (  # noqa: F401  (ré-exports pour les vues)
    DEFAULT_MODELS_ROOT,
    DEFAULT_PROJECTS_ROOT,
    SCRIPTS_DIR,
    arena_coords,
    cropped_dir,
    cropped_videos_exist,
    dlc_config_path,
    dlc_config_status,
    dlc_output_dir,
    excel_path,
    list_dlc_models,
    list_projects,
    load_prefs,
    models_root,
    project_kind,
    projects_root,
    px_per_cm,
    read_pipeline_config,
    save_prefs,
    set_arena_coords,
    set_dlc_config,
    set_px_per_cm,
)

_CLE = "current_project_path"


def current_project() -> Path | None:
    """Projet actuellement ouvert, ou `None` s'il n'y en a pas — ou plus.

    Auto-guérison délibérée (ruling R10.6b) : si le chemin mémorisé ne
    pointe plus vers un dossier (projet supprimé entre-temps, par exemple
    via la page Projet elle-même), on nettoie le `session_state` ici même
    plutôt que de laisser le reste de l'app agir sur un projet fantôme.
    Un getter avec effet de bord n'est pas anodin, mais l'alternative —
    laisser `current_project_path` pointer vers du vide — est ce qui
    permettait à un projet supprimé de « revivre » (ruling R10.6, Critical
    « projet ressuscité »). Coût accepté : un projet sur un disque
    temporairement démonté se lira comme fermé, ce qui vaut mieux qu'un
    projet supprimé à moitié vivant.
    """
    valeur = st.session_state.get(_CLE)
    if not valeur:
        return None
    chemin = Path(valeur)
    if not chemin.is_dir():
        st.session_state.pop(_CLE, None)
        return None
    return chemin


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
    """À appeler en tête de toute vue qui a besoin d'un projet.

    N'a pas de retour si aucun projet n'est ouvert : st.stop() arrête le script.
    """
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
        "transition", "ambiguous", "immobility (imputed)",
    ],
}
