"""Chemins projet et préférences d'interface — sans Streamlit.

Ce module ne connaît que des `Path`. Toute la logique testable de
localisation vit ici ; `lib/config.py` se contente d'y brancher le projet
courant lu dans le `session_state`.

La résolution structurelle (`<projet>/data/raw/` etc.) vient de
`scripts/paths.py`, source unique de vérité partagée avec les CLI.
"""
from __future__ import annotations

import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPTS_DIR = ROOT / "scripts"

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import paths as _paths  # noqa: E402
from interactive import (  # noqa: E402
    DEFAULT_MODELS_ROOT,
    DEFAULT_PROJECTS_ROOT,
)

# Préférences d'interface uniquement (racines, dernier projet ouvert).
# Jamais lues par les scripts CLI.
PREFS_PATH = Path.home() / ".ethoflow" / "app_prefs.yaml"


# ---------------------------------------------------------------- préférences

def prefs_path() -> Path:
    return PREFS_PATH


def load_prefs() -> dict:
    if not PREFS_PATH.exists():
        return {}
    try:
        return yaml.safe_load(PREFS_PATH.read_text(encoding="utf-8")) or {}
    except Exception:
        # Un fichier de préférences corrompu ne doit pas empêcher de
        # démarrer l'app : on repart des défauts.
        return {}


def save_prefs(prefs: dict) -> None:
    PREFS_PATH.parent.mkdir(parents=True, exist_ok=True)
    PREFS_PATH.write_text(
        yaml.safe_dump(prefs, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def projects_root() -> Path:
    return Path(load_prefs().get("projects_root", DEFAULT_PROJECTS_ROOT))


def models_root() -> Path:
    return Path(load_prefs().get("models_root", DEFAULT_MODELS_ROOT))


# ------------------------------------------------------------------ inventaire

def list_projects(root: Path) -> list[Path]:
    """Dossiers de `root` qui ressemblent à un projet EthoFlow."""
    root = Path(root)
    if not root.is_dir():
        return []
    return sorted(d for d in root.iterdir() if d.is_dir() and (d / "data").is_dir())


def list_dlc_models(root: Path) -> list[Path]:
    """Dossiers de `root` contenant un `config.yaml` DLC."""
    root = Path(root)
    if not root.is_dir():
        return []
    return sorted(
        d for d in root.iterdir() if d.is_dir() and (d / "config.yaml").is_file()
    )


# --------------------------------------------------------- pipeline_config.yaml

def read_pipeline_config(project: Path) -> dict:
    """Contenu de `configs/pipeline_config.yaml`, `{}` s'il n'existe pas."""
    cfg = _paths.pipeline_config_path(Path(project))
    if not cfg.exists():
        return {}
    try:
        return yaml.safe_load(cfg.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}


def dlc_config_path(project: Path) -> str | None:
    value = read_pipeline_config(project).get("dlc_project_config")
    return str(value) if value else None


def project_kind(project: Path) -> str:
    """'single' ou 'multi'. 'single' par défaut : pas d'arena splitting."""
    kind = read_pipeline_config(project).get("kind")
    return kind if kind in ("single", "multi") else "single"


def px_per_cm(project: Path) -> float | None:
    value = read_pipeline_config(project).get("px_per_cm")
    return float(value) if value is not None else None


def arena_coords(project: Path) -> dict[str, list[int]]:
    return read_pipeline_config(project).get("default_arenes_coords") or {}
