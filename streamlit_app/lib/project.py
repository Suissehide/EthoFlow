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
from sync_from_excel import find_project_excel as _find_project_excel  # noqa: E402

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


def dlc_config_status(project: Path) -> tuple[str, str | None]:
    """Trois états pour le modèle DLC configuré, pas juste présent/absent.

    `run_dlc_inference.py --mode custom` fait lui-même
    `if dlc_cfg and Path(dlc_cfg).exists()` puis échoue vite (mais
    silencieusement, sans dire pourquoi) en `--no-prompt` si le chemin
    configuré n'existe plus — un modèle déplacé ou supprimé se comporte
    alors exactement comme un modèle qui marche du point de vue de la
    page, jusqu'au crash (ruling R12.1). D'où trois états au lieu de deux :

    - `("absent", None)`       — pas de `dlc_project_config` du tout.
    - `("introuvable", chemin)` — une valeur est configurée mais le fichier
      n'existe plus sur disque (déplacé, disque externe débranché, etc.).
    - `("ok", chemin)`         — le fichier existe, la commande peut tourner.
    """
    chemin = dlc_config_path(project)
    if not chemin:
        return "absent", None
    if not Path(chemin).is_file():
        return "introuvable", chemin
    return "ok", chemin


def set_dlc_config(project: Path, dlc_config: str | Path) -> Path:
    """Écrit `dlc_project_config` dans `pipeline_config.yaml`, sans y toucher au reste.

    Même forme que `calibrate_scale.write_scale` pour `px_per_cm` : on lit
    la config existante (dict vide si le fichier n'existe pas encore), on
    ne modifie que la clé visée, on réécrit. Ne PAS repasser par
    `create_project.py --force`, qui régénère `pipeline_config.yaml` en
    entier (perd `default_arenes_coords`) et régénère aussi l'Excel de
    démarrage même s'il a déjà été rempli par le chercheur (ruling R10.2).
    """
    cfg_path = _paths.pipeline_config_path(Path(project))
    cfg = read_pipeline_config(project)
    cfg["dlc_project_config"] = str(dlc_config)
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(
        yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return cfg_path


def _session_a_des_arenes(project: Path) -> bool:
    """Une metadata.yaml de session a-t-elle une liste `arenes` non vide ?

    Vérité de terrain après un sync : c'est exactement ce que consultent
    `crop_arenes.py` et `assign_arenas.py`. Pas d'import de `lib.sessions`
    ici (il importe déjà `lib.project` : ce serait un cycle) — lecture
    directe, minimale, de la même forme.
    """
    rd = _paths.raw_dir(Path(project))
    if not rd.is_dir():
        return False
    for session_dir in rd.iterdir():
        if not session_dir.is_dir() or session_dir.name.startswith("."):
            continue
        meta_path = session_dir / "metadata.yaml"
        if not meta_path.is_file():
            continue
        try:
            meta = yaml.safe_load(meta_path.read_text(encoding="utf-8")) or {}
        except Exception:
            continue
        if meta.get("arenes"):
            return True
    return False


def project_kind(project: Path) -> str:
    """'single' ou 'multi' — déduit, pas lu, car `create_project.py`
    n'écrit jamais de clé `kind` dans `pipeline_config.yaml` (`{}` pour
    single, seulement `default_arenes_coords` pour multi — ruling R10.1).

    Ordre de préférence :
    1. clé `kind` explicite si un jour présente (pérennité) ;
    2. sinon 'multi' si `default_arenes_coords` est renseigné ;
    3. sinon 'multi' si une session a une liste `arenes` non vide dans sa
       metadata (vérité de terrain après sync, cf. `_session_a_des_arenes`) ;
    4. sinon 'single'.
    """
    cfg = read_pipeline_config(project)
    kind = cfg.get("kind")
    if kind in ("single", "multi"):
        return kind
    if cfg.get("default_arenes_coords"):
        return "multi"
    if _session_a_des_arenes(project):
        return "multi"
    return "single"


def px_per_cm(project: Path) -> float | None:
    value = read_pipeline_config(project).get("px_per_cm")
    return float(value) if value is not None else None


def arena_coords(project: Path) -> dict[str, list[int]]:
    return read_pipeline_config(project).get("default_arenes_coords") or {}


def excel_path(project: Path) -> Path | None:
    """Classeur Excel maître à la racine du projet, ou `None` s'il n'y en a pas.

    Délègue à `sync_from_excel.find_project_excel` (même glob que le
    script : `*_sessions.xlsx` en priorité, sinon un unique `*.xlsx`) pour
    que la localisation du classeur n'ait qu'une seule implémentation,
    partagée entre le CLI et l'app.
    """
    return _find_project_excel(Path(project))


def cropped_dir(project: Path) -> Path:
    """`<projet>/data/cropped/` — délègue à `scripts/paths.py`, source unique."""
    return _paths.cropped_dir(Path(project))


def dlc_output_dir(project: Path) -> Path:
    """`<projet>/data/dlc-output/` — délègue à `scripts/paths.py`, source unique."""
    return _paths.dlc_output_dir(Path(project))


def cropped_videos_exist(project: Path) -> bool:
    """Vrai si au moins une session a déjà des vidéos croppées (étape 4, voie B).

    Sert à déduire le mode d'inférence DLC par défaut (page Pose) : si le
    crop par arène a déjà été fait, la suite logique est `single-animal`
    sur ces vidéos, pas `superanimal` sur la vidéo entière.
    """
    d = cropped_dir(project)
    if not d.is_dir():
        return False
    return any(d.rglob("*.mp4"))
