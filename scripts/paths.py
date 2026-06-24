"""Chemins projet-aware partagés par les scripts CLI et streamlit_app.

Un *projet* EthoFlow est un dossier auto-suffisant qui contient ses propres
données et sa propre config DLC/VAME :

    <project>/
    ├── data/
    │   ├── raw/<session>/metadata.yaml
    │   ├── cropped/<session>/
    │   ├── dlc-output/<session>/
    │   ├── vame-input/<session>/
    │   ├── vame-output/<vame_project>/
    │   └── results/
    └── configs/
        └── pipeline_config.yaml

Avant la migration, tout vivait à plat dans `<repo>/data/`. Pour conserver
la rétrocompatibilité (et ne pas casser les commandes existantes qui
opèrent sur l'arbo legacy), `resolve_project(args)` retombe sur la racine
du repo si `--project-dir` n'est pas fourni.

Usage type dans un script CLI :

    import argparse
    from paths import add_project_dir_arg, resolve_project, raw_dir, dlc_output_dir

    parser = argparse.ArgumentParser()
    add_project_dir_arg(parser)
    # ... autres args ...
    args = parser.parse_args()

    project = resolve_project(args)
    sessions = sorted((raw_dir(project)).iterdir())
    out_dir = dlc_output_dir(project) / session_id

Pour les sous-scripts (e.g. scripts/dlc_bottomview/*), insère le dossier
parent dans sys.path avant l'import :

    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from paths import ...
"""
from __future__ import annotations

import argparse
from pathlib import Path


# Racine du repo : <repo>/scripts/paths.py -> <repo>
REPO_ROOT = Path(__file__).resolve().parent.parent


# ----------------------------------------------------------------------
# Helpers CLI : argparse
# ----------------------------------------------------------------------

def add_project_dir_arg(
    parser: argparse.ArgumentParser,
    *,
    required: bool = False,
) -> None:
    """Ajoute `--project-dir` à un parser argparse.

    Si `required=False` (défaut), l'absence de --project-dir fait retomber
    sur l'arbo legacy `<repo>/data/`. Si `required=True`, le script refuse
    de tourner sans projet explicite — utile pour les nouvelles features
    qui ne devraient jamais s'appliquer à l'arbo legacy (e.g.
    sync_from_excel_bottomview).
    """
    parser.add_argument(
        "--project-dir",
        type=Path,
        default=None,
        required=required,
        help=(
            "Racine du projet EthoFlow (e.g. D:\\ethoflow\\projects\\foo). "
            "Si absent, retombe sur la racine legacy <repo>/."
        ),
    )


def resolve_project(args: argparse.Namespace | Path | None) -> Path:
    """Retourne la racine du projet à utiliser.

    Accepte soit le Namespace argparse complet (et lit args.project_dir),
    soit directement un Path ou None pour les appels programmatiques.
    Fallback sur REPO_ROOT si rien n'est fourni — rétrocompatibilité.
    """
    if args is None:
        return REPO_ROOT
    if isinstance(args, Path):
        return args.resolve()
    project = getattr(args, "project_dir", None)
    if project is None:
        return REPO_ROOT
    return Path(project).resolve()


# ----------------------------------------------------------------------
# Chemins dérivés du projet
# ----------------------------------------------------------------------

def data_dir(project: Path) -> Path:
    return project / "data"


def raw_dir(project: Path) -> Path:
    return project / "data" / "raw"


def cropped_dir(project: Path) -> Path:
    return project / "data" / "cropped"


def dlc_output_dir(project: Path) -> Path:
    return project / "data" / "dlc-output"


def vame_input_dir(project: Path) -> Path:
    return project / "data" / "vame-input"


def vame_output_dir(project: Path) -> Path:
    return project / "data" / "vame-output"


def results_dir(project: Path) -> Path:
    return project / "data" / "results"


def configs_dir(project: Path) -> Path:
    return project / "configs"


def pipeline_config_path(project: Path) -> Path:
    """Path attendu pour la config pipeline_config.yaml.

    Convention : un par projet. Évite de mélanger les DLC/VAME configs
    entre projets différents (un projet topview et un projet bottomview
    auraient des dlc_project_config distincts).
    """
    return project / "configs" / "pipeline_config.yaml"


def vame_config_pointer(project: Path) -> Path:
    """Pointer vers le config.yaml du projet VAME courant pour CE projet
    EthoFlow.

    Contenu : un chemin absolu vers le `config.yaml` d'un projet VAME (qui
    vit dans `vame_output_dir(project)/<name>/`). Permet de basculer entre
    plusieurs projets VAME (ex: comparer deux entraînements) sans toucher
    aux scripts.

    Scope au projet (et non global au repo) pour éviter qu'un projet
    topview et un projet bottomview se marchent dessus.
    """
    return project / ".vame_config_path"
