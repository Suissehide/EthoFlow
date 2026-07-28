"""Chemins projet-aware partagés par les scripts CLI et streamlit_app.

Un *projet* EthoFlow est un dossier auto-suffisant qui contient ses propres
données et sa propre config DLC/VAME :

    <project>/
    ├── data/
    │   ├── raw/<session>/metadata.yaml
    │   ├── cropped/<session>/
    │   ├── dlc-output/<session>/
    │   ├── vame/<vame_project>/
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

Pour les sous-scripts (e.g. scripts/dlc_model-training/*), insère le dossier
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
    """Ajoute `--project-dir` (+ `--no-prompt`) à un parser argparse.

    `--project-dir` n'est jamais marqué `required` au sens argparse :
    s'il manque, `resolve_project()` demande le projet à l'invite (menu
    des projets trouvés sous D:/EthoFlow/projects, ou saisie libre).
    Le paramètre `required` est conservé pour compat d'appel mais n'a
    plus d'effet — passe `--no-prompt` pour un échec franc en CI.
    """
    parser.add_argument(
        "--project-dir",
        type=Path,
        default=None,
        help=(
            "Racine du projet EthoFlow (e.g. D:\\EthoFlow\\projects\\foo). "
            "Demandé à l'invite si absent."
        ),
    )
    # Évite le doublon si le script a déjà ajouté --no-prompt lui-même
    if not any(a.dest == "no_prompt" for a in parser._actions):
        parser.add_argument(
            "--no-prompt",
            action="store_true",
            help=(
                "Échoue si un argument requis manque, au lieu de le "
                "demander à l'invite (mode non-interactif, CI)."
            ),
        )


# Cache du projet résolu interactivement : certains scripts appellent
# resolve_project() plusieurs fois (run_vame par sous-commande), on ne
# veut poser la question qu'une seule fois par process.
_RESOLVED_PROJECT: Path | None = None


def resolve_project(args: argparse.Namespace | Path | None) -> Path:
    """Retourne la racine du projet à utiliser.

    Accepte le Namespace argparse (lit `args.project_dir`), ou
    directement un Path / None pour les appels programmatiques.

    Si `project_dir` est absent, demande le projet à l'invite via
    `interactive.prompt_project()` — menu numéroté des projets trouvés
    sous D:/EthoFlow/projects, ou saisie d'un chemin libre. La réponse
    est mise en cache pour le reste du process.

    En mode `--no-prompt`, échoue au lieu de demander.
    """
    global _RESOLVED_PROJECT

    if isinstance(args, Path):
        return args.resolve()
    if args is not None:
        project = getattr(args, "project_dir", None)
        if project is not None:
            return Path(project).resolve()

    if _RESOLVED_PROJECT is not None:
        return _RESOLVED_PROJECT

    no_prompt = bool(getattr(args, "no_prompt", False)) if args else False
    # Import tardif : interactive importe paths pour DEFAULT_PROJECTS_ROOT
    from interactive import prompt_project  # noqa: WPS433
    _RESOLVED_PROJECT = prompt_project(no_prompt=no_prompt)
    return _RESOLVED_PROJECT


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


def vame_dir(project: Path) -> Path:
    """Racine VAME du projet : <project>/data/vame/.

    C'est l'unique dossier VAME du projet : on y crée les projets VAME
    (`<project>/data/vame/<vame_project_name>/`) via `run_vame setup`.

    Les h5 nettoyés *avant* setup (ce qui était l'ancien `vame-input/`)
    vivent maintenant à côté de leur output DLC d'origine :
    `<project>/data/dlc-output/<session>/<session>_clean.h5`. Logique
    parce que le clean est juste un post-traitement de l'output DLC,
    pas un artefact VAME en soi.
    """
    return project / "data" / "vame"


def cleaned_h5_path(project: Path, session_id: str) -> Path:
    """Path canonique du h5 nettoyé pour VAME, scoped au projet.

    Convention : <project>/data/dlc-output/<session>/<session>_clean.h5.
    """
    return project / "data" / "dlc-output" / session_id / f"{session_id}_clean.h5"


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


