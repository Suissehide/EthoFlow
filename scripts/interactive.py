"""Helpers de saisie interactive partagés par les scripts CLI EthoFlow.

Principe : **aucun argument n'est obligatoire**. Si l'utilisateur ne
passe rien, le script demande à l'invite avec un défaut sensé. Les
utilisateurs avancés (et les scripts / la future app Streamlit)
passent tout en arguments et rien n'est demandé.

Le flag `--no-prompt` (ajouté via `add_no_prompt_arg`) fait échouer au
lieu de demander — mode non-interactif pour CI et automatisation.

Usage type dans un script :

    import argparse
    from interactive import (
        add_no_prompt_arg, prompt, prompt_existing_path, resolve_or_prompt,
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--project-dir", type=Path, default=None)
    parser.add_argument("--excel", type=Path, default=None)
    add_no_prompt_arg(parser)
    args = parser.parse_args()

    project = resolve_or_prompt_project(args)
    excel = args.excel or prompt_existing_path(
        "Chemin de l'Excel", must_exist=True, no_prompt=args.no_prompt,
    )
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


# Racine par défaut où chercher les projets EthoFlow (cohérent avec
# create_project.py). Sert de base aux prompts de sélection de projet.
DEFAULT_PROJECTS_ROOT = Path(r"D:\EthoFlow\projects")

# Racine par défaut des modèles DLC entraînés (cohérent avec le wizard
# 00_init_training_config.py). Un modèle = un dossier contenant un
# config.yaml.
DEFAULT_MODELS_ROOT = Path(r"D:\EthoFlow\models")


def add_no_prompt_arg(parser: argparse.ArgumentParser) -> None:
    """Ajoute --no-prompt (échoue au lieu de demander)."""
    parser.add_argument(
        "--no-prompt", action="store_true",
        help="Échoue si un argument requis manque, au lieu de le demander "
             "à l'invite (mode non-interactif, pour scripts et CI).",
    )


def _fail_or_prompt(no_prompt: bool, what: str) -> None:
    """Sort en erreur si --no-prompt est actif."""
    if no_prompt:
        print(f"❌ {what} requis en mode --no-prompt.", file=sys.stderr)
        sys.exit(1)


def prompt(question: str, default: str | None = None,
            choices: list[str] | None = None,
            allow_empty: bool = False,
            no_prompt: bool = False) -> str:
    """Prompt texte avec défaut + validation optionnelle par liste."""
    _fail_or_prompt(no_prompt, question)
    while True:
        if choices:
            suffix = f" [{'/'.join(choices)}]"
            if default:
                suffix += f" (défaut {default})"
        else:
            suffix = f" [{default}]" if default else ""
        raw = input(f"{question}{suffix} : ").strip().strip('"').strip("'")
        val = raw or (default or "")
        if not val:
            if allow_empty:
                return ""
            print("  ⚠ valeur requise")
            continue
        if choices and val not in choices:
            print(f"  ⚠ choix invalide, attendus : {choices}")
            continue
        return val


def prompt_existing_path(question: str, default: str | None = None,
                          must_exist: bool = True,
                          allow_empty: bool = False,
                          no_prompt: bool = False) -> Path | None:
    """Prompt un chemin, avec re-demande si le fichier n'existe pas.

    Les guillemets sont strippés automatiquement (Windows « Copier en tant
    que chemin » les ajoute).
    """
    _fail_or_prompt(no_prompt, question)
    while True:
        raw = prompt(question, default=default, allow_empty=allow_empty)
        if not raw and allow_empty:
            return None
        p = Path(raw)
        if not must_exist or p.exists():
            return p
        print(f"  ⚠ chemin introuvable : {p}")


def list_projects(root: Path = DEFAULT_PROJECTS_ROOT) -> list[Path]:
    """Liste les projets EthoFlow existants sous `root`.

    Un projet est reconnu à la présence de `configs/pipeline_config.yaml`.
    """
    if not root.exists():
        return []
    out = []
    for d in sorted(root.iterdir()):
        if d.is_dir() and (d / "configs" / "pipeline_config.yaml").exists():
            out.append(d)
    return out


def prompt_project(root: Path = DEFAULT_PROJECTS_ROOT,
                    no_prompt: bool = False) -> Path:
    """Demande quel projet EthoFlow utiliser.

    Si des projets sont trouvés sous `root`, propose un menu numéroté.
    Sinon demande un chemin libre.
    """
    _fail_or_prompt(no_prompt, "--project-dir")
    projects = list_projects(root)
    if projects:
        print(f"Projets trouvés dans {root} :")
        for i, p in enumerate(projects, start=1):
            print(f"  {i}. {p.name}")
        print(f"  {len(projects) + 1}. (autre chemin)")
        while True:
            choice = prompt("Projet", default="1")
            if choice.isdigit():
                idx = int(choice)
                if 1 <= idx <= len(projects):
                    return projects[idx - 1]
                if idx == len(projects) + 1:
                    break
            # L'utilisateur a peut-être tapé un nom de projet directement
            match = [p for p in projects if p.name == choice]
            if match:
                return match[0]
            print("  ⚠ choix invalide")
    p = prompt_existing_path("Chemin du projet EthoFlow", must_exist=True)
    return p.resolve()


def resolve_or_prompt_project(args: argparse.Namespace,
                                root: Path = DEFAULT_PROJECTS_ROOT) -> Path:
    """Retourne le projet depuis args.project_dir, ou le demande.

    Remplace `resolve_project(args)` dans les scripts qui veulent
    l'expérience interactive. Le fallback legacy sur REPO_ROOT est
    supprimé : on demande explicitement plutôt que de deviner.
    """
    pd = getattr(args, "project_dir", None)
    if pd is not None:
        return Path(pd).resolve()
    return prompt_project(root, no_prompt=getattr(args, "no_prompt", False))


def confirm(question: str, default: str = "y",
             no_prompt: bool = False) -> bool:
    """Demande une confirmation y/n. Renvoie True si l'utilisateur accepte.

    En mode --no-prompt, retourne True (on suppose que le user sait ce
    qu'il fait quand il automatise).
    """
    if no_prompt:
        return True
    return prompt(question, default=default, choices=["y", "n"]) == "y"
