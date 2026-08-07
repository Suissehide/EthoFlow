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


def list_session_videos(project: Path) -> list[tuple[str, Path]]:
    """Liste les (session_id, vidéo source) d'un projet.

    Ne garde que les vidéos qui existent réellement sur le disque — une
    session dont le fichier a été déplacé ne pollue pas les menus.
    """
    import yaml

    out: list[tuple[str, Path]] = []
    rd = project / "data" / "raw"
    if not rd.exists():
        return out
    for session_dir in sorted(rd.iterdir()):
        meta_path = session_dir / "metadata.yaml"
        if not meta_path.exists():
            continue
        try:
            with open(meta_path) as f:
                meta = yaml.safe_load(f) or {}
        except Exception:
            continue
        src = meta.get("source_video")
        if not src:
            continue
        p = Path(src)
        if p.exists():
            out.append((meta.get("session_id") or session_dir.name, p))
    return out


def prompt_session_video(project: Path,
                          session: str | None = None,
                          video: Path | None = None,
                          allow_image: bool = False,
                          no_prompt: bool = False,
                          title: str = "Sur quelle vidéo travailler ?",
                          max_shown: int = 15,
                          ) -> tuple[Path | None, Path | None]:
    """Choisit une vidéo (ou une image) pour un script du pipeline.

    Ordre : `video` explicite > `session` explicite > menu des sessions
    du projet > saisie libre.

    Args:
        allow_image: ajoute une entrée « photo » au menu (calibration)
        title: en-tête du menu

    Returns:
        (video, image) — l'un des deux est None.
    """
    if video is not None:
        return Path(video), None

    if session:
        matches = [(sid, v) for sid, v in list_session_videos(project)
                   if sid == session or sid.endswith(f"-{session}")]
        if not matches:
            print(f"❌ Session '{session}' introuvable dans {project}, "
                  f"ou sa vidéo n'existe plus.", file=sys.stderr)
            sys.exit(1)
        print(f"ℹ  Vidéo de la session {matches[0][0]} : {matches[0][1].name}")
        return matches[0][1], None

    if no_prompt:
        print("❌ --session ou --video requis en mode --no-prompt.",
              file=sys.stderr)
        sys.exit(1)

    sessions = list_session_videos(project)
    if not sessions:
        print(f"Aucune session avec vidéo dans {project}.")
        if allow_image:
            print("Source :")
            print("  1. Une image")
            print("  2. Une vidéo")
            if prompt("Source", default="2", choices=["1", "2"]) == "1":
                return None, prompt_existing_path("Chemin de l'image",
                                                   must_exist=True)
        return prompt_existing_path("Chemin de la vidéo",
                                     must_exist=True), None

    print(title)
    shown = sessions[:max_shown]
    for i, (sid, v) in enumerate(shown, start=1):
        print(f"  {i}. {sid}  ({v.name})")
    if len(sessions) > len(shown):
        print(f"     … et {len(sessions) - len(shown)} autre(s)")
    n = len(shown)
    if allow_image:
        print(f"  {n + 1}. Une photo (image)")
        print(f"  {n + 2}. Une autre vidéo (chemin libre)")
    else:
        print(f"  {n + 1}. Une autre vidéo (chemin libre)")

    while True:
        choice = prompt("Choix", default="1")
        if choice.isdigit():
            idx = int(choice)
            if 1 <= idx <= n:
                return shown[idx - 1][1], None
            if allow_image and idx == n + 1:
                return None, prompt_existing_path("Chemin de l'image",
                                                   must_exist=True)
            if idx == (n + 2 if allow_image else n + 1):
                return prompt_existing_path("Chemin de la vidéo",
                                             must_exist=True), None
        match = [v for sid, v in sessions if sid == choice]
        if match:
            return match[0], None
        print("  ⚠ choix invalide")


def prompt_session(project: Path, session: str | None = None,
                    no_prompt: bool = False,
                    title: str = "Quelle session ?") -> str:
    """Choisit un session_id parmi ceux du projet.

    Pour les scripts qui travaillent sur une session sans avoir besoin
    du chemin vidéo (visualisations VAME, analyses par session).
    """
    import yaml

    if session:
        return session
    if no_prompt:
        print("❌ --session requis en mode --no-prompt.", file=sys.stderr)
        sys.exit(1)

    rd = project / "data" / "raw"
    sessions: list[str] = []
    if rd.exists():
        for d in sorted(rd.iterdir()):
            meta = d / "metadata.yaml"
            if not meta.exists():
                continue
            try:
                with open(meta) as f:
                    m = yaml.safe_load(f) or {}
            except Exception:
                m = {}
            sessions.append(m.get("session_id") or d.name)

    if not sessions:
        return prompt("Session ID")

    print(title)
    shown = sessions[:20]
    for i, sid in enumerate(shown, start=1):
        print(f"  {i}. {sid}")
    if len(sessions) > len(shown):
        print(f"     … et {len(sessions) - len(shown)} autre(s)")
    while True:
        choice = prompt("Choix", default="1")
        if choice.isdigit() and 1 <= int(choice) <= len(shown):
            return shown[int(choice) - 1]
        if choice in sessions:
            return choice
        print("  ⚠ choix invalide")


def confirm(question: str, default: str = "y",
             no_prompt: bool = False) -> bool:
    """Demande une confirmation y/n. Renvoie True si l'utilisateur accepte.

    En mode --no-prompt, retourne True (on suppose que le user sait ce
    qu'il fait quand il automatise).
    """
    if no_prompt:
        return True
    return prompt(question, default=default, choices=["y", "n"]) == "y"
