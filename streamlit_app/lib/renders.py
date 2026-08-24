"""Découverte des rendus de la page Visualisations (Task 22) et recouvrement
de leurs paramètres depuis `lib/runner.history`.

`motif_gif.py`, `behavior_structure_gif.py` et `community_dendrogram.py`
écrivent leurs fichiers directement sous `<vame>/analysis/...` sans jamais
persister les paramètres à côté (pas de `.json` compagnon) — un simple glob
dit QUELS fichiers existent, mais pas avec quels réglages (`--session`,
`--start`, `--pool-all-sessions`, …) chacun a été produit. Cette info existe
pourtant déjà : `lib.runner` persiste l'`argv` de chaque job dans
`<projet>/.ethoflow/jobs/<job_id>.json`, qui survit lui aussi à une
navigation ou un redémarrage de l'app.

`match_render` recolle les deux en confrontant la date de modification du
fichier à la fenêtre `[started_at, ended_at]` du job qui a tourné à ce
moment-là (un seul job à la fois par projet — `lib.runner.JobBusy` — donc
cette fenêtre est sans ambiguïté). Reconstruire le nom de fichier à partir
de l'`argv` a été écarté délibérément : `behavior_structure_gif.py` décide
du suffixe `_sidebyside` non pas d'après le flag `--with-video` mais
d'après le succès *réel* de l'ouverture de la vidéo source (voir
`video_cap` dans le script) — un `--with-video` demandé mais dont la vidéo
source est introuvable produit un fichier SANS ce suffixe. Deviner le nom
depuis les flags aurait donc raté certains rendus.

Même refus de deviner que `lib.pipeline.parse_prepare_vame_input_args`
(Task 21, ruling R21.1) : un flag inconnu dans l'`argv` d'un job invalide
tout le parse plutôt que d'être ignoré ou de faire passer sa valeur pour
autre chose.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

from lib import runner
from lib.runner import Job
from lib.vame import vame_project

_VIDEO_EXTS = {".mp4", ".gif"}


# ============================================================
# Emplacements de sortie — lus dans les scripts, pas devinés
# ============================================================
#
# motif_gif.py       : out_dir = vame_proj / "analysis" / "motif_gifs"
# behavior_structure_gif.py : out_dir = vame_proj / "analysis" / "behavior_structure"
# community_dendrogram.py   : out_dir = vame_proj / "analysis" (fichier
#                              "community_dendrogram<suffix>.png")

def motif_gif_dir(project: Path) -> Path:
    return vame_project(project) / "analysis" / "motif_gifs"


def manifold_dir(project: Path) -> Path:
    return vame_project(project) / "analysis" / "behavior_structure"


def _lister(dossier: Path, *, extensions: set[str] | None = None,
           motif: str | None = None) -> list[Path]:
    """Fichiers du dossier, du plus récent au plus ancien (mtime)."""
    if not dossier.is_dir():
        return []
    if motif is not None:
        candidats = list(dossier.glob(motif))
    else:
        candidats = [p for p in dossier.iterdir()
                    if p.is_file() and (extensions is None or p.suffix.lower() in extensions)]
    return sorted(candidats, key=lambda p: p.stat().st_mtime, reverse=True)


def list_motif_gifs(project: Path) -> list[Path]:
    return _lister(motif_gif_dir(project), extensions=_VIDEO_EXTS)


def list_manifolds(project: Path) -> list[Path]:
    return _lister(manifold_dir(project), extensions=_VIDEO_EXTS)


def list_dendrograms(project: Path) -> list[Path]:
    return _lister(vame_project(project) / "analysis", motif="community_dendrogram*.png")


# ============================================================
# Recouvrement des paramètres depuis l'argv d'un job
# ============================================================

def _parse_flags(argv: list[str], script: str, *,
                 value_flags: set[str], bool_flags: set[str]) -> dict[str, str] | None:
    """Coupe l'`argv` d'un job en flags bruts (`--xxx` -> valeur ou True).

    Retourne `None` si le script ne correspond pas, ou si l'`argv`
    contient un flag hors de `value_flags`/`bool_flags` (format futur du
    script, ruling R21.1 appliqué ici aussi) — refuser plutôt que de faire
    passer une valeur pour autre chose.
    """
    args = list(argv)
    try:
        i_script = next(i for i, a in enumerate(args) if a.endswith(script))
    except StopIteration:
        return None

    rest = args[i_script + 1:]
    kwargs: dict[str, str] = {}
    i = 0
    while i < len(rest):
        tok = rest[i]
        if tok == "--project-dir":
            i += 2
            continue
        if tok == "--no-prompt":
            i += 1
            continue
        if tok in bool_flags:
            kwargs[tok] = "1"
            i += 1
            continue
        if tok in value_flags:
            if i + 1 >= len(rest):
                return None
            kwargs[tok] = rest[i + 1]
            i += 2
            continue
        return None
    return kwargs


def parse_motif_gif_args(argv: list[str]) -> dict | None:
    """Reconstruit les kwargs de `lib.pipeline.motif_gif` depuis un `argv`."""
    raw = _parse_flags(
        argv, "motif_gif.py",
        value_flags={"--session", "--algo", "--labels", "--start",
                     "--duration", "--output-format"},
        bool_flags=set(),
    )
    if raw is None or "--session" not in raw:
        return None
    return {
        "session": raw["--session"],
        "algo": raw.get("--algo", "hmm"),
        "start": float(raw.get("--start", 0.0)),
        "duration": float(raw["--duration"]) if "--duration" in raw else None,
        "output_format": raw.get("--output-format", "mp4"),
        "labels": raw.get("--labels"),
    }


def parse_behavior_structure_gif_args(argv: list[str]) -> dict | None:
    """Reconstruit les kwargs de `lib.pipeline.behavior_structure_gif`."""
    raw = _parse_flags(
        argv, "behavior_structure_gif.py",
        value_flags={"--session", "--algo", "--projection", "--labels",
                     "--start", "--duration", "--output-format"},
        bool_flags={"--with-video", "--pool-all-sessions"},
    )
    if raw is None or "--session" not in raw:
        return None
    return {
        "session": raw["--session"],
        "algo": raw.get("--algo", "hmm"),
        "projection": raw.get("--projection", "umap"),
        "start": float(raw.get("--start", 0.0)),
        "duration": float(raw["--duration"]) if "--duration" in raw else None,
        "output_format": raw.get("--output-format", "gif"),
        "with_video": "--with-video" in raw,
        "pool_all_sessions": "--pool-all-sessions" in raw,
        "labels": raw.get("--labels"),
    }


def parse_community_dendrogram_args(argv: list[str]) -> dict | None:
    """Reconstruit les kwargs de `lib.pipeline.community_dendrogram`."""
    raw = _parse_flags(
        argv, "community_dendrogram.py",
        value_flags={"--algo", "--labels", "--group", "--linkage"},
        bool_flags=set(),
    )
    if raw is None:
        return None
    return {
        "algo": raw.get("--algo", "hmm"),
        "group": raw.get("--group"),
        "linkage": raw.get("--linkage", "ward"),
        "labels": raw.get("--labels"),
    }


# ============================================================
# Rapprochement fichier <-> job par fenêtre temporelle
# ============================================================

@dataclass(frozen=True)
class Render:
    path: Path
    job: Job | None
    params: dict | None


def _fenetre(job: Job) -> tuple[float, float] | None:
    try:
        debut = datetime.fromisoformat(job.started_at).timestamp()
    except (ValueError, TypeError):
        return None
    fin = debut
    if job.ended_at:
        try:
            fin = datetime.fromisoformat(job.ended_at).timestamp()
        except ValueError:
            fin = debut
    return debut, fin


def match_render(path: Path, jobs: list[Job],
                 parser: Callable[[list[str]], dict | None]) -> Render:
    """Associe `path` au job (parmi `jobs`, déjà filtrés par script+succès)
    dont la fenêtre `[started_at, ended_at]` contient sa date de
    modification. Marge de 2 s : `started_at`/`ended_at` sont horodatés à
    la seconde (`timespec="seconds"`), le fichier peut être écrit
    quelques centaines de ms avant/après cette frontière arrondie.
    """
    mtime = path.stat().st_mtime
    for job in jobs:
        fenetre = _fenetre(job)
        if fenetre is None:
            continue
        debut, fin = fenetre
        if debut - 2 <= mtime <= fin + 2:
            params = parser(job.argv)
            if params is not None:
                return Render(path=path, job=job, params=params)
    return Render(path=path, job=None, params=None)


def _renders(project: Path, files: list[Path], script: str,
            parser: Callable[[list[str]], dict | None]) -> list[Render]:
    jobs = [j for j in runner.history(project, limit=200)
            if j.script == script and j.state == "succeeded"]
    return [match_render(p, jobs, parser) for p in files]


def motif_gif_renders(project: Path) -> list[Render]:
    return _renders(project, list_motif_gifs(project), "motif_gif.py",
                    parse_motif_gif_args)


def manifold_renders(project: Path) -> list[Render]:
    return _renders(project, list_manifolds(project), "behavior_structure_gif.py",
                    parse_behavior_structure_gif_args)


def dendrogram_renders(project: Path) -> list[Render]:
    return _renders(project, list_dendrograms(project), "community_dendrogram.py",
                    parse_community_dendrogram_args)
