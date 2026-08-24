"""Exécution des scripts en tâche de fond, avec état persisté sur disque.

Un `session_state` Streamlit ne survit ni à la navigation, ni au
rafraîchissement du navigateur, ni à la fermeture de l'onglet. L'état d'un
job vit donc dans `<projet>/.ethoflow/jobs/` :

    <job_id>.json   métadonnées et état
    <job_id>.log    stdout + stderr fusionnés, écrits au fil de l'eau
    current.lock    job_id du job en cours, s'il y en a un

Un seul job à la fois par projet : DLC et VAME veulent tous les deux le
GPU, les laisser tourner ensemble est une source de plantage.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import threading
from dataclasses import asdict, dataclass, replace as _dc_replace
from datetime import datetime
from pathlib import Path

from lib.pipeline import Command, to_argv

replace = _dc_replace


class JobBusy(RuntimeError):
    """Un job tourne déjà pour ce projet."""


@dataclass(frozen=True)
class Job:
    job_id: str
    script: str
    env: str
    label: str
    argv: list[str]
    started_at: str
    ended_at: str | None = None
    returncode: int | None = None
    pid: int | None = None
    state: str = "running"     # running|succeeded|failed|cancelled|interrupted
    cancel_requested: bool = False   # posé par cancel() ; l'état terminal
                                      # est décidé par le watcher, seul à
                                      # savoir quand le process meurt vraiment
    owner_pid: int | None = None     # pid du process Python qui a lancé ce
                                      # job (celui qui héberge son watcher) ;
                                      # sert à distinguer "j'ai un watcher qui
                                      # va écrire l'état terminal" de "orphelin
                                      # d'une exécution précédente de l'app"


# ------------------------------------------------------------------- chemins
#
# `create` par défaut à False partout ici (ruling R10.6c) : ce module sert
# aussi bien des lectures pures (current(), history(), is_running(),
# read_log() — appelées à chaque rendu de page, y compris sur un projet
# qu'on vient de supprimer) que des écritures (start()). Avant ce ruling,
# `jobs_dir()` faisait systématiquement `mkdir(parents=True, exist_ok=True)`
# : consulter le panneau de job d'un projet supprimé recréait
# silencieusement `<projet>/.ethoflow/jobs/` — un squelette de projet
# ressuscité par un simple rafraîchissement de page. Seuls les appelants
# qui vont réellement écrire dedans passent `create=True`.

def jobs_dir(project: Path, *, create: bool = False) -> Path:
    d = Path(project) / ".ethoflow" / "jobs"
    if create:
        d.mkdir(parents=True, exist_ok=True)
    return d


def _job_json(project: Path, job_id: str, *, create: bool = False) -> Path:
    return jobs_dir(project, create=create) / f"{job_id}.json"


def _job_log(project: Path, job_id: str, *, create: bool = False) -> Path:
    return jobs_dir(project, create=create) / f"{job_id}.log"


def _lock(project: Path, *, create: bool = False) -> Path:
    return jobs_dir(project, create=create) / "current.lock"


def _write_job(project: Path, job: Job) -> None:
    _job_json(project, job.job_id, create=True).write_text(
        json.dumps(asdict(job), indent=2), encoding="utf-8")


def _read_job(project: Path, job_id: str) -> Job | None:
    path = _job_json(project, job_id)
    if not path.exists():
        return None
    try:
        return Job(**json.loads(path.read_text(encoding="utf-8")))
    except Exception:
        return None


def _argv(cmd: Command) -> list[str]:
    """Indirection pour que les tests puissent court-circuiter conda."""
    return to_argv(cmd)


def _apres_reclamation_hook() -> None:
    """Point d'ancrage pour les tests : ne fait rien en production.

    Appelé par le watcher juste après `proc.wait()`, avant qu'il écrive
    l'état terminal. Sans lui, la fenêtre entre « le process est réclamé »
    et « l'état terminal est écrit » n'est atteignable en test que par
    hasard (un flake ~1/20 en pratique) — ce hook permet de la bloquer
    délibérément via `monkeypatch` pour la couvrir à coup sûr. Ne change
    aucune décision, ne fait rien tant qu'il n'est pas monkeypatché.
    """


def _pid_vivant(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except (OSError, ProcessLookupError):
        return False
    return True


def _liberer_verrou_si_proprietaire(project: Path, job_id: str) -> None:
    """Supprime le verrou seulement s'il désigne encore CE job.

    Un job qui se termine en retard (watcher lent à observer la vraie mort
    du process, ex : SIGTERM ignoré pendant une fermeture de contexte CUDA)
    ne doit jamais emporter le verrou d'un job plus récent démarré entre-
    temps — sinon ce job plus récent se retrouve considéré comme arrêté
    alors qu'il tourne toujours.
    """
    lock = _lock(project)
    if lock.exists() and lock.read_text(encoding="utf-8").strip() == job_id:
        lock.unlink(missing_ok=True)


def _reconcilier(project: Path, job: Job) -> Job:
    """Rattrape les jobs orphelins d'une exécution précédente de l'app.

    Un job `running` dont le process a disparu a normalement un watcher
    (thread démarré par `start()`) qui va lui-même écrire l'état terminal
    dès que `proc.wait()` renvoie. Si ce job appartient au process Python
    courant (`owner_pid == os.getpid()`), il A ce watcher : il ne faut
    surtout pas lui marcher dessus entre le moment où il a réclamé le pid
    et celui où il a écrit l'état terminal, sous peine d'écraser un
    `cancelled` par un `interrupted` (R4.4). Seuls les jobs sans watcher
    vivant — parce que l'app a été tuée puis relancée — ont besoin d'être
    reconciliés ici.
    """
    if job.state != "running" or _pid_vivant(job.pid):
        return job
    if job.owner_pid == os.getpid():
        return job     # un watcher de CE process va écrire l'état terminal
    # Défense en profondeur : un job orphelin dont l'utilisateur avait
    # demandé l'annulation reste `cancelled`, jamais `interrupted`.
    etat = "cancelled" if job.cancel_requested else "interrupted"
    job = replace(job, state=etat,
                  ended_at=datetime.now().isoformat(timespec="seconds"))
    _write_job(project, job)
    _liberer_verrou_si_proprietaire(project, job.job_id)
    return job


# ------------------------------------------------------------------ lecture

def current(project: Path) -> Job | None:
    lock = _lock(project)
    if lock.exists():
        job = _read_job(project, lock.read_text(encoding="utf-8").strip())
        if job:
            return _reconcilier(project, job)
    hist = history(project, limit=1)
    return hist[0] if hist else None


def is_running(project: Path) -> bool:
    """Autorité : verrou présent ET pid réellement vivant.

    Pas l'état JSON — un job peut y être `cancelled` alors que le process
    résiste encore au signal (fermeture de contexte CUDA, par exemple). Il
    occupe le GPU tant que son pid existe, donc c'est le pid qui décide,
    pas l'étiquette posée par `cancel()`.
    """
    lock = _lock(project)
    if not lock.exists():
        return False
    job = _read_job(project, lock.read_text(encoding="utf-8").strip())
    return bool(job and _pid_vivant(job.pid))


def history(project: Path, limit: int = 20) -> list[Job]:
    """Du plus récent au plus ancien. Les job_id sont horodatés donc triables.

    Lecture pure : ne crée jamais `.ethoflow/jobs/` (ruling R10.6c). Un
    projet qui n'a jamais eu de job — ou qui vient d'être supprimé — n'a
    juste pas de dossier, ce n'est pas une erreur.
    """
    d = jobs_dir(project)
    if not d.is_dir():
        return []
    jobs = []
    for path in sorted(d.glob("*.json"), reverse=True):
        job = _read_job(project, path.stem)
        if job:
            jobs.append(_reconcilier(project, job))
    return jobs[:limit]


def read_log(project: Path, job_id: str, tail: int | None = None) -> str:
    path = _job_log(project, job_id)
    if not path.exists():
        return ""
    texte = path.read_text(encoding="utf-8", errors="replace")
    if tail is None:
        return texte
    return "\n".join(texte.splitlines()[-tail:])


# ------------------------------------------------------ spécifique plateforme
#
# EthoFlow tourne surtout sous Windows (voir README). `start_new_session`
# et `os.killpg` sont POSIX uniquement et lèvent `AttributeError` sous
# Windows. Il faut pourtant tuer tout le groupe de process dans les deux
# cas : le process visible est `conda run`, qui lance à son tour le
# Python qui fait le vrai travail. Ne tuer que le parent laisse l'enfant
# orphelin — sur un job GPU, cet orphelin garde la VRAM occupée.
#
# Les deux branches sont isolées dans de petites fonctions dédiées pour
# qu'un relecteur puisse les comparer côte à côte, plutôt que noyées dans
# des `if os.name` au milieu de `start()` et `cancel()`.

def _popen_kwargs_pour_plateforme(is_windows: bool) -> dict:
    """kwargs à passer à `subprocess.Popen` pour isoler le groupe de process.

    `subprocess.CREATE_NEW_PROCESS_GROUP` n'existe comme attribut du module
    que sous Windows. `getattr` avec sa valeur documentée (0x00000200)
    permet de construire ce dict de test sur macOS/Linux sans jamais
    l'utiliser réellement (la branche `is_windows=True` n'est prise en
    production que quand `os.name == "nt"`).
    """
    if is_windows:
        flag = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x00000200)
        return {"creationflags": flag}
    return {"start_new_session": True}


def _tuer_groupe_de_process(pid: int, is_windows: bool) -> None:
    """Tue le process `pid` et tout son groupe (ex: le Python lancé par `conda run`)."""
    if is_windows:
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(pid)])
        return
    try:
        os.killpg(os.getpgid(pid), signal.SIGTERM)
    except (OSError, ProcessLookupError):
        pass


# ---------------------------------------------------------------- exécution

def start(project: Path, cmd: Command) -> Job:
    project = Path(project)
    if is_running(project):
        en_cours = current(project)
        raise JobBusy(
            f"« {en_cours.label} » tourne déjà (démarré à {en_cours.started_at}). "
            "Attends la fin ou annule-le."
        )

    # Microsecondes incluses : deux jobs `__test__` (echo quasi instantané)
    # peuvent démarrer dans la même seconde et se marcheraient dessus sinon
    # (même job_id -> même fichier JSON écrasé). Le tri lexicographique
    # utilisé par `history()` reste valable, `%f` est du zero-padding.
    horodatage = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    job_id = f"{horodatage}-{Path(cmd.script).stem}"
    argv = _argv(cmd)
    # Seul point d'écriture qui doit faire exister `.ethoflow/jobs/` s'il
    # n'existe pas encore (premier job du projet, ou projet tout juste
    # recréé) — voir le commentaire sur `jobs_dir` (ruling R10.6c).
    log_path = _job_log(project, job_id, create=True)

    env = dict(os.environ)
    # Sans ça Python tamponne sa sortie quand elle n'est pas un terminal,
    # et les logs arrivent par blocs au lieu d'arriver ligne à ligne.
    env["PYTHONUNBUFFERED"] = "1"

    log_file = open(log_path, "w", encoding="utf-8", buffering=1)
    try:
        proc = subprocess.Popen(
            argv,
            stdout=log_file, stderr=subprocess.STDOUT,
            env=env, cwd=str(project), text=True,
            **_popen_kwargs_pour_plateforme(os.name == "nt"),   # groupe de process propre, pour l'annulation
        )
    except Exception:
        # Popen a échoué (ex : binaire introuvable) : le descripteur ne
        # doit pas fuir, rien d'autre n'a encore été écrit sur disque.
        log_file.close()
        raise

    job = Job(
        job_id=job_id, script=cmd.script, env=cmd.env, label=cmd.label,
        argv=argv, started_at=datetime.now().isoformat(timespec="seconds"),
        pid=proc.pid, state="running", owner_pid=os.getpid(),
    )
    _write_job(project, job)
    _lock(project, create=True).write_text(job_id, encoding="utf-8")

    def _surveiller() -> None:
        # Seul ce thread sait quand le process est VRAIMENT mort
        # (`proc.wait()` bloque jusque-là) : c'est donc lui, et lui seul,
        # qui décide de l'état terminal. `cancel()` ne fait que poser
        # l'intention (`cancel_requested`) et envoyer le signal — sinon un
        # process lent à mourir serait déclaré arrêté alors qu'il tourne
        # encore et retient le GPU.
        code = proc.wait()
        log_file.close()
        _apres_reclamation_hook()
        courant = _read_job(project, job_id) or job
        etat = "cancelled" if courant.cancel_requested else (
            "succeeded" if code == 0 else "failed")
        _write_job(project, replace(
            courant, state=etat, returncode=code,
            ended_at=datetime.now().isoformat(timespec="seconds"),
        ))
        _liberer_verrou_si_proprietaire(project, job_id)

    threading.Thread(target=_surveiller, daemon=True).start()
    return job


def cancel(project: Path, job_id: str) -> None:
    """Demande l'annulation et tue le groupe de process.

    N'écrit PAS d'état terminal : le process peut mettre du temps à
    mourir (fermeture de contexte CUDA, SIGTERM ignoré...), et tant qu'il
    est vivant il occupe encore le GPU. C'est le watcher de `start()`, qui
    observe la vraie fin du process via `proc.wait()`, qui décide de
    l'état terminal — `cancelled` ici puisque `cancel_requested` est posé.

    Le process visible est `conda run`, pas le Python qui travaille : il
    faut descendre au groupe entier, sinon l'enfant survit à son parent.
    """
    job = _read_job(project, job_id)
    if not job or job.state != "running" or not job.pid:
        return
    _write_job(project, replace(job, cancel_requested=True))
    _tuer_groupe_de_process(job.pid, os.name == "nt")
