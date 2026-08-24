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


# ------------------------------------------------------------------- chemins

def jobs_dir(project: Path) -> Path:
    d = Path(project) / ".ethoflow" / "jobs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _job_json(project: Path, job_id: str) -> Path:
    return jobs_dir(project) / f"{job_id}.json"


def _job_log(project: Path, job_id: str) -> Path:
    return jobs_dir(project) / f"{job_id}.log"


def _lock(project: Path) -> Path:
    return jobs_dir(project) / "current.lock"


def _write_job(project: Path, job: Job) -> None:
    _job_json(project, job.job_id).write_text(
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


def _pid_vivant(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except (OSError, ProcessLookupError):
        return False
    return True


def _reconcilier(project: Path, job: Job) -> Job:
    """Un job `running` dont le process a disparu = app tuée en cours de route."""
    if job.state != "running" or _pid_vivant(job.pid):
        return job
    job = replace(job, state="interrupted",
                  ended_at=datetime.now().isoformat(timespec="seconds"))
    _write_job(project, job)
    _lock(project).unlink(missing_ok=True)
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
    job = current(project)
    return bool(job and job.state == "running")


def history(project: Path, limit: int = 20) -> list[Job]:
    """Du plus récent au plus ancien. Les job_id sont horodatés donc triables."""
    jobs = []
    for path in sorted(jobs_dir(project).glob("*.json"), reverse=True):
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
    log_path = _job_log(project, job_id)

    env = dict(os.environ)
    # Sans ça Python tamponne sa sortie quand elle n'est pas un terminal,
    # et les logs arrivent par blocs au lieu d'arriver ligne à ligne.
    env["PYTHONUNBUFFERED"] = "1"

    log_file = open(log_path, "w", encoding="utf-8", buffering=1)
    proc = subprocess.Popen(
        argv,
        stdout=log_file, stderr=subprocess.STDOUT,
        env=env, cwd=str(project), text=True,
        **_popen_kwargs_pour_plateforme(os.name == "nt"),   # groupe de process propre, pour l'annulation
    )

    job = Job(
        job_id=job_id, script=cmd.script, env=cmd.env, label=cmd.label,
        argv=argv, started_at=datetime.now().isoformat(timespec="seconds"),
        pid=proc.pid, state="running",
    )
    _write_job(project, job)
    _lock(project).write_text(job_id, encoding="utf-8")

    def _surveiller() -> None:
        code = proc.wait()
        log_file.close()
        courant = _read_job(project, job_id) or job
        if courant.state == "cancelled":
            etat = "cancelled"
        else:
            etat = "succeeded" if code == 0 else "failed"
        _write_job(project, replace(
            courant, state=etat, returncode=code,
            ended_at=datetime.now().isoformat(timespec="seconds"),
        ))
        _lock(project).unlink(missing_ok=True)

    threading.Thread(target=_surveiller, daemon=True).start()
    return job


def cancel(project: Path, job_id: str) -> None:
    """Tue le groupe de process.

    Le process visible est `conda run`, pas le Python qui travaille : il
    faut descendre au groupe entier, sinon l'enfant survit à son parent.
    """
    job = _read_job(project, job_id)
    if not job or job.state != "running" or not job.pid:
        return
    _write_job(project, replace(job, state="cancelled"))
    _tuer_groupe_de_process(job.pid, os.name == "nt")
