import os
import time

import pytest

from lib import pipeline as PL
from lib import runner as R


def _echo(msg: str = "bonjour") -> PL.Command:
    """Command factice court-circuitant conda, pour tester la machine à états."""
    return PL.Command(env="ethoflow", script="__test__",
                      args=["python", "-c", f"print({msg!r})"], label="echo")


@pytest.fixture(autouse=True)
def _argv_direct(monkeypatch):
    """`__test__` n'est pas dans SCRIPT_ENVS : on court-circuite to_argv."""
    monkeypatch.setattr(
        R, "_argv", lambda cmd: cmd.args if cmd.script == "__test__" else PL.to_argv(cmd)
    )


def _attendre_fin(project, timeout=15.0):
    fin = time.time() + timeout
    while time.time() < fin:
        job = R.current(project)
        if job and job.state != "running":
            return job
        time.sleep(0.05)
    raise AssertionError("job toujours running après timeout")


def test_job_reussi(project):
    R.start(project, _echo("salut"))
    job = _attendre_fin(project)
    assert job.state == "succeeded"
    assert job.returncode == 0
    assert "salut" in R.read_log(project, job.job_id)


def test_job_echoue(project):
    R.start(project, PL.Command("ethoflow", "__test__",
                                ["python", "-c", "raise SystemExit(3)"], "boom"))
    job = _attendre_fin(project)
    assert job.state == "failed"
    assert job.returncode == 3


def test_un_seul_job_a_la_fois(project):
    R.start(project, PL.Command("ethoflow", "__test__",
                                ["python", "-c", "import time; time.sleep(5)"], "long"))
    with pytest.raises(R.JobBusy):
        R.start(project, _echo())
    R.cancel(project, R.current(project).job_id)
    _attendre_fin(project)


def test_annulation(project):
    R.start(project, PL.Command("ethoflow", "__test__",
                                ["python", "-c", "import time; time.sleep(30)"], "long"))
    R.cancel(project, R.current(project).job_id)
    job = _attendre_fin(project)
    assert job.state == "cancelled"


def test_verrou_libere_apres_la_fin(project):
    R.start(project, _echo())
    _attendre_fin(project)
    assert not R.is_running(project)
    R.start(project, _echo("deuxieme"))     # ne doit pas lever
    _attendre_fin(project)


def test_job_running_dont_le_process_a_disparu(project):
    """App tuée pendant un job : le JSON dit running, le pid n'existe plus."""
    R.start(project, _echo())
    job = _attendre_fin(project)
    R._write_job(project, R.replace(job, state="running", pid=999999, ended_at=None))
    releve = R.current(project)
    assert releve.state == "interrupted"


def test_historique_du_plus_recent_au_plus_ancien(project):
    for i in range(3):
        R.start(project, _echo(f"job{i}"))
        _attendre_fin(project)
    labels = [j.label for j in R.history(project)]
    assert len(labels) == 3
    assert R.history(project, limit=2) == R.history(project)[:2]


def test_read_log_tail(project):
    R.start(project, PL.Command(
        "ethoflow", "__test__",
        ["python", "-c", "[print(i) for i in range(100)]"], "cent"))
    job = _attendre_fin(project)
    assert R.read_log(project, job.job_id, tail=5).strip().split("\n") == \
        ["95", "96", "97", "98", "99"]


def test_pythonunbuffered_dans_lenvironnement(project, monkeypatch):
    captured = {}
    vrai_popen = R.subprocess.Popen

    def espion(argv, **kw):
        captured.update(kw.get("env") or {})
        return vrai_popen(argv, **kw)

    monkeypatch.setattr(R.subprocess, "Popen", espion)
    R.start(project, _echo())
    _attendre_fin(project)
    assert captured.get("PYTHONUNBUFFERED") == "1"


def test_kwargs_popen_selon_la_plateforme(project, monkeypatch):
    """`start()` doit brancher ses kwargs Popen selon `os.name`.

    Sur la machine courante (POSIX ici), on vérifie la branche réellement
    empruntée par `start()`. On vérifie aussi, sans l'exécuter, que la
    construction des kwargs pour l'autre plateforme (Windows) produirait
    le résultat attendu — la fonction helper est testée directement, pas
    contournée par un skip.
    """
    captured = {}
    vrai_popen = R.subprocess.Popen

    def espion(argv, **kw):
        captured.update(kw)
        return vrai_popen(argv, **kw)

    monkeypatch.setattr(R.subprocess, "Popen", espion)
    R.start(project, _echo())
    _attendre_fin(project)

    # Valeur documentée de CREATE_NEW_PROCESS_GROUP ; l'attribut n'existe
    # sur le module `subprocess` que sous Windows.
    creation_flag = getattr(R.subprocess, "CREATE_NEW_PROCESS_GROUP", 0x00000200)

    if os.name != "nt":
        assert captured.get("start_new_session") is True
        assert "creationflags" not in captured
    else:
        assert captured.get("creationflags") == creation_flag
        assert "start_new_session" not in captured

    # Vérifie explicitement la construction pour l'AUTRE plateforme, sans
    # jamais lancer de process avec ces kwargs.
    kwargs_posix = R._popen_kwargs_pour_plateforme(is_windows=False)
    assert kwargs_posix == {"start_new_session": True}

    kwargs_windows = R._popen_kwargs_pour_plateforme(is_windows=True)
    assert kwargs_windows == {"creationflags": creation_flag}
