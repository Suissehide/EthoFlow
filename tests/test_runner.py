import os
import shutil
import signal
import threading
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
    job_avant = R.current(project)
    R.cancel(project, job_avant.job_id)
    job = _attendre_fin(project)
    assert job.state == "cancelled"
    # Ne pas se contenter de l'étiquette : la fonction dont le seul rôle
    # est d'arrêter des runs GPU de plusieurs heures doit prouver que le
    # process est vraiment mort, pas juste que le JSON dit "cancelled".
    assert not R._pid_vivant(job.pid)


def test_start_bloque_pendant_que_le_job_annule_est_encore_vivant(project):
    """Régression Critical 1 : un process lent à honorer SIGTERM (ex :
    fermeture de contexte CUDA) doit continuer à bloquer un second
    `start()` tant qu'il est réellement vivant, même si `cancel()` a déjà
    été appelé et que l'intention d'annulation est posée."""
    R.start(project, PL.Command(
        "ethoflow", "__test__",
        ["python", "-c",
         "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); "
         "time.sleep(3)"],
        "ignore-sigterm"))
    job_a = R.current(project)

    # Laisse le child installer son handler avant d'envoyer SIGTERM, sinon
    # le signal peut arriver avant `signal.signal(...)` et le tuer pour de
    # vrai — ce serait un artefact du test, pas la preuve qu'on cherche.
    time.sleep(0.3)
    R.cancel(project, job_a.job_id)

    time.sleep(0.3)   # laisse cancel() envoyer le signal, sans attendre la mort
    assert R._pid_vivant(job_a.pid), "le process ignore SIGTERM : il doit être encore vivant"

    with pytest.raises(R.JobBusy):
        R.start(project, _echo())

    os.kill(job_a.pid, signal.SIGKILL)   # nettoyage : SIGTERM est ignoré
    _attendre_fin(project)


def test_watcher_tardif_ne_vole_pas_le_verrou_dun_job_plus_recent(project, monkeypatch):
    """Régression Critical 2 : entre le moment où le watcher de A a réclamé
    son process (`proc.wait()` a renvoyé -> pid réellement mort et réputé
    disponible par `is_running()`) et le moment où il libère effectivement
    le verrou, un job B a le temps de démarrer légitimement. Le nettoyage
    tardif de A ne doit pas emporter le verrou de B.
    """
    peut_continuer = threading.Event()
    a_atteint_le_nettoyage = threading.Event()
    vrai_liberer = R._liberer_verrou_si_proprietaire

    def liberer_retardee(project, job_id):
        a_atteint_le_nettoyage.set()
        peut_continuer.wait(timeout=5.0)
        vrai_liberer(project, job_id)

    monkeypatch.setattr(R, "_liberer_verrou_si_proprietaire", liberer_retardee)

    R.start(project, _echo("A"))
    job_a = R.current(project)

    # Le watcher de A a fini `proc.wait()` (le process est réclamé) et
    # s'apprête à libérer son verrou, mais on le bloque juste avant.
    assert a_atteint_le_nettoyage.wait(timeout=5.0)
    assert not R._pid_vivant(job_a.pid)     # réclamé : plus dans la table des process

    # Le pid de A est mort et réclamé : `is_running()` le sait déjà, alors
    # même que le nettoyage (écriture finale + libération du verrou) de A
    # n'est pas terminé. B peut donc démarrer légitimement.
    assert not R.is_running(project)
    # B doit être encore en cours au moment où on vérifie son verrou : la
    # même fonction `_liberer_verrou_si_proprietaire` étant patchée pour
    # tout le monde, B écho-instantané se terminerait et libérerait déjà
    # (légitimement) son propre verrou avant qu'on ait pu vérifier quoi
    # que ce soit.
    R.start(project, PL.Command(
        "ethoflow", "__test__", ["python", "-c", "import time; time.sleep(1)"], "B"))
    job_b = R.current(project)
    assert job_b.job_id != job_a.job_id

    # Le nettoyage tardif de A se termine enfin : il ne doit pas emporter
    # le verrou, désormais celui de B (toujours en cours).
    peut_continuer.set()
    time.sleep(0.3)   # laisse le thread de A terminer son appel débloqué
    assert R._lock(project).read_text(encoding="utf-8").strip() == job_b.job_id

    _attendre_fin(project)


def test_verrou_libere_apres_la_fin(project):
    R.start(project, _echo())
    _attendre_fin(project)
    assert not R.is_running(project)
    R.start(project, _echo("deuxieme"))     # ne doit pas lever
    _attendre_fin(project)


def test_job_running_dont_le_process_a_disparu(project):
    """App tuée pendant un job, PUIS relancée : le JSON dit running, le pid
    n'existe plus, et `owner_pid` désigne un process qui n'est plus le
    nôtre (sans quoi `_reconcilier` considérerait, à raison, qu'un watcher
    de CE process va s'en charger — voir R4.4)."""
    R.start(project, _echo())
    job = _attendre_fin(project)
    R._write_job(project, R.replace(
        job, state="running", pid=999999, owner_pid=999998, ended_at=None))
    releve = R.current(project)
    assert releve.state == "interrupted"


def test_reconciliation_orphelin_honore_cancel_requested(project):
    """Régression R4.4 (défense en profondeur) : un job orphelin d'une
    exécution précédente (owner_pid étranger, pid mort) que l'utilisateur
    avait demandé d'annuler doit rester `cancelled`, jamais `interrupted`
    — même quand c'est `_reconcilier`, pas le watcher d'origine, qui le
    constate (l'app a été tuée avant que ce watcher-là ait pu écrire quoi
    que ce soit)."""
    R.start(project, _echo())
    job = _attendre_fin(project)
    orphelin = R.replace(
        job, state="running", pid=999999, owner_pid=999998,
        cancel_requested=True, ended_at=None)
    R._write_job(project, orphelin)
    releve = R.current(project)
    assert releve.state == "cancelled"


def test_reconciliation_ne_marche_pas_sur_lannulation_en_cours(project, monkeypatch):
    """Régression Critical (fix round 2, R4.4) : entre le moment où le
    watcher d'un job qu'ON a lancé réclame son process mort et celui où il
    écrit l'état terminal, marteler `current()`/`history()` ne doit jamais
    faire dévier l'état de l'annulation en cours vers `interrupted`.

    Avant R4.4, `_reconcilier` ne distinguait pas "watcher à moi qui va
    écrire l'état terminal" de "orphelin d'une exécution précédente" : un
    appel concurrent pendant cette fenêtre pouvait écraser un `cancelled`
    à venir par un `interrupted`. Constaté ~1 run sur 20 en pratique — on
    force ici la fenêtre au lieu de compter sur la chance pour la couvrir.

    `_apres_reclamation_hook` est un point d'ancrage no-op en production,
    ajouté spécifiquement pour rendre cette fenêtre déterministe : le
    monkeypatcher ne change aucune décision du watcher, il ne fait que le
    bloquer un instant juste après `proc.wait()`, exactement là où la
    fenêtre existe réellement.
    """
    reclame = threading.Event()
    peut_ecrire = threading.Event()

    def hook_retarde():
        reclame.set()
        peut_ecrire.wait(timeout=5.0)

    monkeypatch.setattr(R, "_apres_reclamation_hook", hook_retarde)

    R.start(project, PL.Command(
        "ethoflow", "__test__", ["python", "-c", "import time; time.sleep(0.2)"], "long"))
    job = R.current(project)
    R.cancel(project, job.job_id)

    # Le process est mort et réclamé (proc.wait() a rendu la main), le
    # watcher est bloqué juste avant d'écrire l'état terminal.
    assert reclame.wait(timeout=5.0)

    # Martèle current()/history() pendant la fenêtre, comme le ferait une
    # page Streamlit qui poll le statut : aucun appel ne doit jamais faire
    # apparaître "interrupted".
    for _ in range(50):
        vu = R.current(project)
        assert vu is None or vu.state in ("running", "cancelled")
        assert all(j.state in ("running", "cancelled") for j in R.history(project))

    peut_ecrire.set()
    job_final = _attendre_fin(project)
    assert job_final.state == "cancelled"


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


def test_lectures_ne_ressuscitent_pas_un_projet_supprime(tmp_path):
    """Régression du projet zombie (ruling R10.6c) : avant ce ruling,
    `jobs_dir()` faisait `mkdir(parents=True, exist_ok=True)` sur CHAQUE
    appel, y compris depuis des lectures pures (`current`, `history`,
    `is_running`). Consulter le panneau de job d'un projet qu'on vient de
    supprimer recréait donc silencieusement `<projet>/.ethoflow/jobs/` —
    un squelette de projet ressuscité par un simple rafraîchissement de
    page. Ces trois fonctions ne doivent plus jamais créer quoi que ce
    soit sur un projet absent."""
    projet = tmp_path / "projet-supprime"
    (projet / "data").mkdir(parents=True)
    shutil.rmtree(projet)
    assert not projet.exists()

    assert R.current(projet) is None
    assert not projet.exists()

    assert R.history(projet) == []
    assert not projet.exists()

    assert R.is_running(projet) is False
    assert not projet.exists()


def test_start_cree_ethoflow_jobs_si_absent(project):
    """`start()` reste le seul point d'écriture autorisé à créer
    `.ethoflow/jobs/` : il doit donc marcher même quand ce dossier n'existe
    pas encore — premier job du projet, ou projet tout juste recréé."""
    assert not (project / ".ethoflow").exists()
    R.start(project, _echo("premier-job"))
    assert (project / ".ethoflow" / "jobs").is_dir()
    job = _attendre_fin(project)
    assert job.state == "succeeded"
