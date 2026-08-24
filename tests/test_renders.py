"""Tests de `lib.renders` (Task 22) : découverte des rendus de la page
Visualisations et recouvrement de leurs paramètres depuis l'`argv` d'un job.

Sans Streamlit, sans conda, sans GPU — fabrique juste des fichiers de
rendus factices et des `.json`/`.log` de jobs à la forme que
`lib.runner.start()` écrit réellement (mêmes conventions que
`tests/test_app_analyses.py::_ecrire_job_list_columns_succeeded`).
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from lib import pipeline as PL
from lib import renders as RD


# ============================================================
# Fabrique de jobs — mêmes champs que `lib.runner.Job`
# ============================================================

def _ecrire_job(project: Path, *, job_id: str, script: str, argv: list[str],
                started_at: datetime, ended_at: datetime | None,
                state: str = "succeeded") -> None:
    jobs_dir = project / ".ethoflow" / "jobs"
    jobs_dir.mkdir(parents=True, exist_ok=True)
    job = {
        "job_id": job_id, "script": script, "env": "vame", "label": "test",
        "argv": argv,
        "started_at": started_at.isoformat(timespec="seconds"),
        "ended_at": ended_at.isoformat(timespec="seconds") if ended_at else None,
        "returncode": 0 if state == "succeeded" else 1,
        "pid": None, "state": state, "cancel_requested": False, "owner_pid": None,
    }
    (jobs_dir / f"{job_id}.json").write_text(json.dumps(job), encoding="utf-8")
    (jobs_dir / f"{job_id}.log").write_text("ok", encoding="utf-8")


def _toucher(path: Path, quand: datetime) -> None:
    """Pose la mtime d'un fichier factice à une date donnée."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x00")
    ts = quand.timestamp()
    os.utime(path, (ts, ts))


# ============================================================
# parse_motif_gif_args
# ============================================================

def test_parse_motif_gif_args_aller_retour(project):
    cmd = PL.motif_gif(project, session="S1", algo="hmm", start=12.0,
                       duration=30.0, output_format="gif", labels="/x/motif_labels.csv")
    kwargs = RD.parse_motif_gif_args(PL.to_argv(cmd))
    assert kwargs == {
        "session": "S1", "algo": "hmm", "start": 12.0, "duration": 30.0,
        "output_format": "gif", "labels": "/x/motif_labels.csv",
    }


def test_parse_motif_gif_args_options_par_defaut_absentes(project):
    cmd = PL.motif_gif(project, session="S1")
    kwargs = RD.parse_motif_gif_args(PL.to_argv(cmd))
    assert kwargs["duration"] is None
    assert kwargs["labels"] is None
    assert kwargs["output_format"] == "mp4"


def test_parse_motif_gif_args_script_different_refuse():
    assert RD.parse_motif_gif_args(
        ["conda", "run", "-n", "vame", "python", "community_dendrogram.py",
         "--algo", "hmm"]
    ) is None


def test_parse_motif_gif_args_flag_inconnu_refuse(project):
    argv = PL.to_argv(PL.motif_gif(project, session="S1")) + ["--strip-height", "60"]
    assert RD.parse_motif_gif_args(argv) is None


def test_parse_motif_gif_args_sans_session_refuse():
    assert RD.parse_motif_gif_args(
        ["conda", "run", "-n", "ethoflow", "python", "motif_gif.py",
         "--project-dir", "/p", "--no-prompt", "--algo", "hmm"]
    ) is None


# ============================================================
# parse_behavior_structure_gif_args
# ============================================================

def test_parse_behavior_structure_gif_args_aller_retour(project):
    cmd = PL.behavior_structure_gif(
        project, session="S1", algo="kmeans", projection="pca",
        start=5.0, duration=None, output_format="mp4",
        with_video=True, pool_all_sessions=True, labels=None,
    )
    kwargs = RD.parse_behavior_structure_gif_args(PL.to_argv(cmd))
    assert kwargs == {
        "session": "S1", "algo": "kmeans", "projection": "pca",
        "start": 5.0, "duration": None, "output_format": "mp4",
        "with_video": True, "pool_all_sessions": True, "labels": None,
    }


def test_parse_behavior_structure_gif_args_drapeaux_absents_par_defaut(project):
    cmd = PL.behavior_structure_gif(project, session="S1")
    kwargs = RD.parse_behavior_structure_gif_args(PL.to_argv(cmd))
    assert kwargs["with_video"] is False
    assert kwargs["pool_all_sessions"] is False


# ============================================================
# parse_community_dendrogram_args
# ============================================================

def test_parse_community_dendrogram_args_aller_retour(project):
    cmd = PL.community_dendrogram(project, algo="hmm", group="MCCiECKO",
                                  linkage="average", labels="/x/motif_labels.csv")
    kwargs = RD.parse_community_dendrogram_args(PL.to_argv(cmd))
    assert kwargs == {
        "algo": "hmm", "group": "MCCiECKO", "linkage": "average",
        "labels": "/x/motif_labels.csv",
    }


def test_parse_community_dendrogram_args_sans_groupe(project):
    cmd = PL.community_dendrogram(project)
    kwargs = RD.parse_community_dendrogram_args(PL.to_argv(cmd))
    assert kwargs["group"] is None
    assert kwargs["algo"] == "hmm"
    assert kwargs["linkage"] == "ward"


# ============================================================
# Listage des fichiers de rendu — triés du plus récent au plus ancien
# ============================================================

def test_list_motif_gifs_tri_par_mtime_desc(project):
    d = RD.motif_gif_dir(project)
    maintenant = datetime.now()
    _toucher(d / "S1_annotated.mp4", maintenant - timedelta(hours=2))
    _toucher(d / "S2_annotated.gif", maintenant)
    fichiers = RD.list_motif_gifs(project)
    assert [p.name for p in fichiers] == ["S2_annotated.gif", "S1_annotated.mp4"]


def test_list_motif_gifs_dossier_absent(project):
    assert RD.list_motif_gifs(project) == []


def test_list_manifolds_ignore_extensions_inconnues(project):
    d = RD.manifold_dir(project)
    _toucher(d / "S1_manifold_umap_full.gif", datetime.now())
    _toucher(d / "poolé.npz", datetime.now())  # cache pool, pas un rendu
    fichiers = RD.list_manifolds(project)
    assert [p.name for p in fichiers] == ["S1_manifold_umap_full.gif"]


def test_list_dendrograms_pattern_prefixe(project):
    from lib.vame import vame_project
    d = vame_project(project) / "analysis"
    _toucher(d / "community_dendrogram.png", datetime.now() - timedelta(hours=1))
    _toucher(d / "community_dendrogram_MCCiECKO.png", datetime.now())
    _toucher(d / "autre_figure.png", datetime.now())
    fichiers = RD.list_dendrograms(project)
    assert [p.name for p in fichiers] == [
        "community_dendrogram_MCCiECKO.png", "community_dendrogram.png",
    ]


# ============================================================
# match_render : fenêtre temporelle [started_at, ended_at]
# ============================================================

def test_match_render_associe_le_job_dont_la_fenetre_contient_le_fichier(project):
    debut = datetime(2026, 1, 1, 10, 0, 0)
    fin = datetime(2026, 1, 1, 10, 5, 0)
    argv = PL.to_argv(PL.motif_gif(project, session="S1", start=0.0, duration=30.0))
    from lib.runner import Job
    job = Job(job_id="j1", script="motif_gif.py", env="ethoflow", label="x",
             argv=argv, started_at=debut.isoformat(timespec="seconds"),
             ended_at=fin.isoformat(timespec="seconds"), returncode=0, pid=None,
             state="succeeded")

    fichier = RD.motif_gif_dir(project) / "S1_annotated_0s_30s.mp4"
    _toucher(fichier, debut + timedelta(minutes=2))

    rendu = RD.match_render(fichier, [job], RD.parse_motif_gif_args, RD.valide_motif_gif)
    assert rendu.job is job
    assert rendu.params["session"] == "S1"
    assert rendu.params["duration"] == 30.0


def test_match_render_aucun_job_dans_la_fenetre(project):
    debut = datetime(2026, 1, 1, 10, 0, 0)
    fin = datetime(2026, 1, 1, 10, 5, 0)
    argv = PL.to_argv(PL.motif_gif(project, session="S1"))
    from lib.runner import Job
    job = Job(job_id="j1", script="motif_gif.py", env="ethoflow", label="x",
             argv=argv, started_at=debut.isoformat(timespec="seconds"),
             ended_at=fin.isoformat(timespec="seconds"), returncode=0, pid=None,
             state="succeeded")

    fichier = RD.motif_gif_dir(project) / "S1_annotated.mp4"
    _toucher(fichier, debut - timedelta(hours=3))   # bien avant le job

    rendu = RD.match_render(fichier, [job], RD.parse_motif_gif_args, RD.valide_motif_gif)
    assert rendu.job is None
    assert rendu.params is None
    assert rendu.path == fichier


# ============================================================
# R22.1 : deux jobs du même script rapprochés dans le temps ne doivent
# jamais faire étiqueter un fichier avec les paramètres de l'AUTRE job.
# ============================================================

def test_match_render_ne_mislabel_pas_avec_un_job_voisin_motif_gif(project):
    """Job A (S1) très rapide, puis Job B (S2) démarre 2 s après sa fin.
    La fenêtre élargie (±2s) de B recouvre la mtime du fichier de A — sans
    la validation R22.1, le job le plus récent (B, itéré en premier)
    aurait gagné et légendé le fichier de S1 avec la session S2."""
    from lib.runner import Job

    debut_a = datetime(2026, 1, 1, 10, 0, 0)
    fin_a = datetime(2026, 1, 1, 10, 0, 1)
    job_a = Job(job_id="a", script="motif_gif.py", env="ethoflow", label="x",
               argv=PL.to_argv(PL.motif_gif(project, session="S1")),
               started_at=debut_a.isoformat(timespec="seconds"),
               ended_at=fin_a.isoformat(timespec="seconds"),
               returncode=0, pid=None, state="succeeded")

    debut_b = datetime(2026, 1, 1, 10, 0, 2)
    fin_b = datetime(2026, 1, 1, 10, 0, 3)
    job_b = Job(job_id="b", script="motif_gif.py", env="ethoflow", label="x",
               argv=PL.to_argv(PL.motif_gif(project, session="S2")),
               started_at=debut_b.isoformat(timespec="seconds"),
               ended_at=fin_b.isoformat(timespec="seconds"),
               returncode=0, pid=None, state="succeeded")

    fichier_a = RD.motif_gif_dir(project) / "S1_annotated.mp4"
    _toucher(fichier_a, fin_a)   # mtime = 10:00:01, dans la fenêtre élargie de B aussi

    # jobs passés newest-first, comme le fait lib.runner.history()
    rendu = RD.match_render(fichier_a, [job_b, job_a], RD.parse_motif_gif_args,
                            RD.valide_motif_gif)
    assert rendu.job is job_a, "le fichier S1 ne doit jamais être associé au job S2"
    assert rendu.params["session"] == "S1"


def test_match_render_sans_producteur_recouvrable_reste_sans_legende(project):
    """Si le job qui a VRAIMENT produit le fichier n'est pas dans la liste
    (log purgé, historique tronqué...), mais qu'un job voisin d'un AUTRE
    fichier tombe dans la fenêtre élargie, le fichier doit rester sans
    légende plutôt que d'hériter des paramètres du voisin."""
    from lib.runner import Job

    debut_b = datetime(2026, 1, 1, 10, 0, 2)
    fin_b = datetime(2026, 1, 1, 10, 0, 3)
    job_b = Job(job_id="b", script="motif_gif.py", env="ethoflow", label="x",
               argv=PL.to_argv(PL.motif_gif(project, session="S2")),
               started_at=debut_b.isoformat(timespec="seconds"),
               ended_at=fin_b.isoformat(timespec="seconds"),
               returncode=0, pid=None, state="succeeded")

    fichier_a = RD.motif_gif_dir(project) / "S1_annotated.mp4"
    _toucher(fichier_a, datetime(2026, 1, 1, 10, 0, 1))   # job producteur absent de la liste

    rendu = RD.match_render(fichier_a, [job_b], RD.parse_motif_gif_args, RD.valide_motif_gif)
    assert rendu.job is None
    assert rendu.params is None


def test_match_render_ne_mislabel_pas_avec_un_job_voisin_dendrogram(project):
    """Même scénario que motif_gif, pour un script rapide (mentionné par
    la revue) : deux `community_dendrogram.py` sur deux groupes différents,
    lancés à quelques secondes d'écart."""
    from lib.runner import Job

    debut_a = datetime(2026, 1, 1, 10, 0, 0)
    fin_a = datetime(2026, 1, 1, 10, 0, 1)
    job_a = Job(job_id="a", script="community_dendrogram.py", env="vame", label="x",
               argv=PL.to_argv(PL.community_dendrogram(project, group="GroupeA")),
               started_at=debut_a.isoformat(timespec="seconds"),
               ended_at=fin_a.isoformat(timespec="seconds"),
               returncode=0, pid=None, state="succeeded")

    debut_b = datetime(2026, 1, 1, 10, 0, 2)
    fin_b = datetime(2026, 1, 1, 10, 0, 3)
    job_b = Job(job_id="b", script="community_dendrogram.py", env="vame", label="x",
               argv=PL.to_argv(PL.community_dendrogram(project, group="GroupeB")),
               started_at=debut_b.isoformat(timespec="seconds"),
               ended_at=fin_b.isoformat(timespec="seconds"),
               returncode=0, pid=None, state="succeeded")

    from lib.vame import vame_project
    fichier_a = vame_project(project) / "analysis" / "community_dendrogram_GroupeA.png"
    _toucher(fichier_a, fin_a)

    rendu = RD.match_render(fichier_a, [job_b, job_a], RD.parse_community_dendrogram_args,
                            RD.valide_dendrogram)
    assert rendu.job is job_a
    assert rendu.params["group"] == "GroupeA"


# ============================================================
# *_renders : intégration avec lib.runner.history (lit le disque)
# ============================================================

def test_motif_gif_renders_relit_les_parametres_depuis_lhistorique(project):
    debut = datetime(2026, 1, 1, 10, 0, 0)
    fin = datetime(2026, 1, 1, 10, 5, 0)
    argv = PL.to_argv(PL.motif_gif(project, session="S1", algo="hmm",
                                   output_format="mp4"))
    _ecrire_job(project, job_id="20260101-100000-000000-motif_gif",
               script="motif_gif.py", argv=argv, started_at=debut, ended_at=fin)

    fichier = RD.motif_gif_dir(project) / "S1_annotated.mp4"
    _toucher(fichier, debut + timedelta(minutes=1))

    rendus = RD.motif_gif_renders(project)
    assert len(rendus) == 1
    assert rendus[0].path == fichier
    assert rendus[0].params["session"] == "S1"
    assert rendus[0].job.job_id == "20260101-100000-000000-motif_gif"


def test_motif_gif_renders_fichier_orphelin_sans_job_correspondant(project):
    fichier = RD.motif_gif_dir(project) / "S1_annotated.mp4"
    _toucher(fichier, datetime.now())
    rendus = RD.motif_gif_renders(project)
    assert len(rendus) == 1
    assert rendus[0].job is None
    assert rendus[0].params is None


def test_manifold_renders_ne_confond_pas_avec_motif_gif(project):
    """Un job `motif_gif.py` réussi ne doit jamais matcher un fichier du
    dossier manifold, même si sa fenêtre temporelle coïncide."""
    debut = datetime(2026, 1, 1, 10, 0, 0)
    fin = datetime(2026, 1, 1, 10, 5, 0)
    argv = PL.to_argv(PL.motif_gif(project, session="S1"))
    _ecrire_job(project, job_id="20260101-100000-000000-motif_gif",
               script="motif_gif.py", argv=argv, started_at=debut, ended_at=fin)

    fichier = RD.manifold_dir(project) / "S1_manifold_umap_full.gif"
    _toucher(fichier, debut + timedelta(minutes=1))

    rendus = RD.manifold_renders(project)
    assert len(rendus) == 1
    assert rendus[0].job is None   # aucun job behavior_structure_gif.py


def test_dendrogram_renders_avec_groupe(project):
    from lib.vame import vame_project
    debut = datetime(2026, 1, 1, 10, 0, 0)
    fin = datetime(2026, 1, 1, 10, 1, 0)
    argv = PL.to_argv(PL.community_dendrogram(project, group="MCCiECKO"))
    _ecrire_job(project, job_id="20260101-100000-000000-community_dendrogram",
               script="community_dendrogram.py", argv=argv,
               started_at=debut, ended_at=fin)

    fichier = vame_project(project) / "analysis" / "community_dendrogram_MCCiECKO.png"
    _toucher(fichier, debut + timedelta(seconds=30))

    rendus = RD.dendrogram_renders(project)
    assert len(rendus) == 1
    assert rendus[0].params["group"] == "MCCiECKO"
