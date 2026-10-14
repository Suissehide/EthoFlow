"""lib/projets.py : lignes de la liste des projets et sous-étapes de création."""
from __future__ import annotations

import json
from datetime import datetime

from lib import projets as PR


def _job(projet, job_id, script, state, ended="2026-01-01T10:00:00"):
    d = projet / ".ethoflow" / "jobs"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{job_id}.json").write_text(json.dumps({
        "job_id": job_id, "script": script, "env": "x", "label": script, "argv": [],
        "started_at": "2026-01-01T09:00:00", "ended_at": ended, "returncode": 1,
        "pid": None, "state": state, "cancel_requested": False, "owner_pid": None}))


def test_apercu_projet_vide(project):
    a = PR.apercu(project)
    assert a["type"] == "1 animal par vidéo" and a["sessions"] == 0 and a["modele"] is None
    assert a["segments"][0] == "done" and set(a["segments"][1:]) == {"todo"}
    assert a["statut"] == "02 · Données" and a["couleur"] == PR.ENCRE_SECONDAIRE


def test_apercu_echec_colore_l_etape_en_rouge(project, session_factory):
    session_factory("S1")
    d = project / "data" / "dlc-output" / "S1"
    d.mkdir(parents=True)
    (d / "S1DLC.h5").write_bytes(b"\x00")
    (d / "S1_clean.h5").write_bytes(b"\x00")
    _job(project, "20260101-090000-000000-run_vame", "run_vame.py", "failed")
    a = PR.apercu(project)
    assert a["segments"][5] == "fail" and a["segments"][4] == "done"
    assert a["statut"] == "06 · VAME en échec" and a["couleur"] == PR.COULEURS["fail"]
    assert a["activite"] == datetime(2026, 1, 1, 10, 0)


def test_tri_par_activite(tmp_path, project):
    import shutil
    autre = tmp_path / "autre"
    shutil.copytree(project, autre)
    _job(project, "20260101-090000-000000-x", "x.py", "succeeded", ended="2026-01-01T10:00:00")
    _job(autre, "20260102-090000-000000-x", "x.py", "succeeded", ended="2026-01-02T10:00:00")
    assert [a["nom"] for a in PR.apercus(tmp_path, [project, autre])] == ["autre", project.name]


def test_date_relative():
    m = datetime(2026, 6, 20, 15, 0)
    assert PR.date_relative(datetime(2026, 6, 20, 14, 55), m) == "il y a 5 min"
    assert PR.date_relative(datetime(2026, 6, 19, 9, 0), m) == "hier"
    assert PR.date_relative(datetime(2026, 6, 12, 9, 0), m) == "12 juin"
    assert PR.date_relative(datetime(2025, 6, 12, 9, 0), m) == "12 juin 2025"


def test_sous_etapes_creation():
    journal = "  ✓ data/raw/\n  ✓ configs/pipeline_config.yaml créé\n"
    assert PR.sous_etapes_creation(journal, "running") == [
        ("Dossiers data/ et configs/ créés", "done"),
        ("Configuration du pipeline écrite", "done"),
        ("Modèle Excel généré", "run"),
        ("Ouverture du projet", "todo"),
    ]
    assert PR.sous_etapes_creation(journal, "failed")[2][1] == "fail"
    tout = journal + "  ✓ p_sessions.xlsx généré (à la racine du projet)\n"
    assert all(e == "done" for _, e in PR.sous_etapes_creation(tout, "succeeded"))
