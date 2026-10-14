"""lib/etapes.py : état des 9 étapes, étape courante et « ensuite »."""
from __future__ import annotations

from lib import etapes as E


def _etats(lignes):
    return {ligne["etape"].nom: (ligne["etat"], ligne["statut"]) for ligne in lignes}


def _dlc(project, sid):
    d = project / "data" / "dlc-output" / sid
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{sid}DLC.h5").write_bytes(b"\x00")


def test_projet_vide_commence_par_les_donnees(project):
    etats = _etats(E.etat_pipeline(project))
    assert etats["Projet"] == ("done", "prêt")
    assert etats["Données"] == ("todo", "à faire")
    assert etats["Vidéos & calibration"] == ("next", "après les données")
    assert etats["Pose (DLC)"] == ("locked", "verrouillé")


def test_pose_en_cours(project, session_factory):
    for sid in ("S1", "S2", "S3"):
        session_factory(sid)
    _dlc(project, "S1")
    lignes = E.etat_pipeline(project, script_en_cours="run_dlc_inference.py")
    etats = _etats(lignes)
    assert etats["Données"] == ("done", "3/3")
    assert etats["Vidéos & calibration"] == ("done", "3/3")
    assert etats["Pose (DLC)"] == ("run", "1/3")
    assert etats["Nettoyage"] == ("next", "après la pose")   # comme la maquette
    assert etats["VAME"] == ("locked", "verrouillé")
    assert E.courante(lignes)["etape"].nom == "Pose (DLC)"
    suite = E.ensuite(lignes)
    assert suite["etape"].nom == "Nettoyage" and "3 sessions" in suite["texte"]


def test_pose_partielle_sans_job_reste_l_etape_courante(project, session_factory):
    session_factory("S1")
    session_factory("S2")
    _dlc(project, "S1")
    etats = _etats(E.etat_pipeline(project))
    assert etats["Pose (DLC)"] == ("todo", "1/2")
    assert etats["Nettoyage"] == ("next", "après la pose")


def test_relance_d_une_etape_finie_la_marque_en_cours(project, session_factory):
    session_factory("S1")
    etats = _etats(E.etat_pipeline(project, script_en_cours="sync_from_excel.py"))
    assert etats["Données"][0] == "run"
    assert etats["Pose (DLC)"][0] == "todo"


def test_tout_fait_ouvre_l_exploitation(project, session_factory, monkeypatch):
    import pandas as pd
    df = pd.DataFrame([{"session_id": "S1", "vidéo": "OK", "DLC": "OK", "split": "—",
                        "nettoyage": "OK", "VAME": "OK"}])
    monkeypatch.setattr(E, "list_sessions", lambda p: df)
    lignes = E.etat_pipeline(project)
    etats = _etats(lignes)
    assert etats["VAME"] == ("done", "1/1")
    assert etats["Motifs"] == ("dispo", "disponible")
    assert E.courante(lignes) is None
    assert E.ensuite(lignes)["etape"].nom == "Motifs"
