"""Données de la page Projets (handoff « projets », écrans 2a/2b).

Pour chaque projet d'un emplacement : type, sessions, modèle DLC, frise
d'avancement des 9 étapes, ligne d'état (« 04 · Pose en cours — 6/10 »),
date de la dernière activité. Et, pour la création, les sous-étapes
lues dans le journal de `create_project.py`.

Ce module n'importe pas Streamlit.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

from lib import etapes as E
from lib import runner
from lib.project import dlc_config_path, est_projet, project_kind
from lib.sessions import list_sessions

COULEURS = {"done": "#2f9567", "run": "#c27a12", "fail": "#c9473f", "todo": "#e4e2db"}
ENCRE_SECONDAIRE = "#56534d"


def _dernier_job(projet: Path):
    historique = runner.history(projet, limit=1)
    return historique[0] if historique else None


def _date(job, projet: Path) -> datetime:
    for iso in (job.ended_at, job.started_at) if job else ():
        try:
            return datetime.fromisoformat(iso)
        except (TypeError, ValueError):
            pass
    return datetime.fromtimestamp(projet.stat().st_mtime)


def apercu(projet: Path) -> dict:
    """Une ligne de la liste des projets."""
    projet = Path(projet)
    en_cours = runner.current(projet) if runner.is_running(projet) else None
    lignes = E.etat_pipeline(projet, en_cours.script if en_cours else None)
    dernier = _dernier_job(projet)
    echec = (dernier is not None and dernier.state == "failed"
             and en_cours is None and E.etape_du_script(dernier.script))

    cour = E.courante(lignes)
    if echec and (cour is None or echec.n <= cour["etape"].n):
        cour = next(x for x in lignes if x["etape"] is echec)
    segments = []
    for ligne in lignes:
        if cour is not None and ligne is cour:
            segments.append("fail" if echec and ligne["etape"] is echec
                            else "run" if en_cours else "todo")
        else:
            segments.append("done" if ligne["etat"] == "done" or ligne["etape"].n == 1 else "todo")

    if cour is None:
        statut, couleur = "Terminé", COULEURS["done"]
    else:
        e = cour["etape"]
        compte = f" — {cour['fait']}/{cour['total']}" if cour["total"] else ""
        if segments[e.n - 1] == "fail":
            statut, couleur = f"{e.n:02d} · {e.nom} en échec", COULEURS["fail"]
        elif segments[e.n - 1] == "run":
            statut, couleur = f"{e.n:02d} · {e.nom} en cours{compte}", COULEURS["run"]
        else:
            statut, couleur = f"{e.n:02d} · {e.nom}{compte}", ENCRE_SECONDAIRE

    chemin_modele = dlc_config_path(projet)
    return {
        "chemin": projet, "nom": projet.name,
        "type": "Arènes séparées" if project_kind(projet) == "multi" else "1 animal par vidéo",
        "sessions": len(list_sessions(projet)),
        "modele": Path(chemin_modele).parent.name if chemin_modele else None,
        "segments": segments, "statut": statut, "couleur": couleur,
        "activite": _date(dernier, projet),
    }


def apercus(racine: Path, projets: list[Path]) -> list[dict]:
    """Les projets, du plus récemment actif au moins récent."""
    return sorted((apercu(p) for p in projets if est_projet(p)),
                  key=lambda a: a["activite"], reverse=True)


def date_relative(quand: datetime, maintenant: datetime | None = None) -> str:
    """« il y a 5 min », « hier », « 12 juin »."""
    maintenant = maintenant or datetime.now()
    secondes = (maintenant - quand).total_seconds()
    if secondes < 60:
        return "à l'instant"
    if secondes < 3600:
        return f"il y a {int(secondes // 60)} min"
    if quand.date() == maintenant.date():
        return f"il y a {int(secondes // 3600)} h"
    if (maintenant.date() - quand.date()).days == 1:
        return "hier"
    mois = ["janv.", "févr.", "mars", "avr.", "mai", "juin", "juil.", "août",
            "sept.", "oct.", "nov.", "déc."][quand.month - 1]
    annee = f" {quand.year}" if quand.year != maintenant.year else ""
    return f"{quand.day} {mois}{annee}"


# Sous-étapes de la création, repérées dans le journal de create_project.py.
SOUS_ETAPES_CREATION = (
    ("Dossiers data/ et configs/ créés", ("✓ data",)),
    ("Configuration du pipeline écrite", ("pipeline_config.yaml créé", "pipeline_config.yaml existe déjà")),
    ("Modèle Excel généré", ("_sessions.xlsx généré", "_sessions.xlsx existe déjà",
                             "Template Excel non généré")),
    ("Ouverture du projet", ()),
)


def sous_etapes_creation(journal: str, etat_job: str) -> list[tuple[str, str]]:
    """(libellé, état) ; état : done / run / todo / fail."""
    sorties, courant_pose = [], False
    for libelle, marqueurs in SOUS_ETAPES_CREATION:
        fait = (etat_job == "succeeded" if not marqueurs
                else any(m in journal for m in marqueurs))
        if fait:
            sorties.append((libelle, "done"))
        elif not courant_pose:
            courant_pose = True
            sorties.append((libelle, "fail" if etat_job == "failed" else
                            "run" if etat_job == "running" else "todo"))
        else:
            sorties.append((libelle, "todo"))
    return sorties
