"""Les 9 étapes du pipeline et leur état — source unique de l'interface.

Règle métier (handoff de la refonte) : le pipeline avance **étape par
étape, en lot**. Toutes les sessions passent une étape avant que la
suivante ne s'ouvre, donc l'état se lit par étape (« Pose 6/10 »).

États d'une étape :
- ``done``   : faite pour toutes les sessions ;
- ``run``    : un job de cette étape tourne ;
- ``todo``   : l'étape courante (première incomplète), rien ne tourne ;
- ``next``   : celle qui s'ouvrira après l'étape courante ;
- ``locked`` : plus loin, prérequis non remplis ;
- ``dispo``  : exploitation (Motifs, Analyses, Visualisations) une fois
  VAME fait — pas de notion de « fini », donc jamais ``done``.

Ce module n'importe pas Streamlit.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from lib.sessions import list_sessions


@dataclass(frozen=True)
class Etape:
    n: int
    nom: str          # nom de la page dans l'app
    icone: str        # clé de lib.icons.ICONS
    titre: str        # titre en langage clair
    description: str  # ce que fait l'étape, ce qu'elle produit
    colonne: str | None = None    # colonne « OK » de list_sessions, si suivie par session
    scripts: tuple[str, ...] = ()  # scripts dont le job appartient à l'étape
    apres: str = ""   # « après la pose » : ce qu'on attend quand l'étape est verrouillée


ETAPES: tuple[Etape, ...] = (
    Etape(1, "Projet", "layout-dashboard", "Projet",
          "Vue d'ensemble du projet et de son avancement."),
    Etape(2, "Données", "file-spreadsheet", "Décrire les animaux",
          "Chaque vidéo devient une session avec ses informations : souris, "
          "groupe, traitement, date. EthoFlow les lit depuis le fichier Excel.",
          scripts=("sync_from_excel.py",), apres="après les données"),
    Etape(3, "Vidéos & calibration", "video", "Vidéos et calibration",
          "Vérifie chaque vidéo, calibre les arènes et l'échelle en px/cm.",
          colonne="vidéo", scripts=("crop_arenes.py",), apres="après les vidéos"),
    Etape(4, "Pose (DLC)", "scan-line", "Estimation de pose",
          "Détecte nez, oreilles, pattes et queue image par image. Produit "
          "un fichier de coordonnées par session.",
          colonne="DLC", scripts=("run_dlc_inference.py",), apres="après la pose"),
    Etape(5, "Nettoyage", "brush-cleaning", "Nettoyage des poses",
          "Écarte les points peu fiables et répare les petits trous, pour "
          "donner à VAME des trajectoires propres. Facultatif, mais évite "
          "qu'un point mal suivi devienne un faux comportement.",
          colonne="nettoyage",
          scripts=("prepare_vame_input_custom.py", "assign_arenas.py", "relabel_video.py"),
          apres="après le nettoyage"),
    Etape(6, "VAME", "waypoints", "Segmentation comportementale (VAME)",
          "Apprend un modèle propre à ce projet, puis découpe chaque session "
          "en motifs comportementaux.",
          colonne="VAME", scripts=("run_vame.py",), apres="après VAME"),
    Etape(7, "Motifs", "tags", "Nommer les motifs",
          "Regarde chaque motif et donne-lui un nom et une catégorie : ces "
          "noms remplacent « motif_0, motif_1… » dans toutes les figures."),
    Etape(8, "Analyses", "chart-column", "Comparer les groupes",
          "Compare l'usage des motifs entre groupes expérimentaux, avec les "
          "statistiques.", scripts=("analyze_vame.py",)),
    Etape(9, "Visualisations", "clapperboard", "Visualisations",
          "Bandes de motifs, manifold et dendrogramme pour les figures.",
          scripts=("motif_gif.py", "behavior_structure_gif.py", "community_dendrogram.py")),
)

PAR_PAGE = {e.nom: e for e in ETAPES}
_EXPLOITATION = (7, 8, 9)


def etape_du_script(script: str) -> Etape | None:
    return next((e for e in ETAPES if script in e.scripts), None)


def etat_pipeline(project: Path, script_en_cours: str | None = None) -> list[dict]:
    """Une entrée par étape : l'`Etape` + `fait`, `total`, `etat`, `statut`.

    `script_en_cours` : script du job qui tourne (`runner`), pour marquer
    son étape ``run`` même si ce n'est pas l'étape courante (relance).
    """
    df = list_sessions(project)
    total = len(df)
    en_cours = etape_du_script(script_en_cours) if script_en_cours else None

    lignes = []
    for e in ETAPES:
        if e.n == 1:
            fait = tot = 1
        elif e.n == 2:
            fait = tot = total
        elif e.colonne:
            fait, tot = int((df[e.colonne] == "OK").sum()) if total else 0, total
        else:
            fait = tot = 0
        lignes.append({"etape": e, "fait": fait, "total": tot,
                       "done": e.n not in _EXPLOITATION and tot > 0 and fait == tot})

    courante = next((ligne for ligne in lignes
                     if ligne["etape"].n not in _EXPLOITATION and not ligne["done"]), None)
    for ligne in lignes:
        e = ligne["etape"]
        if en_cours is not None and e.n == en_cours.n:
            ligne["etat"] = "run"
        elif ligne["done"]:
            ligne["etat"] = "done"
        elif ligne is courante:
            ligne["etat"] = "todo"
        elif courante is None:
            ligne["etat"] = "dispo"
        elif e.n == courante["etape"].n + 1:
            ligne["etat"] = "next"
        else:
            ligne["etat"] = "locked"
        ligne["statut"] = _statut(ligne, courante)
    return lignes


def _statut(ligne: dict, courante: dict | None) -> str:
    e, etat = ligne["etape"], ligne["etat"]
    compte = f"{ligne['fait']}/{ligne['total']}" if ligne["total"] else ""
    if e.n == 1:
        return "prêt"
    if etat in ("done", "run"):
        return compte or ("en cours" if etat == "run" else "fait")
    if etat == "todo":
        return compte if ligne["fait"] else "à faire"
    if etat == "next":
        return courante["etape"].apres or "ensuite"
    if etat == "dispo":
        return "disponible"
    return "verrouillé"


def courante(lignes: list[dict]) -> dict | None:
    """L'étape où l'on en est : celle qui tourne, sinon la première à faire."""
    return (next((ligne for ligne in lignes if ligne["etat"] == "run"), None)
            or next((ligne for ligne in lignes if ligne["etat"] == "todo"), None))


def ensuite(lignes: list[dict]) -> dict | None:
    """Ce qui vient après, avec sa condition d'ouverture (« S'ouvre quand… »).

    Retourne `{etape, texte}` ou None quand tout est fait.
    """
    cour = courante(lignes)
    if cour is None:
        dispo = [ligne for ligne in lignes if ligne["etat"] == "dispo"]
        return ({"etape": dispo[0]["etape"],
                 "texte": "VAME est fait : nomme les motifs, puis compare les groupes."}
                if dispo else None)
    suivante = next((ligne for ligne in lignes if ligne["etape"].n == cour["etape"].n + 1), None)
    if suivante is None:
        return None
    total = cour["total"]
    quand = (f"S'ouvre quand l'étape {cour['etape'].nom} aura traité les {total} sessions."
             if total else f"S'ouvre une fois l'étape {cour['etape'].nom} terminée.")
    return {"etape": suivante["etape"], "texte": quand}
