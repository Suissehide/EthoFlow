"""Échecs de job expliqués en langage clair (refonte 1c, bannière d'échec).

Une table motif d'erreur → cause + marche à suivre, lue de haut en bas :
le premier motif présent dans la fin du journal l'emporte. Sans
correspondance, `expliquer` renvoie None et l'interface garde le message
générique (« la cause est dans les dernières lignes du journal »).

Ce module n'importe pas Streamlit.
"""
from __future__ import annotations

import re

# (motif, cause, marche à suivre). `{1}` reprend le premier groupe capturé.
_TABLE: tuple[tuple[str, str, str], ...] = (
    (r"CUDA out of memory|OutOfMemoryError|CUBLAS_STATUS_ALLOC_FAILED",
     "La carte graphique n'a plus assez de mémoire.",
     "Réduis la taille de lot (option « taille de lot » de l'étape), ou ferme "
     "les autres programmes qui utilisent le GPU, puis relance."),
    (r"No space left on device",
     "Le disque est plein.",
     "Libère de la place sur le disque du projet, puis relance."),
    (r"\bMemoryError\b|Killed\b",
     "La mémoire vive de l'ordinateur est épuisée.",
     "Traite moins de sessions à la fois ou ferme d'autres programmes, puis relance."),
    (r"Could not find a shuffle|No snapshots? found|snapshot.*not found",
     "Le modèle DLC est incomplet ou mal référencé (aucun réseau entraîné trouvé).",
     "Ouvre le menu du projet → Modèle DLC → « Diagnostiquer », puis « Réparer » si besoin."),
    (r"EnvironmentLocationNotFound|EnvironmentNameNotFound|Could not find conda environment",
     "Un environnement conda nécessaire n'est pas installé sur cette machine.",
     "Page Configuration : vérifie les environnements, puis crée celui qui manque."),
    (r"No module named '([\w.]+)'",
     "Il manque le module Python « {1} » dans l'environnement de ce job.",
     "Installe-le dans l'environnement conda indiqué dans « Détails techniques », "
     "ou recrée l'environnement depuis son fichier .yml."),
    (r"PermissionError|Permission denied",
     "Un fichier est verrouillé ou en lecture seule.",
     "Ferme le fichier s'il est ouvert ailleurs (Excel, lecteur vidéo), puis relance."),
    (r"moov atom not found|Could not open video|cannot open video|Invalid data found when processing input",
     "Une vidéo est illisible : fichier corrompu ou copie incomplète.",
     "Page Vidéos : vérifie la vidéo de la session concernée, recopie-la si besoin."),
    (r"FileNotFoundError: .*?['\"]([^'\"]+)['\"]",
     "Un fichier attendu est introuvable : {1}",
     "Disque débranché ou fichier déplacé ? Page Vidéos → « Re-pointer » pour les "
     "vidéos, sinon vérifie que l'étape précédente a bien produit ce fichier."),
)


def expliquer(journal: str) -> dict | None:
    """`{cause, conseil}` pour la fin d'un journal d'échec, ou None."""
    for motif, cause, conseil in _TABLE:
        m = re.search(motif, journal or "")
        if m:
            groupe = m.group(1) if m.groups() else ""
            return {"cause": cause.replace("{1}", groupe),
                    "conseil": conseil.replace("{1}", groupe)}
    return None
