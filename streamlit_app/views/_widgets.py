"""Widgets partagés entre plusieurs pages d'étape.

Contrairement à `views/_job.py` (affichage de job), ce module regroupe des
éléments de formulaire réutilisés tels quels par plusieurs pages — rien de
spécifique à un script du pipeline ne doit y entrer.
"""
from __future__ import annotations

from pathlib import Path

import streamlit as st

from lib import sessions


def selecteur_sessions(projet: Path, *, cle: str) -> tuple[list[str], bool]:
    """Retourne (sessions choisies, drapeau --all).

    Utilisé par les pages Pose, Nettoyage et Vidéos & calibration : chacune
    lance un script qui accepte soit `--all`, soit une liste de
    `session_id` en positionnel — jamais les deux à la fois (voir
    `lib.pipeline`, qui exclut les sessions dès que `all_sessions=True`).
    """
    df = sessions.list_sessions(projet)
    tout = st.checkbox("Toutes les sessions (`--all`)", value=True, key=f"{cle}_all")
    choisies = st.multiselect(
        "Sessions", options=list(df["session_id"]), disabled=tout, key=f"{cle}_sel",
    )
    return choisies, tout
