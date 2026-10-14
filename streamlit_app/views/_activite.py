"""Colonne « Activité », visible sur toutes les pages d'un projet (refonte 1c).

Trois blocs : le job en cours (barre, compteur, temps restant, Annuler),
l'historique des derniers jobs, puis « Ensuite » — l'étape qui suit et sa
condition d'ouverture. Positionnée en CSS (`style.css`,
`.st-key-ef_activite`) : une colonne Streamlit imposerait un niveau
d'imbrication de plus à toutes les pages.
"""
from __future__ import annotations

from datetime import datetime
from html import escape
from pathlib import Path

import streamlit as st

from lib import etapes as E
from lib import runner

COULEURS = {"succeeded": "#2f9567", "running": "#c27a12", "failed": "#c9473f",
            "cancelled": "#b9b5ac", "interrupted": "#c9473f"}
ETATS = {"succeeded": "Terminé", "running": "En cours", "failed": "Échec",
         "cancelled": "Annulé", "interrupted": "Interrompu"}


def heure(iso: str | None) -> str:
    try:
        return datetime.fromisoformat(iso).strftime("%H:%M")
    except (TypeError, ValueError):
        return "—"


def temps_restant(debut_iso: str, valeur: float, maintenant: datetime | None = None) -> str | None:
    """« ≈ 18 min » par extrapolation linéaire du temps écoulé, ou None
    tant que la progression est trop faible pour être parlante."""
    if valeur < 0.02 or valeur >= 1:
        return None
    try:
        ecoule = ((maintenant or datetime.now()) - datetime.fromisoformat(debut_iso)).total_seconds()
    except (TypeError, ValueError):
        return None
    minutes = round(ecoule * (1 - valeur) / valeur / 60)
    if minutes < 1:
        return "< 1 min"
    return f"≈ {minutes} min" if minutes < 60 else f"≈ {minutes // 60} h {minutes % 60:02d}"


@st.fragment(run_every=2.0)
def _job_en_cours(projet: Path) -> None:
    job = runner.current(projet)
    if job is None or job.state != "running":
        st.rerun()   # le job vient de finir : toute la page se met à jour
    prog = runner.progress(projet, job.job_id) or {}
    valeur = prog.get("valeur", 0.0)
    with st.container(key="carte_job"):
        titre, annuler = st.columns([3, 1], vertical_alignment="center")
        titre.markdown(f"**{escape(job.label)}**")
        with annuler:
            if st.button("Annuler", key="ef_annuler"):
                runner.cancel(projet, job.job_id)
                st.rerun()
        compte = (f"{prog['i'] - 1} / {prog['n']}" if prog.get("n")
                  else f"{prog['pct']} %" if prog.get("pct") is not None
                  else f"depuis {heure(job.started_at)}")
        reste = temps_restant(job.started_at, valeur) or ""
        st.markdown(
            f'<div class="ef-barre"><div style="width:{100 * valeur:.0f}%"></div></div>'
            f'<div class="ef-job-pied" style="margin-top:10px"><span>{compte}</span>'
            f"<span>{reste}</span></div>",
            unsafe_allow_html=True)


def _historique(projet: Path, sauf: str | None) -> None:
    lignes = []
    for job in runner.history(projet, limit=8):
        if job.job_id == sauf:
            continue
        couleur = COULEURS.get(job.state, "#b9b5ac")
        lignes.append(
            f'<div class="ef-historique-ligne"><span class="ef-dot" style="background:{couleur}"></span>'
            f"<div><b>{escape(job.label)}</b>"
            f"<span>{ETATS.get(job.state, job.state)} · {heure(job.ended_at or job.started_at)}</span>"
            "</div></div>")
    if lignes:
        st.markdown(f'<div class="ef-historique">{"".join(lignes[:6])}</div>',
                    unsafe_allow_html=True)
    else:
        st.caption("Aucun job lancé sur ce projet pour l'instant.")


def render(projet: Path, lignes: list[dict]) -> None:
    """`lignes` : `etapes.etat_pipeline(...)`, déjà calculé par app.py."""
    with st.container(key="ef_activite"):
        st.markdown('<p class="ef-activite-titre">Activité</p>', unsafe_allow_html=True)
        en_cours = runner.current(projet) if runner.is_running(projet) else None
        if en_cours is not None:
            _job_en_cours(projet)
        _historique(projet, en_cours.job_id if en_cours else None)
        suite = E.ensuite(lignes)
        if suite:
            e = suite["etape"]
            # Bloc cliquable : il mène à la page de l'étape suivante.
            with st.container(key="ef_ensuite"):
                st.markdown(
                    f'<div class="ef-ensuite"><p class="ef-label">Ensuite</p>'
                    f"<b>{e.n:02d} · {escape(e.nom)} →</b>"
                    f"<span>{escape(suite['texte'])}</span></div>",
                    unsafe_allow_html=True)
                if st.button(f"Ouvrir : {e.nom}", key="ef_ensuite_btn", help=f"Ouvrir : {e.nom}"):
                    st.session_state.current_page = e.nom
                    st.rerun()
