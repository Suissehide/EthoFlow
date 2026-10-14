"""Affichage du job en cours — partagé par toutes les pages d'étape."""
from __future__ import annotations

from pathlib import Path

import streamlit as st

from lib import runner
from lib.erreurs import expliquer
from views._widgets import verrou_etape

_ETATS = {
    "running":     ("⏳", "En cours"),
    "succeeded":   ("✅", "Terminé"),
    "failed":      ("❌", "Échec"),
    "cancelled":   ("⛔", "Annulé"),
    "interrupted": ("⚠️", "Interrompu (l'app a été arrêtée pendant le job)"),
}


def bouton_lancer(projet: Path, label: str, cmd, *, cle: str,
                  type: str = "primary", disabled: bool = False,
                  help: str | None = None) -> None:
    """Bouton de lancement d'un job, désactivé si un autre job tourne déjà.

    Gère `JobBusy` (course entre l'affichage du bouton et le clic) en
    affichant l'erreur plutôt que de laisser l'exception remonter.
    """
    occupe = runner.is_running(projet)
    verrou = verrou_etape()
    aide = verrou or help
    if aide is None and occupe:
        # Le verrou est par projet, pas par page : l'utilisateur peut être
        # sur la page Pose pendant qu'un entraînement VAME lancé depuis une
        # autre page tourne depuis trois heures. Sans nommer ce job ici, il
        # n'a aucun moyen de savoir ce qui bloque le bouton (ruling R9.1).
        en_cours = runner.current(projet)
        if en_cours is not None:
            aide = f"« {en_cours.label} » tourne déjà (démarré à {en_cours.started_at})."
        else:
            aide = "Un autre job tourne déjà."
    bloque = disabled or occupe or verrou is not None
    if st.button(label, key=cle, type=type, disabled=bloque, help=aide):
        try:
            runner.start(projet, cmd)
            st.rerun()
        except runner.JobBusy as e:
            st.error(str(e))
    # Un bouton désactivé dit toujours pourquoi, en toutes lettres (handoff) :
    # l'infobulle seule ne se voit pas.
    if bloque and aide:
        st.caption(aide)


@st.fragment(run_every=2.0)
def _log_en_direct(projet: Path, job_id: str, lignes: int) -> None:
    """Rafraîchit le log toutes les 2 s sans recharger la page entière.

    `st.fragment(run_every=...)` demande Streamlit >= 1.33. Dès que le job
    quitte l'état `running`, on déclenche un `st.rerun()` complet (pas un
    rerun de fragment) pour que le reste de la page — sections
    « artefacts produits », etc. — se mette à jour en conséquence.
    """
    prog = runner.progress(projet, job_id)
    if prog:
        st.progress(prog["valeur"], text=prog["texte"])
    st.code(runner.read_log(projet, job_id, tail=lignes) or "(pas encore de sortie)",
            language="text")
    job = runner.current(projet)
    if job and job.state != "running":
        # Le job vient de finir : on sort du fragment pour rafraîchir la page,
        # afin que les sections « artefacts produits » apparaissent.
        st.rerun()


def _duree(job) -> str:
    from datetime import datetime
    try:
        secondes = (datetime.fromisoformat(job.ended_at)
                    - datetime.fromisoformat(job.started_at)).total_seconds()
    except (TypeError, ValueError):
        return ""
    minutes = round(secondes / 60)
    return (" après moins d'une minute" if minutes < 1 else f" après {minutes} min"
            if minutes < 60 else f" après {minutes // 60} h {minutes % 60:02d}")


def _banniere_echec(projet: Path, job) -> None:
    """Cause et marche à suivre en clair, puis la fin du journal (maquette 1c)."""
    from html import escape

    from lib.icons import lucide_data_uri

    fin = runner.read_log(projet, job.job_id, tail=12)
    explication = expliquer(fin)
    cause = (f"{explication['cause']} {explication['conseil']}" if explication
             else "La cause est dans les dernières lignes du journal ci-dessous.")
    with st.container(key="carte_echec"):
        texte, action = st.columns([5, 1], vertical_alignment="top")
        texte.markdown(
            f'<div class="ef-echec"><img src="{lucide_data_uri("circle-x", "c9473f")}" '
            f'width="22" height="22"><div><p class="ef-echec-titre">'
            f"« {escape(job.label)} » a échoué{_duree(job)}</p>"
            f'<p class="ef-echec-cause">{escape(cause)}</p></div></div>',
            unsafe_allow_html=True)
        # Relancer suit la même règle que Lancer : l'étape du job doit être ouverte.
        from lib.etapes import etape_du_script
        etape = etape_du_script(job.script)
        verrou = verrou_etape(etape.nom) if etape else None
        with action:
            if st.button("Relancer", key=f"relancer_{job.job_id}", type="primary",
                         disabled=runner.is_running(projet) or verrou is not None,
                         help=verrou):
                runner.restart(projet, job)
                st.rerun()
        if verrou:
            st.caption(verrou)
        st.markdown(f'<p class="ef-label" style="margin-top:12px">Dernières lignes du journal · '
                    f"code de retour {job.returncode}</p>", unsafe_allow_html=True)
        st.code(fin or "(pas de sortie)", language="text")


def panneau(projet: Path, *, lignes: int = 30) -> None:
    """État du job courant : étiquette, log en direct, annulation.

    Ne met rien en cache dans `session_state` : chaque rendu relit l'état
    depuis `lib.runner`, qui lit lui-même les fichiers sous
    `<projet>/.ethoflow/jobs/` — seule source de vérité, la seule à
    survivre à une navigation, un rafraîchissement du navigateur ou une
    fermeture d'onglet.
    """
    job = runner.current(projet)
    if job is None:
        return
    # Le détail d'un job (log, bannière d'échec) vit sur la page de SON
    # étape ; ailleurs, la colonne Activité le signale déjà. Un job sans
    # étape (diagnostic du modèle DLC…) reste affiché là où on l'a lancé.
    from lib.etapes import PAR_PAGE, etape_du_script
    etape_job = etape_du_script(job.script)
    page = PAR_PAGE.get(st.session_state.get("current_page", ""))
    if etape_job is not None and page is not None and etape_job.n != page.n:
        return
    if job.state == "failed":
        # La bannière dit tout, en clair : pas d'en-tête technique au-dessus.
        _banniere_echec(projet, job)
        with st.expander("Log complet", expanded=False):
            st.code(runner.read_log(projet, job.job_id) or "(vide)", language="text")
        return
    icone, texte = _ETATS.get(job.state, ("•", job.state))

    entete, action = st.columns([5, 1])
    with entete:
        st.markdown(f"**{icone} {job.label}** — {texte}")
        st.caption(f"`{job.script}` (env `{job.env}`) · démarré {job.started_at}")
    with action:
        if job.state == "running":
            if st.button("Annuler", key=f"annuler_{job.job_id}"):
                runner.cancel(projet, job.job_id)
                st.rerun()

    if job.state == "running":
        _log_en_direct(projet, job.job_id, lignes)
    else:
        st.code(runner.read_log(projet, job.job_id, tail=lignes) or "(pas de sortie)",
                language="text")

    with st.expander("Log complet", expanded=False):
        st.code(runner.read_log(projet, job.job_id) or "(vide)", language="text")



def historique(projet: Path, limite: int = 10) -> None:
    """Liste dépliable des jobs précédents, avec leur log complet."""
    jobs = runner.history(projet, limit=limite)
    if not jobs:
        return
    with st.expander(f"Jobs précédents ({len(jobs)})", expanded=False):
        for job in jobs:
            icone, texte = _ETATS.get(job.state, ("•", job.state))
            st.markdown(f"{icone} **{job.label}** — {texte} · {job.started_at}")
            # Pas d'expander imbriqué (déconseillé par Streamlit) : la
            # visibilité du log de CE job se pilote par une case à cocher,
            # à plat dans l'expander parent — lisible même avec dix jobs.
            if st.checkbox("Voir le log", key=f"voir_log_{job.job_id}"):
                st.code(runner.read_log(projet, job.job_id, tail=200) or "(vide)",
                        language="text")
            st.divider()
