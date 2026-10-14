"""Affichage du job en cours — partagé par toutes les pages d'étape."""
from __future__ import annotations

from pathlib import Path

import streamlit as st

from lib import runner
from lib.erreurs import expliquer
from views._widgets import verrou_etape

# Pastille (code couleur unique de la refonte) et libellé par état.
_ETATS = {
    "running":     ("#c27a12", "En cours"),
    "succeeded":   ("#2f9567", "Terminé"),
    "failed":      ("#c9473f", "Échec"),
    "cancelled":   ("#b9b5ac", "Annulé"),
    "interrupted": ("#c9473f", "Interrompu : l'app a été fermée pendant le job"),
}


def _ligne_job(job, *, heure_debut: bool = True) -> str:
    from html import escape

    from views._activite import heure

    couleur, texte = _ETATS.get(job.state, ("#b9b5ac", job.state))
    quand = f" · {heure(job.started_at)}" if heure_debut else ""
    return (f'<span class="ef-dot" style="background:{couleur}"></span> '
            f"<b>{escape(job.label)}</b> — {texte}{quand}")


def bouton_lancer(projet: Path, label: str, cmd, *, cle: str,
                  type: str = "primary", disabled: bool = False,
                  help: str | None = None, afficher_raison: bool = True) -> None:
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
            from views._activite import heure
            aide = f"« {en_cours.label} » tourne déjà (démarré à {heure(en_cours.started_at)})."
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
    # Le verrou d'étape est le même pour tous les boutons d'une page : écrit
    # sous le premier seulement (les suivants le gardent en infobulle).
    vues = st.session_state.setdefault("_raisons_vues", set())
    if bloque and aide and afficher_raison and not (aide == verrou and aide in vues):
        st.caption(aide)
        vues.add(aide)


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
            st.session_state.setdefault("_raisons_vues", set()).add(verrou)
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
        with st.expander("Journal complet", expanded=False):
            st.code(runner.read_log(projet, job.job_id) or "(vide)", language="text")
        return

    entete, action = st.columns([5, 1], vertical_alignment="center")
    with entete:
        st.markdown(_ligne_job(job), unsafe_allow_html=True)
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

    with st.expander("Journal complet", expanded=False):
        st.code(runner.read_log(projet, job.job_id) or "(vide)", language="text")



def historique(projet: Path, limite: int = 10) -> None:
    """Liste dépliable des jobs précédents, avec leur log complet."""
    jobs = runner.history(projet, limit=limite)
    if not jobs:
        return
    with st.expander(f"Journaux des jobs précédents ({len(jobs)})", expanded=False):
        for job in jobs:
            st.markdown(_ligne_job(job), unsafe_allow_html=True)
            # Pas d'expander imbriqué (déconseillé par Streamlit) : la
            # visibilité du log de CE job se pilote par une case à cocher,
            # à plat dans l'expander parent — lisible même avec dix jobs.
            if st.checkbox("Voir le journal", key=f"voir_log_{job.job_id}"):
                st.code(runner.read_log(projet, job.job_id, tail=200) or "(vide)",
                        language="text")
            st.divider()
