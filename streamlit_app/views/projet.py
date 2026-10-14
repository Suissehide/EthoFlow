"""Page Projet — tableau de bord du projet ouvert (refonte 1c).

En-tête (nom du projet : menu « Tous les projets » + modèle DLC, bouton
« ouvrir le dossier »), neuf tuiles d'étape, puis la carte « étape en
cours / prochaine étape ». Ouvrir, créer et supprimer un projet se font
sur la page Projets (`views/projets.py`).
"""
from __future__ import annotations

from html import escape
from pathlib import Path

import streamlit as st

import lib.pipeline as PL
from lib import reveal, runner
from lib.config import current_project, dlc_config_path, project_kind
from lib.icons import lucide_data_uri
from lib.sessions import list_sessions
from views import _job
from views._widgets import choix_modele_dlc


# ============================================================
# Modèle DLC
# ============================================================

def _section_modele_dlc(projet: Path) -> None:
    st.caption(
        "Le modèle vit hors du projet et n'est jamais copié. Un même modèle "
        "sert à autant de projets que tu veux. Pour en entraîner un nouveau, "
        "voir le Parcours B du README — ça se fait au terminal."
    )
    actuel = dlc_config_path(projet)
    if actuel:
        existe = Path(actuel).is_file()
        (st.success if existe else st.error)(
            f"`{actuel}`" + ("" if existe else " — introuvable, modèle déplacé ?")
        )
    else:
        st.warning(
            "Aucun modèle désigné. L'estimation de pose avec « Mon modèle entraîné » en a besoin."
        )

    choix_modele_dlc(projet, cle="modele")

    col1, col2 = st.columns(2)
    with col1:
        _job.bouton_lancer(projet, "Diagnostiquer", PL.diagnose_dlc_model(projet),
                           cle="btn_diag", type="secondary",
                           disabled=not actuel,
                           help="Répond à l'erreur « Could not find a shuffle… ».")
    with col2:
        _job.bouton_lancer(projet, "Réparer", PL.diagnose_dlc_model(projet, fix=True),
                           cle="btn_fix", type="secondary", disabled=not actuel)


# ============================================================
# Tableau de bord (refonte 1c) : en-tête, tuiles, étape en cours
# ============================================================

_COULEUR = {"done": "#2f9567", "run": "#c27a12", "todo": "#4f46c8", "next": "#1c1b19",
            "dispo": "#1c1b19", "locked": "#b9b5ac"}


def _entete(projet: Path) -> None:
    from lib.project import dlc_config_path as _dlc

    n = len(list_sessions(projet))
    modele = Path(_dlc(projet)).parent.name if _dlc(projet) else "pas de modèle DLC"
    genre = ("N animaux par vidéo (arènes)" if project_kind(projet) == "multi"
             else "1 animal par vidéo")
    st.markdown(f'<style>.st-key-btn_ouvrir_dossier button {{ --ef-icone-dossier: '
                f'url("{lucide_data_uri("folder-open", "1c1b19")}"); }}</style>',
                unsafe_allow_html=True)
    # Nom du projet (ouvre la gestion des projets, maquette 1c : nom +
    # chevron) puis, juste à sa droite, le bouton « ouvrir le dossier ».
    with st.container(horizontal=True, vertical_alignment="center", gap="small"):
        with st.popover(projet.name, type="tertiary", key="ef_projet_menu"):
            if st.button("Tous les projets →", key="btn_tous_projets", width="stretch"):
                st.session_state.current_page = "Projets"
                st.rerun()
            st.markdown('<p class="ef-section">Modèle DLC</p>', unsafe_allow_html=True)
            _section_modele_dlc(projet)
        # Libellé réduit à une espace : l'icône est posée en `::before` par
        # le CSS. Le dossier s'ouvre sur la machine qui héberge le serveur —
        # l'aide le dit, pour qui consulte l'app à distance.
        if st.button(" ", key="btn_ouvrir_dossier",
                     help=f"Ouvrir {projet} dans le {reveal.nom_explorateur()} "
                          "(sur la machine qui fait tourner l'app)"):
            ok, message = reveal.ouvrir_dans_explorateur(projet)
            (st.toast if ok else st.error)(message)
    st.markdown(f'<p class="ef-muted" style="margin:-6px 0 4px">{genre} · '
                f'{n} session{"s" if n > 1 else ""} · {escape(modele)}</p>',
                unsafe_allow_html=True)


def _tuile_html(ligne: dict) -> str:
    e, etat = ligne["etape"], ligne["etat"]
    couleur = _COULEUR[etat]
    pct = 100 * ligne["fait"] / ligne["total"] if ligne["total"] else (100 if etat == "done" else 0)
    statut_couleur = couleur if etat in ("done", "run", "todo") else "#8a867e"
    return (
        f'<div class="ef-tuile ef-tuile-{etat}">'
        f'<div class="ef-tuile-haut"><img src="{lucide_data_uri(e.icone, couleur[1:])}" '
        f'width="17" height="17"><span>{e.n:02d}</span></div>'
        f'<b>{escape(e.nom)}</b>'
        f'<div class="ef-tuile-barre"><div style="width:{pct:.0f}%;background:{couleur}"></div></div>'
        f'<span class="ef-tuile-statut" style="color:{statut_couleur}">{escape(ligne["statut"])}</span>'
        "</div>")


def _tuiles(lignes: list[dict]) -> None:
    with st.container(key="tuiles"):
        _rangee_tuiles(lignes)


def _rangee_tuiles(lignes: list[dict]) -> None:
    for col, ligne in zip(st.columns(len(lignes), gap="small"), lignes):
        e = ligne["etape"]
        with col, st.container(key=f"tuile_{e.n}"):
            st.markdown(_tuile_html(ligne), unsafe_allow_html=True)
            # Bouton invisible étendu sur toute la tuile : elle est cliquable.
            if st.button(e.nom, key=f"tuile_btn_{e.n}", help=f"Ouvrir : {e.nom}"):
                st.session_state.current_page = e.nom
                st.rerun()


def _sessions_de_l_etape(projet: Path, ligne: dict) -> list[dict]:
    """Une entrée par session : `{id, etat, note}` pour l'étape `ligne`."""
    df = list_sessions(projet)
    prog = {}
    if ligne["etat"] == "run":
        job = runner.current(projet)
        prog = (runner.progress(projet, job.job_id) or {}) if job else {}
    sorties = []
    for sid, ok in zip(df["session_id"], df[ligne["etape"].colonne]):
        if ok == "OK":
            sorties.append({"id": sid, "etat": "done", "note": "terminé"})
        elif prog.get("session") == sid:
            note = f"{prog['pct']} %" if prog.get("pct") is not None else "en cours"
            sorties.append({"id": sid, "etat": "run", "note": note})
        else:
            sorties.append({"id": sid, "etat": "todo",
                            "note": "en attente" if ligne["etat"] == "run" else "à faire"})
    return sorties


def _carte_etape(projet: Path, lignes: list[dict]) -> None:
    from lib import etapes as E

    ligne = E.courante(lignes)
    if ligne is None:
        suite = E.ensuite(lignes)
        if suite is None:
            return
        ligne = next(x for x in lignes if x["etape"] is suite["etape"])
    e, en_cours = ligne["etape"], ligne["etat"] == "run"
    etiquette = (f'<p class="ef-label" style="color:#c27a12">Étape en cours · {e.n:02d}</p>'
                 if en_cours else
                 f'<p class="ef-label" style="color:#4f46c8">Prochaine étape · {e.n:02d}</p>')
    titre = e.titre + (f" — {ligne['fait']} vidéo{'s' if ligne['fait'] > 1 else ''} "
                       f"sur {ligne['total']}" if e.colonne and ligne["total"] else "")
    phrase = ("La suite s'ouvrira quand toutes les vidéos seront passées. Rien à faire d'ici là."
              if en_cours else e.description)

    with st.container(key="carte_etape"):
        haut, lien = st.columns([5, 1], vertical_alignment="center")
        haut.markdown(f'{etiquette}<p class="ef-carte-titre">{escape(titre)}</p>'
                      f'<p class="ef-muted" style="margin:0">{escape(phrase)}</p>',
                      unsafe_allow_html=True)
        # Bouton calé sur le bord droit de la carte.
        with lien, st.container(horizontal_alignment="right"):
            if st.button("Ouvrir l'étape →", key="btn_ouvrir_etape",
                         type="secondary" if en_cours else "primary"):
                st.session_state.current_page = e.nom
                st.rerun()
        if not (e.colonne and ligne["total"]):
            return
        sessions = _sessions_de_l_etape(projet, ligne)
        segments = "".join(
            f'<div><div class="ef-segment" style="background:{_COULEUR_SESSION[s["etat"]]}"></div>'
            f'<span style="color:{"#1c1b19" if s["etat"] != "todo" else "#8a867e"}">'
            f'{escape(s["id"])}</span></div>' for s in sessions)
        liste = "".join(
            f'<div class="ef-session"><span class="ef-dot" style="background:{_COULEUR_SESSION[s["etat"]]}">'
            f'</span><span class="ef-mono" style="color:{"#1c1b19" if s["etat"] != "todo" else "#8a867e"}">'
            f'{escape(s["id"])}</span><span class="ef-muted">{s["note"]}</span></div>' for s in sessions)
        st.markdown(f'<div class="ef-frise" style="grid-template-columns:repeat({min(len(sessions), 20)},1fr)">'
                    f'{segments}</div><div class="ef-sessions">{liste}</div>',
                    unsafe_allow_html=True)


_COULEUR_SESSION = {"done": "#2f9567", "run": "#c27a12", "todo": "#e4e2db"}


# ============================================================
# Entrée
# ============================================================

def render() -> None:
    projet = current_project()
    if projet is None:   # garde-fou : app.py n'y mène pas sans projet
        st.session_state.current_page = "Projets"
        st.rerun()
    _entete(projet)
    # Pendant un job, tuiles et frise suivent l'avancement sans recharger la page.
    (_corps_en_direct if runner.is_running(projet) else _corps)(projet)


def _corps(projet: Path) -> None:
    from lib import etapes as E

    job = runner.current(projet) if runner.is_running(projet) else None
    lignes = E.etat_pipeline(projet, job.script if job else None)
    _tuiles(lignes)
    _carte_etape(projet, lignes)


_corps_en_direct = st.fragment(run_every=3)(_corps)
