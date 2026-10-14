"""Widgets partagés entre plusieurs pages d'étape.

Contrairement à `views/_job.py` (affichage de job), ce module regroupe des
éléments de formulaire réutilisés tels quels par plusieurs pages — rien de
spécifique à un script du pipeline ne doit y entrer.
"""
from __future__ import annotations

from pathlib import Path

import streamlit as st

from lib import reveal, sessions


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
    # Le multiselect désactivé garde son ancienne valeur : sans ce garde,
    # un appelant qui ne transmet que `choisies` (prepare_vame_input n'a
    # pas de --all) lancerait la sélection périmée au lieu de tout.
    return ([] if tout else choisies), tout


def champ_chemin(
    label: str,
    *,
    cle: str,
    valeur_defaut: str = "",
    mode: str = "dossier",
    extensions: list[str] | None = None,
    titre_dialogue: str | None = None,
    depart: Path | None = None,
    help: str | None = None,
    label_visibility: str = "visible",
    placeholder: str | None = None,
) -> str:
    """Champ de saisie de chemin, doublé d'un bouton « Parcourir… ».

    Retourne la valeur courante du champ. `mode` vaut « dossier » ou
    « fichier » ; `extensions` (« .yaml », « .xlsx »…) filtre l'affichage du
    sélecteur en mode fichier.

    Le sélecteur natif s'ouvre sur la machine qui héberge le serveur — en
    local, le cas normal, c'est celle de l'utilisateur. L'aide du bouton le
    rappelle pour l'usage à distance.

    Détail Streamlit non négociable : on ne peut pas écrire dans la clé d'un
    widget déjà instancié. Le chemin choisi transite donc par une clé
    tampon, appliquée au run suivant AVANT que le champ ne soit créé.
    """
    cle_tampon = f"_{cle}_a_appliquer"
    if cle_tampon in st.session_state:
        st.session_state[cle] = st.session_state.pop(cle_tampon)
    if cle not in st.session_state:
        st.session_state[cle] = valeur_defaut

    col_champ, col_bouton = st.columns([1, 0.28], vertical_alignment="bottom")
    with col_champ:
        st.text_input(
            label, key=cle, help=help,
            label_visibility=label_visibility,
            placeholder=placeholder,
        )
    with col_bouton:
        if st.button(
            "Parcourir…",
            key=f"btn_parcourir_{cle}",
            help=f"Ouvre un sélecteur ({reveal.nom_explorateur()}) sur la "
                 "machine qui fait tourner l'app.",
            width="stretch",
        ):
            actuel = Path(st.session_state[cle]).expanduser() if st.session_state[cle] else None
            point_depart = depart or (
                actuel if actuel and actuel.is_dir()
                else (actuel.parent if actuel and actuel.parent.is_dir() else None)
            )
            if mode == "fichier":
                choisi, message = reveal.choisir_fichier(
                    titre=titre_dialogue or f"Choisir : {label}",
                    depart=point_depart, extensions=extensions,
                )
            else:
                choisi, message = reveal.choisir_dossier(
                    titre=titre_dialogue or f"Choisir : {label}",
                    depart=point_depart,
                )
            if choisi is not None:
                st.session_state[cle_tampon] = str(choisi)
                st.rerun()
            elif message:
                # Une annulation renvoie un message vide : ne rien afficher.
                st.warning(message)

    return st.session_state[cle]


# ============================================================
# Pages d'étape (refonte 1c) : en-tête et verrou
# ============================================================

def entete_etape(projet: Path) -> None:
    """« PROJET · ÉTAPE n / 9 », titre en clair, ce que fait l'étape.

    L'étape est celle de la page courante (`lib.etapes.PAR_PAGE`) : le
    titre et la phrase sont définis une seule fois, au même endroit que
    l'état du pipeline.
    """
    from html import escape

    from lib.etapes import ETAPES, PAR_PAGE

    etape = PAR_PAGE[st.session_state.get("current_page", "")]
    st.markdown(
        f'<p class="ef-label">{escape(projet.name)} · Étape {etape.n} / {len(ETAPES)}</p>'
        f'<h1 class="ef-titre-etape">{escape(etape.titre)}</h1>'
        f'<p class="ef-description-etape">{escape(etape.description)}</p>',
        unsafe_allow_html=True,
    )


def verrou_etape(page: str | None = None) -> str | None:
    """Pourquoi les lancements de l'étape `page` (défaut : la page courante)
    sont bloqués, ou None.

    Règle métier : une étape s'ouvre quand la précédente est faite pour
    toutes les sessions. La page reste consultable ; seuls ses boutons de
    lancement sont désactivés, avec cette phrase à côté.
    """
    from lib import etapes as E

    lignes = st.session_state.get("_etat_pipeline") or []
    page = page or st.session_state.get("current_page")
    ligne = next((x for x in lignes if x["etape"].nom == page), None)
    if ligne is None or ligne["etat"] not in ("next", "locked"):
        return None
    # La première étape pas encore faite — pas une étape finie qu'on relance.
    cour = next((x for x in lignes if x["etat"] == "todo"), None) or E.courante(lignes)
    if cour is None:
        return None
    return (f"Disponible quand l'étape « {cour['etape'].nom} » sera faite pour "
            f"toutes les sessions ({cour['fait']}/{cour['total']} aujourd'hui)."
            if cour["total"] else
            f"Disponible quand l'étape « {cour['etape'].nom} » sera faite.")
