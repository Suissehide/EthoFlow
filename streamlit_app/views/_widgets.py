"""Widgets partagés entre plusieurs pages d'étape.

Contrairement à `views/_job.py` (affichage de job), ce module regroupe des
éléments de formulaire réutilisés tels quels par plusieurs pages — rien de
spécifique à un script du pipeline ne doit y entrer.
"""
from __future__ import annotations

from pathlib import Path

import streamlit as st

from lib import reveal, sessions


def selecteur_sessions(projet: Path, *, cle: str,
                       colonne: str | None = None) -> tuple[list[str], bool]:
    """Puces « Toutes · N », « Non traitées · k », « Choisir… » (maquette).

    Retourne (sessions choisies, drapeau « toutes »), comme avant : chaque
    script accepte soit `--all`, soit une liste de `session_id` — jamais les
    deux (voir `lib.pipeline`). `colonne` : colonne de `list_sessions` qui dit
    si l'étape est faite pour une session ; sans elle, pas de puce « Non
    traitées ».
    """
    df = sessions.list_sessions(projet)
    ids = list(df["session_id"])
    restantes = list(df.loc[df[colonne] != "OK", "session_id"]) if colonne else []
    libelles = {"tout": f"Toutes · {len(ids)}", "reste": f"Non traitées · {len(restantes)}",
                "choix": "Choisir…"}
    options = ["tout", "reste", "choix"] if colonne else ["tout", "choix"]
    st.markdown('<p class="ef-section">Sessions</p>', unsafe_allow_html=True)
    mode = st.pills("Sessions", options, default="tout", required=True,
                    format_func=libelles.get, key=f"{cle}_portee",
                    label_visibility="collapsed")
    if mode == "reste":
        return restantes, False
    if mode == "choix":
        return st.multiselect("Sessions à traiter", options=ids, key=f"{cle}_sel",
                              placeholder="Choisis une ou plusieurs sessions",
                              label_visibility="collapsed"), False
    return [], True


def nb_sessions(projet: Path, choisies: list[str], tout: bool, *,
                colonne: str | None = None, ignorer_faites: bool = False) -> int:
    """Combien de sessions un job traitera réellement (libellé « Lancer sur N »).

    `ignorer_faites` : le script saute celles où `colonne` est déjà « OK ».
    """
    df = sessions.list_sessions(projet)
    cible = df if tout else df[df["session_id"].isin(choisies)]
    if ignorer_faites and colonne:
        cible = cible[cible[colonne] != "OK"]
    return len(cible)


def libelle_lancer(n: int) -> str:
    return f"Lancer sur {n} session{'s' if n > 1 else ''}"


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


def details_techniques(cmd=None, *, notes: list[str] | None = None) -> None:
    """Le jargon (script, env conda, ligne de commande) sort du premier
    niveau et se range ici, replié (handoff : « Détails techniques »)."""
    import shlex

    import lib.pipeline as PL

    with st.expander("Détails techniques"):
        if cmd is not None:
            st.markdown(f"Script `{cmd.script}` · environnement conda `{cmd.env}`")
            st.code(shlex.join(PL.to_argv(cmd)), language="bash", wrap_lines=True)
        for note in notes or []:
            st.markdown(note)


def choix_modele_dlc(projet: Path, *, cle: str) -> None:
    """Désigner le modèle DLC du projet : modèles trouvés, ou un chemin.

    Partagé par le menu du projet (Projet) et la page Pose, qui le propose
    sur place quand « Mon modèle entraîné » n'a pas de modèle valide.
    """
    from lib.config import list_dlc_models, models_root, set_dlc_config

    modeles = list_dlc_models(models_root())
    options = [str(m / "config.yaml") for m in modeles]
    choisi = (st.selectbox("Modèles trouvés", options=options, key=f"{cle}_dlc_trouve")
              if options else None)
    libre = champ_chemin(
        "…ou un chemin" if options else "Dossier du modèle", cle=f"{cle}_dlc_libre",
        mode="fichier", extensions=[".yaml", ".yml"],
        titre_dialogue="Choisir le config.yaml du modèle DLC",
        help="Le dossier du modèle suffit : son `config.yaml` est trouvé "
             "tout seul. Le fichier lui-même marche aussi.",
    )
    chemin = libre.strip() or choisi
    if chemin:
        # Écriture locale instantanée, pas un job : `set_dlc_config` ne
        # touche qu'à la clé `dlc_project_config` et préserve le reste
        # (`default_arenes_coords`, `px_per_cm`...). Passer par
        # `create_project.py --force` — ce que faisait la première version
        # — régénère pipeline_config.yaml en entier (arènes perdues) ET
        # regénère l'Excel de démarrage même déjà rempli par le chercheur
        # (ruling R10.2, Critical 1+2 de la revue).
        if st.button("Utiliser ce modèle", key=f"btn_{cle}", type="primary"):
            set_dlc_config(projet, chemin)
            st.toast(f"Modèle DLC configuré : {chemin}")
            st.rerun()
