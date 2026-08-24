"""Page Projet — ouvrir/créer un projet, désigner le modèle DLC, avancement.

Trois responsabilités, aucune logique dupliquée : la création et le
diagnostic DLC passent par `lib.pipeline` (constructeurs de commandes) et
`views._job` (exécution + affichage du job), jamais par un appel direct.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import streamlit as st

import lib.pipeline as PL
from lib.config import (
    current_project,
    dlc_config_path,
    list_dlc_models,
    list_projects,
    models_root,
    project_kind,
    projects_root,
    set_current_project,
)
from lib.icons import lucide_title
from lib.sessions import list_sessions
from views import _job


# ============================================================
# Ouvrir un projet
# ============================================================

def _section_ouverture() -> None:
    st.markdown(lucide_title("folder-open", "Ouvrir un projet"), unsafe_allow_html=True)

    existants = list_projects(projects_root())
    courant = current_project()

    if not existants:
        st.caption(f"Aucun projet dans `{projects_root()}`.")
        return

    noms = [p.name for p in existants]
    index_defaut = 0
    if courant and courant.name in noms:
        index_defaut = noms.index(courant.name)

    nom_choisi = st.selectbox(
        "Sélectionner un projet",
        options=noms,
        index=index_defaut,
        key="projet_selector",
        label_visibility="collapsed",
    )
    chemin_choisi = projects_root() / nom_choisi

    if courant is None or courant.name != nom_choisi:
        set_current_project(chemin_choisi)
        st.rerun()

    if st.button("Supprimer ce projet", key="btn_supprimer_projet"):
        st.session_state["_confirmer_suppression"] = nom_choisi

    if st.session_state.get("_confirmer_suppression"):
        a_supprimer = st.session_state["_confirmer_suppression"]
        st.warning(
            f"Supprimer le projet **{a_supprimer}** et toutes ses données ? "
            "Cette action est irréversible."
        )
        col_annuler, col_confirmer = st.columns(2)
        with col_annuler:
            if st.button("Annuler", key="btn_annuler_suppression"):
                st.session_state.pop("_confirmer_suppression", None)
                st.rerun()
        with col_confirmer:
            if st.button("Oui, supprimer", key="btn_confirmer_suppression", type="primary"):
                chemin_a_supprimer = projects_root() / a_supprimer
                if chemin_a_supprimer.exists():
                    shutil.rmtree(chemin_a_supprimer)
                if courant and courant.name == a_supprimer:
                    set_current_project(None)
                st.session_state.pop("_confirmer_suppression", None)
                st.toast(f"Projet « {a_supprimer} » supprimé")
                st.rerun()


# ============================================================
# Créer un projet
# ============================================================

def _section_creation() -> None:
    st.subheader("Créer un projet")
    nom = st.text_input("Nom du projet", placeholder="ex : bottomview-MCC-2026-06")
    kind = st.radio(
        "Nombre d'animaux par vidéo",
        ["single", "multi"],
        format_func=lambda k: {
            "single": "1 animal par vidéo (1 vidéo = 1 session)",
            "multi": "N animaux dans N arènes séparées (1 vidéo = N sessions)",
        }[k],
        help="Choisis selon le nombre d'animaux, pas selon l'angle caméra. "
             "'multi' active le split par arène et écrit des coordonnées par défaut.",
    )
    modeles = list_dlc_models(models_root())
    choix = st.selectbox(
        "Modèle DLC (optionnel)",
        options=["(choisir plus tard)"] + [str(m / "config.yaml") for m in modeles],
        help="Le modèle reste où il est, il n'est jamais copié dans le projet. "
             "Tu peux le désigner plus tard.",
    )
    cible = projects_root() / nom.strip().replace(" ", "-") if nom.strip() else None
    if cible is None:
        return

    # `cible` peut exister sur disque pour deux raisons bien différentes :
    # un projet du même nom existait déjà avant qu'on clique (à refuser), ou
    # c'est le nôtre, tout juste créé par le job lancé plus bas (à ouvrir).
    # `_creation_en_cours` distingue les deux : posé juste avant de lancer
    # le job, il n'est vrai que pour LA cible qu'on vient de soumettre.
    #
    # « Créé » se juge par présence dans list_projects() (dossier `data/`
    # présent), pas par le contenu de pipeline_config.yaml : pour un projet
    # `single` sans modèle DLC, ce fichier est légitimement `{}` — un dict
    # vide est toujours faux en Python, donc un test sur son contenu ne
    # verrait jamais ce cas comme un succès.
    notre_creation = st.session_state.get("_creation_en_cours") == str(cible)
    deja_creee = cible in list_projects(projects_root())

    if deja_creee and notre_creation:
        if current_project() != cible:
            # Rerun immédiat pour que la sélection prenne effet avant
            # d'afficher le succès — sinon le sélecteur de gauche, déjà
            # rendu ce tour-ci, resterait sur l'ancien projet jusqu'au
            # prochain rafraîchissement.
            set_current_project(cible)
            st.rerun()
        st.session_state.pop("_creation_en_cours", None)
        st.success(f"Projet **{cible.name}** créé avec succès.")
    elif cible.exists():
        st.error(f"`{cible}` existe déjà.")
    else:
        cmd = PL.create_project(
            cible, kind=kind,
            dlc_config=None if choix.startswith("(") else choix,
        )
        st.session_state["_creation_en_cours"] = str(cible)
        # Le projet n'existe pas encore : pas de dossier <projet>/.ethoflow/
        # où loger le job. On le fait tourner dans la racine des projets à
        # la place (ruling P4) ; un .ethoflow/ y apparaît, ce qui est
        # attendu, pas un bug à corriger.
        _job.bouton_lancer(cible.parent, "Créer le projet", cmd, cle="btn_creer")

    _job.panneau(cible.parent)


# ============================================================
# Modèle DLC
# ============================================================

def _section_modele_dlc(projet: Path) -> None:
    st.subheader("Modèle DLC")
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
            "Aucun modèle configuré. `run_dlc_inference --mode custom` en a besoin."
        )

    modeles = list_dlc_models(models_root())
    options = [str(m / "config.yaml") for m in modeles]
    choisi = st.selectbox("Modèles trouvés", options=options) if options else None
    libre = st.text_input("…ou un chemin de config.yaml", value="")
    chemin = libre.strip() or choisi
    if chemin:
        _job.bouton_lancer(
            projet, "Utiliser ce modèle",
            PL.create_project(projet, kind=project_kind(projet),
                              dlc_config=chemin, force=True),
            cle="btn_modele",
        )

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
# Avancement
# ============================================================

def _section_sessions(projet: Path) -> None:
    st.markdown(lucide_title("layout-dashboard", "Sessions"), unsafe_allow_html=True)
    st.caption("Statut d'avancement du pipeline")

    df = list_sessions(projet)
    if df.empty:
        st.info(
            "Aucune session dans ce projet. Va dans **Données** "
            "pour importer depuis Excel ou créer des metadata."
        )
        return

    if project_kind(projet) != "multi":
        df = df.drop(columns=["split"])

    etapes = list(df.columns[1:])  # tout sauf session_id
    cols = st.columns(len(etapes) + 1)
    cols[0].metric("Sessions", len(df))
    for i, colonne in enumerate(etapes, start=1):
        cols[i].metric(colonne, int((df[colonne] == "OK").sum()))

    st.dataframe(df, use_container_width=True, hide_index=True)


# ============================================================
# Entrée
# ============================================================

def render() -> None:
    st.title("Projet")

    col_ouvrir, col_creer = st.columns(2)
    with col_ouvrir:
        _section_ouverture()
    with col_creer:
        _section_creation()

    projet = current_project()
    if projet is None:
        return

    st.divider()
    _job.panneau(projet)
    _job.historique(projet)

    st.divider()
    _section_modele_dlc(projet)

    st.divider()
    _section_sessions(projet)
