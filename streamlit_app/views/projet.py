"""Page Projet — ouvrir/créer un projet, désigner le modèle DLC, avancement.

Trois responsabilités, aucune logique dupliquée : la création et le
diagnostic DLC passent par `lib.pipeline` (constructeurs de commandes) et
`views._job` (exécution + affichage du job), jamais par un appel direct.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

import streamlit as st

import lib.pipeline as PL
from lib import reveal, runner
from lib.config import (
    current_project,
    dlc_config_path,
    est_projet,
    list_dlc_models,
    list_projects,
    models_root,
    project_kind,
    projects_root,
    recent_roots,
    set_current_project,
    set_dlc_config,
    set_projects_root,
)
from lib.icons import lucide_title
from lib.sessions import list_sessions
from views import _job
from views._widgets import champ_chemin


# ============================================================
# Ouvrir un projet
# ============================================================

def _section_emplacement() -> Path:
    """Champ d'emplacement + raccourcis récents. Retourne la racine effective.

    Le champ accepte indifféremment une **racine** (un dossier qui contient
    des projets) ou un **projet** directement, reconnu à son sous-dossier
    `data/` : un projet sur disque externe n'a aucune raison de vivre sous
    la racine configurée. Coller un chemin de projet l'ouvre et retient son
    dossier parent comme racine, pour que la liste montre ses voisins.
    """
    st.markdown(lucide_title("folder", "Emplacement"), unsafe_allow_html=True)

    # Le champ est piloté par son `key` seul (pas de `value=`), sinon
    # Streamlit réimposerait la valeur à chaque passage et écraserait la
    # saisie. Corollaire : on ne peut plus écrire dans cette clé une fois le
    # widget instancié — Streamlit lève `StreamlitAPIException`. Les
    # bascules (bouton « récent », projet collé) déposent donc leur valeur
    # dans une clé tampon, appliquée ici au run suivant, avant le widget.
    if "_emplacement_a_appliquer" in st.session_state:
        st.session_state["emplacement_saisi"] = st.session_state.pop(
            "_emplacement_a_appliquer"
        )

    if "emplacement_saisi" not in st.session_state:
        st.session_state["emplacement_saisi"] = str(projects_root())

    col_champ, col_parcourir = st.columns([1, 0.28], vertical_alignment="bottom")
    with col_champ:
        st.text_input(
            "Dossier des projets",
            key="emplacement_saisi",
            label_visibility="collapsed",
            help="Un dossier qui contient tes projets, ou le chemin d'un "
                 "projet précis. Le choix est mémorisé et retrouvé au "
                 "prochain démarrage.",
        )
    with col_parcourir:
        # La fenêtre s'ouvre sur la machine qui héberge le serveur, et le
        # run attend qu'on ait choisi. En local — le cas normal — c'est
        # transparent ; l'aide le précise pour l'usage à distance.
        if st.button(
            "Parcourir…",
            key="btn_parcourir",
            help=f"Ouvre un sélecteur de dossier ({reveal.nom_explorateur()}) "
                 "sur la machine qui fait tourner l'app.",
            width="stretch",
        ):
            choisi, message = reveal.choisir_dossier(
                titre="Dossier des projets EthoFlow",
                depart=Path(st.session_state["emplacement_saisi"]).expanduser(),
            )
            if choisi is not None:
                # Même détour par la clé tampon que les boutons « récents » :
                # écrire dans la clé d'un widget déjà instancié lève.
                st.session_state["_emplacement_a_appliquer"] = str(choisi)
                st.rerun()
            elif message:
                # Une annulation renvoie un message vide : ne rien afficher.
                st.warning(message)

    recents = [p for p in recent_roots() if str(p) != st.session_state["emplacement_saisi"]]
    if recents:
        st.caption("Emplacements récents")
        colonnes = st.columns(min(len(recents), 4))
        for i, chemin_recent in enumerate(recents[:4]):
            if colonnes[i].button(chemin_recent.name or str(chemin_recent),
                                  key=f"recent_{i}",
                                  help=str(chemin_recent),
                                  width="stretch"):
                st.session_state["_emplacement_a_appliquer"] = str(chemin_recent)
                st.rerun()

    saisi = st.session_state["emplacement_saisi"].strip()
    if not saisi:
        st.warning("Indique un emplacement.")
        return projects_root()

    chemin = Path(saisi).expanduser()

    # Un chemin relatif se résout par rapport au dossier de lancement de
    # l'app, donc ailleurs à chaque démarrage. Le cas classique est un
    # `D:\EthoFlow\projects` saisi sous macOS ou Linux : pathlib y voit un
    # unique composant portant littéralement ce nom, et le dossier créé
    # apparaît sous un nom absurde dans le Finder.
    if not chemin.is_absolute():
        st.error(
            f"`{chemin}` n'est pas un chemin absolu. "
            "Donne un chemin complet — par exemple "
            f"`{Path.home() / 'EthoFlow' / 'projects'}`"
            + (" sous macOS/Linux, ou `D:\\EthoFlow\\projects` sous Windows."
               if os.name != "nt" else ".")
        )
        return projects_root()

    if est_projet(chemin):
        # Un projet a été collé directement : on l'ouvre et on liste ses
        # voisins. Le champ garde ce que l'utilisateur a tapé — le réécrire
        # sur le dossier parent obligerait à un aller-retour par la clé
        # tampon, pour un gain nul : la liste dessous montre déjà les
        # voisins, et voir sa propre saisie est plus lisible qu'un chemin
        # substitué sous les doigts.
        racine = chemin.parent
        if current_project() != chemin:
            set_current_project(chemin)
        if racine != projects_root():
            set_projects_root(racine)
        st.caption(f"Projet ouvert directement — voisins dans `{racine}`.")
        return racine

    if not chemin.is_dir():
        st.error(f"`{chemin}` n'existe pas.")
        return projects_root()

    if chemin != projects_root():
        set_projects_root(chemin)

    n = len(list_projects(chemin))
    if n:
        st.caption(f"{n} projet(s) à cet emplacement.")
    else:
        st.caption("Aucun projet ici — tu peux en créer un ci-dessous.")
    return chemin


def _section_ouverture(racine: Path) -> None:
    st.markdown(lucide_title("folder-open", "Ouvrir un projet"), unsafe_allow_html=True)

    existants = list_projects(racine)
    courant = current_project()

    if not existants:
        st.caption(f"Aucun projet dans `{racine}`.")
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
    chemin_choisi = racine / nom_choisi

    if courant is None or courant.name != nom_choisi:
        set_current_project(chemin_choisi)
        st.rerun()

    # Un job (inférence DLC, entraînement VAME...) peut écrire dans l'arbre
    # de CE projet précis (ruling R10.4) : supprimer pendant qu'il tourne
    # corromprait ses sorties en plus des données déjà présentes. Le
    # verrou est par projet (comme pour bouton_lancer), donc on vérifie
    # `chemin_choisi`, pas la racine des projets.
    occupe = runner.is_running(chemin_choisi)
    aide_suppression = None
    if occupe:
        en_cours = runner.current(chemin_choisi)
        if en_cours is not None:
            aide_suppression = (
                f"« {en_cours.label} » tourne déjà (démarré à {en_cours.started_at}) "
                "— attends la fin avant de supprimer."
            )
        else:
            aide_suppression = "Un job tourne déjà sur ce projet — attends la fin avant de supprimer."

    if st.button("Supprimer ce projet", key="btn_supprimer_projet",
                 disabled=occupe, help=aide_suppression):
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
                chemin_a_supprimer = racine / a_supprimer
                # Recontrôle juste avant le rmtree : un job a pu démarrer
                # entre l'affichage du bouton et ce clic (même course que
                # JobBusy dans bouton_lancer).
                if runner.is_running(chemin_a_supprimer):
                    st.error(
                        "Un job a démarré entre-temps sur ce projet — "
                        "suppression annulée."
                    )
                else:
                    if chemin_a_supprimer.exists():
                        shutil.rmtree(chemin_a_supprimer)
                    if courant and courant.name == a_supprimer:
                        set_current_project(None)
                    st.toast(f"Projet « {a_supprimer} » supprimé")
                st.session_state.pop("_confirmer_suppression", None)
                st.rerun()


# ============================================================
# Créer un projet
# ============================================================

def _section_creation(racine: Path) -> None:
    st.subheader("Créer un projet")
    # Clés explicites : sans elles, Streamlit dérive la clé du widget de sa
    # position dans le script. `_section_ouverture()`, rendue juste avant,
    # dessine un nombre de widgets différent selon qu'il existe ou non des
    # projets à lister — dès que ce nombre change (typiquement : le tout
    # premier projet vient d'apparaître), la clé implicite de "Nom du
    # projet" change avec, et Streamlit lui redonne sa valeur par défaut
    # (vide) au lieu de garder ce que l'utilisateur venait de taper. Avec
    # une clé stable, ce champ garde sa valeur quoi qu'il se passe ailleurs
    # sur la page.
    nom = st.text_input("Nom du projet", placeholder="ex : bottomview-MCC-2026-06",
                        key="creation_nom")
    kind = st.radio(
        "Nombre d'animaux par vidéo",
        ["single", "multi"],
        format_func=lambda k: {
            "single": "1 animal par vidéo (1 vidéo = 1 session)",
            "multi": "N animaux dans N arènes séparées (1 vidéo = N sessions)",
        }[k],
        help="Choisis selon le nombre d'animaux, pas selon l'angle caméra. "
             "'multi' active le split par arène et écrit des coordonnées par défaut.",
        key="creation_kind",
    )
    modeles = list_dlc_models(models_root())
    choix = st.selectbox(
        "Modèle DLC (optionnel)",
        options=["(choisir plus tard)"] + [str(m / "config.yaml") for m in modeles],
        help="Le modèle reste où il est, il n'est jamais copié dans le projet. "
             "Tu peux le désigner plus tard.",
        key="creation_modele",
    )
    cible = racine / nom.strip().replace(" ", "-") if nom.strip() else None
    if cible is None:
        return

    # Le succès ou l'échec se lisent sur l'ÉTAT DU JOB (runner.current),
    # jamais sur le système de fichiers (ruling R10.3) : create_project.py
    # crée data/ avant d'écrire pipeline_config.yaml, donc un test sur la
    # présence de data/ affiche « créé avec succès » même quand le script a
    # planté juste après (ex : configs/ pas accessible en écriture) alors
    # que le panneau de job juste en dessous dit « ❌ Échec ».
    #
    # `_creation_en_cours` distingue « ce job concerne CETTE cible, on
    # vient de le lancer » de « un job antérieur (autre projet, ou même nom
    # supprimé puis retapé) traîne encore dans l'historique de la racine » :
    # posé juste avant de lancer le job, il n'est vrai que pour la cible
    # qu'on vient de soumettre.
    #
    # Il est effacé dès qu'un état TERMINAL (succeeded/failed/cancelled/
    # interrupted) a été affiché une fois (ruling R10.6a) — sinon, en
    # créant X, en le supprimant via le flux guardé de `_section_ouverture`,
    # puis en retapant le même nom X, ce job resterait « le nôtre » pour
    # toujours : la bannière « créé avec succès » revient sur un projet qui
    # n'existe plus, et le bouton « Créer le projet » disparaît pour de bon
    # (branché sur `succeeded` => `afficher_bouton = False`) — la cible ne
    # peut alors plus jamais être recréée pour ce nom. `running` reste seul
    # à ne PAS effacer le drapeau : le job n'est pas fini, il faut encore
    # le reconnaître comme nôtre au prochain rendu.
    job = runner.current(cible.parent)
    notre_job = job if (job is not None and
                        st.session_state.get("_creation_en_cours") == str(cible)) else None

    afficher_bouton = True
    if notre_job is not None and notre_job.state == "succeeded":
        afficher_bouton = False
        if current_project() != cible:
            # Rerun immédiat pour que la sélection prenne effet avant
            # d'afficher le succès — sinon le sélecteur de gauche, déjà
            # rendu ce tour-ci, resterait sur l'ancien projet jusqu'au
            # prochain rafraîchissement. Le drapeau n'est PAS encore
            # effacé ici : ce rerun doit encore reconnaître ce job comme
            # nôtre pour afficher le succès une fois arrivé à destination.
            set_current_project(cible)
            st.rerun()
        st.session_state.pop("_creation_en_cours", None)
        st.success(f"Projet **{cible.name}** créé avec succès.")
    elif notre_job is not None and notre_job.state == "failed":
        st.session_state.pop("_creation_en_cours", None)
        st.error(
            f"Échec de la création de `{cible.name}` "
            f"(code de retour {notre_job.returncode}). Voir le log ci-dessous."
        )
    elif notre_job is not None and notre_job.state in ("cancelled", "interrupted"):
        st.session_state.pop("_creation_en_cours", None)
        mot = "annulée" if notre_job.state == "cancelled" else "interrompue"
        st.warning(f"Création de `{cible.name}` {mot}.")
    elif notre_job is not None and notre_job.state == "running":
        pass  # rien à afficher en plus du panneau de job ci-dessous
    elif cible.exists():
        st.error(f"`{cible}` existe déjà.")
        afficher_bouton = False

    if afficher_bouton:
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
    choisi = (st.selectbox("Modèles trouvés", options=options, key="modele_dlc_trouve")
              if options else None)
    libre = champ_chemin(
        "…ou un chemin", cle="modele_dlc_libre",
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
        if st.button("Utiliser ce modèle", key="btn_modele", type="primary"):
            set_dlc_config(projet, chemin)
            st.toast(f"Modèle DLC configuré : {chemin}")
            st.rerun()

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

    st.dataframe(df, width="stretch", hide_index=True)


# ============================================================
# Entrée
# ============================================================

def render() -> None:
    st.title("Projet")

    racine = _section_emplacement()

    st.divider()
    col_ouvrir, col_creer = st.columns(2)
    with col_ouvrir:
        _section_ouverture(racine)
    with col_creer:
        _section_creation(racine)

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
