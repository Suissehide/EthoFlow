"""Page Projets — écran d'arrivée : ouvrir ou créer un projet (handoff 2a/2b).

Colonne centrale : l'emplacement (Changer… / Récents), puis la liste des
projets avec leur avancement, ou un état vide. Panneau de droite :
« Nouveau projet », qui se replie en récapitulatif + carte de job pendant
la création.

Les règles de l'ancienne page Projet sont conservées telles quelles :
- un chemin de projet collé ouvre ce projet (`est_projet`) ;
- chemin relatif ou inexistant refusé, avec le même message ;
- création : l'issue se lit sur l'ÉTAT DU JOB (`runner.current`), jamais sur
  le système de fichiers (ruling R10.3), et le drapeau `_creation_en_cours`
  est effacé sur tout état terminal (ruling R10.6a) ;
- suppression : confirmation, bloquée pendant un job du projet (R10.4).
"""
from __future__ import annotations

import os
import shutil
from html import escape
from pathlib import Path

import streamlit as st

import lib.pipeline as PL
from lib import projets as PR
from lib import reveal, runner
from lib.config import (
    current_project,
    est_projet,
    list_dlc_models,
    list_projects,
    models_root,
    projects_root,
    recent_roots,
    set_current_project,
    set_projects_root,
)
from lib.icons import lucide_data_uri
from lib.project import load_prefs
from views import _job


def _ouvrir(chemin: Path) -> None:
    """Ouvre le projet et va sur son tableau de bord."""
    set_current_project(chemin)
    st.session_state.current_page = "Projet"
    st.rerun()


# ============================================================
# Emplacement
# ============================================================

def _barre_emplacement() -> Path | None:
    """Barre « Emplacement ». Retourne la racine effective, ou None si la
    saisie est invalide (le message est déjà affiché)."""
    # Le champ « Coller un chemin » est piloté par sa clé seule : une fois
    # instancié, on ne peut plus écrire dans sa clé. Les bascules (Parcourir,
    # Récents, projet collé) passent par cette clé tampon, appliquée ici,
    # avant le widget, au run suivant.
    if "_emplacement_a_appliquer" in st.session_state:
        st.session_state["emplacement_saisi"] = st.session_state.pop("_emplacement_a_appliquer")
    if "emplacement_saisi" not in st.session_state:
        st.session_state["emplacement_saisi"] = str(projects_root())

    saisi = st.session_state["emplacement_saisi"].strip()
    recents = [p for p in recent_roots() if str(p) != saisi][:4]
    dossier = lucide_data_uri("folder", "1c1b19")
    with st.container(key="barre_emplacement", horizontal=True, vertical_alignment="center"):
        st.markdown(
            f'<div class="ef-emplacement"><img src="{dossier}" width="16" height="16">'
            f'<div><p class="ef-label">Emplacement</p>'
            f'<p class="ef-chemin" title="{escape(saisi)}">{escape(saisi) or "—"}</p></div></div>',
            unsafe_allow_html=True)
        with st.popover("Changer…", key="ef_changer"):
            if st.button("Parcourir…", key="btn_parcourir", width="stretch",
                         help=f"Ouvre un sélecteur de dossier ({reveal.nom_explorateur()}) "
                              "sur la machine qui fait tourner l'app."):
                choisi, message = reveal.choisir_dossier(
                    titre="Dossier des projets EthoFlow",
                    depart=Path(saisi or str(projects_root())).expanduser())
                if choisi is not None:
                    st.session_state["_emplacement_a_appliquer"] = str(choisi)
                    st.rerun()
                elif message:
                    st.warning(message)   # une annulation renvoie un message vide
            st.text_input("Coller un chemin", key="emplacement_saisi",
                          help="Un dossier qui contient des projets, ou le chemin d'un "
                               "projet précis : il s'ouvre directement.")
        if recents:
            with st.popover("Récents", key="ef_recents"):
                for i, chemin in enumerate(recents):
                    if st.button(chemin.name or str(chemin), key=f"recent_{i}",
                                 help=str(chemin), width="stretch"):
                        st.session_state["_emplacement_a_appliquer"] = str(chemin)
                        st.rerun()

    if not saisi:
        st.warning("Indique un emplacement.")
        return None
    chemin = Path(saisi).expanduser()
    # Un chemin relatif se résout par rapport au dossier de lancement de
    # l'app, donc ailleurs à chaque démarrage (cas classique : `D:\...`
    # saisi sous macOS, vu comme un seul composant au nom absurde).
    if not chemin.is_absolute():
        st.error(
            f"`{chemin}` n'est pas un chemin absolu. Donne un chemin complet — par "
            f"exemple `{Path.home() / 'EthoFlow' / 'projects'}`"
            + (" sous macOS/Linux, ou `D:\\EthoFlow\\projects` sous Windows."
               if os.name != "nt" else "."))
        return None
    if est_projet(chemin):
        # Projet collé : il s'ouvre, et l'emplacement devient son dossier parent.
        st.session_state["_emplacement_a_appliquer"] = str(chemin.parent)
        if chemin.parent != projects_root():
            set_projects_root(chemin.parent)
        _ouvrir(chemin)
    if not chemin.is_dir():
        st.error(f"`{chemin}` n'existe pas.")
        return None
    if chemin != projects_root():
        # Rerun : la barre (et ses « Récents ») a été dessinée avec l'ancien
        # emplacement ; elle doit refléter le nouveau tout de suite.
        set_projects_root(chemin)
        st.rerun()
    return chemin


# ============================================================
# Liste des projets
# ============================================================

@st.cache_data(ttl=10, show_spinner=False)
def _apercus(racine: str, noms: tuple[str, ...]) -> list[dict]:
    # ponytail: cache de 10 s (comme l'avancement du tableau de bord) — la
    # liste relit chaque projet ; un job qui finit s'y voit avec ce délai.
    return PR.apercus(Path(racine), [Path(racine) / n for n in noms])


# Mêmes proportions pour l'en-tête et les lignes : colonnes alignées.
_COLONNES = [1.5, 0.95, 1.4, 0.75, 0.75]


def _frise(segments: list[str]) -> str:
    return '<div class="ef-mini-frise">' + "".join(
        f'<span style="background:{PR.COULEURS[s]}"></span>' for s in segments) + "</div>"


def _ligne_projet(i: int, a: dict, dernier: bool) -> None:
    sous = f"{a['sessions']} session{'s' if a['sessions'] != 1 else ''}"
    sous += f" · modèle {a['modele']}" if a["modele"] else " · pas de modèle DLC"
    tag = '<span class="ef-tag">dernier ouvert</span>' if dernier else ""
    with st.container(key=f"ligne_{'dernier_' if dernier else ''}{i}"):
        c_nom, c_type, c_avance, c_act, c_actions = st.columns(
            _COLONNES, vertical_alignment="center", gap="medium")
        c_nom.markdown(f'<p class="ef-projet-nom">{escape(a["nom"])}{tag}</p>'
                       f'<p class="ef-projet-sous">{escape(sous)}</p>', unsafe_allow_html=True)
        c_type.markdown(f'<p class="ef-projet-type">{a["type"]}</p>', unsafe_allow_html=True)
        c_avance.markdown(_frise(a["segments"]) + f'<p class="ef-projet-statut" '
                          f'style="color:{a["couleur"]}">{escape(a["statut"])}</p>',
                          unsafe_allow_html=True)
        c_act.markdown(f'<p class="ef-projet-date">{PR.date_relative(a["activite"])}</p>',
                       unsafe_allow_html=True)
        with c_actions, st.container(horizontal=True, horizontal_alignment="right",
                                     vertical_alignment="center", gap="small", key=f"actions_{i}"):
            if st.button("Ouvrir", key=f"ouvrir_{i}", type="primary" if dernier else "secondary"):
                _ouvrir(a["chemin"])
            with st.popover("⋯", key=f"menu_{i}"):
                if st.button(f"Afficher dans le {reveal.nom_explorateur()}", key=f"reveler_{i}",
                             width="stretch"):
                    ok, message = reveal.ouvrir_dans_explorateur(a["chemin"])
                    (st.toast if ok else st.error)(message)
                # Un job peut écrire dans l'arbre de CE projet : le supprimer
                # pendant qu'il tourne corromprait ses sorties (R10.4).
                occupe = runner.is_running(a["chemin"])
                aide = None
                if occupe:
                    job = runner.current(a["chemin"])
                    aide = (f"« {job.label} » tourne : attends la fin avant de supprimer."
                            if job else "Un job tourne sur ce projet.")
                if st.button("Supprimer…", key=f"supprimer_{i}", width="stretch",
                             disabled=occupe, help=aide):
                    st.session_state["_confirmer_suppression"] = a["nom"]
                    st.rerun()


def _liste(racine: Path) -> None:
    noms = tuple(p.name for p in list_projects(racine))
    # Un dossier en cours de création n'est pas encore un projet : aperçus vides.
    apercus = _apercus(str(racine), noms) if noms else []
    if not apercus:
        dossier = lucide_data_uri("folder", "8a867e")
        st.markdown(
            f'<div class="ef-vide"><img src="{dossier}" width="28" height="28">'
            "<b>Aucun projet à cet emplacement</b><span>Crée le premier à droite, ou "
            "choisis un autre dossier. Tu peux aussi coller le chemin d'un projet "
            "précis : il s'ouvre directement.</span></div>", unsafe_allow_html=True)
        return
    dernier = load_prefs().get("last_project")
    with st.container(key="liste_projets"):
        with st.container(key="entete_liste"):
            for col, titre in zip(st.columns(_COLONNES, gap="medium"),
                                  [f"Projet · {len(apercus)}", "Type", "Avancement", "Activité", ""]):
                col.markdown(f'<p class="ef-label">{titre}</p>', unsafe_allow_html=True)
        for i, a in enumerate(apercus):
            _ligne_projet(i, a, str(a["chemin"]) == dernier)


def _confirmation_suppression(racine: Path) -> None:
    nom = st.session_state.get("_confirmer_suppression")
    if not nom:
        return
    st.warning(f"Supprimer le projet **{nom}** et toutes ses données ? "
               "Cette action est irréversible.")
    with st.container(horizontal=True):
        if st.button("Annuler", key="btn_annuler_suppression"):
            st.session_state.pop("_confirmer_suppression", None)
            st.rerun()
        if st.button("Oui, supprimer", key="btn_confirmer_suppression", type="primary"):
            cible = racine / nom
            # Recontrôle juste avant le rmtree : un job a pu démarrer entre
            # l'affichage du bouton et ce clic.
            if runner.is_running(cible):
                st.error("Un job a démarré entre-temps sur ce projet : suppression annulée.")
            else:
                if cible.exists():
                    shutil.rmtree(cible)
                if current_project() is not None and current_project().name == nom:
                    set_current_project(None)
                st.toast(f"Projet « {nom} » supprimé")
            st.session_state.pop("_confirmer_suppression", None)
            _apercus.clear()
            st.rerun()


# ============================================================
# Panneau « Nouveau projet »
# ============================================================

_GENRES = {"single": ("Un seul", "1 vidéo = 1 session."),
           "multi": ("Plusieurs, en arènes séparées", "1 vidéo = N sessions, découpée par arène.")}
_PLUS_TARD = "(choisir plus tard)"


def _formulaire(racine: Path | None) -> None:
    st.markdown('<p class="ef-panneau-titre">Nouveau projet</p>'
                '<p class="ef-muted ef-panneau-sous">Créé dans l\'emplacement ci-contre.</p>',
                unsafe_allow_html=True)
    nom = st.text_input("Nom", placeholder="ex : bottomview-MCC-2026-06", key="creation_nom")
    cible = racine / nom.strip().replace(" ", "-") if (racine and nom.strip()) else None
    if cible is not None and cible.exists():
        st.markdown(f'<p class="ef-erreur">{escape(str(cible))} existe déjà.</p>',
                    unsafe_allow_html=True)
    else:
        st.caption("Les espaces deviennent des tirets.")

    st.markdown('<p class="ef-champ-titre">Combien d\'animaux par vidéo ?</p>',
                unsafe_allow_html=True)
    kind = st.radio("Combien d'animaux par vidéo ?", ["single", "multi"],
                    format_func=lambda k: _GENRES[k][0],
                    captions=[_GENRES[k][1] for k in ("single", "multi")],
                    key="creation_kind", label_visibility="collapsed", width="stretch")
    st.caption("Ne dépend pas de l'angle de la caméra.")

    modeles = list_dlc_models(models_root())
    chemins = {str(m / "config.yaml"): m.name for m in modeles}
    st.markdown('<p class="ef-champ-titre">Modèle DLC <span>facultatif</span></p>',
                unsafe_allow_html=True)
    choix = st.selectbox("Modèle DLC", [_PLUS_TARD, *chemins], key="creation_modele",
                         format_func=lambda c: "Choisir plus tard" if c == _PLUS_TARD else chemins[c],
                         label_visibility="collapsed")
    st.caption("Le modèle reste à sa place, il n'est pas copié. Modifiable plus tard.")

    bloque = cible is None or cible.exists()
    cmd = PL.create_project(cible, kind=kind, dlc_config=None if choix == _PLUS_TARD else choix) \
        if cible is not None else None
    if cmd is not None and not bloque:
        st.session_state["_creation_en_cours"] = str(cible)
        st.session_state["_creation_recap"] = {"nom": cible.name, "kind": kind,
                                               "modele": chemins.get(choix)}
    with st.container(key="pied_creation"):
        if cmd is None:
            st.button("Créer et ouvrir le projet", key="btn_creer", type="primary",
                      disabled=True, width="stretch",
                      help="Donne un nom au projet." if racine else "Choisis un emplacement valide.")
        else:
            # Le projet n'existe pas encore : son job tourne dans la racine
            # des projets (ruling P4) — un .ethoflow/ y apparaît, c'est voulu.
            _job.bouton_lancer(cible.parent, "Créer et ouvrir le projet", cmd,
                               cle="btn_creer", disabled=bloque, width="stretch",
                               help="Ce dossier existe déjà." if bloque else None)
        st.markdown('<p class="ef-pied-note">Étape suivante : décrire les animaux (Données)</p>',
                    unsafe_allow_html=True)


def _creation_en_cours(racine: Path) -> bool:
    """Récapitulatif + carte de job pendant/après la création. False s'il n'y
    a pas de création à suivre (le formulaire s'affiche alors)."""
    cible_txt = st.session_state.get("_creation_en_cours")
    job = runner.current(racine)
    # Le job doit viser CE dossier (dans sa ligne de commande) : un job de
    # création antérieur (autre nom) ne doit jamais passer pour le nôtre.
    if not cible_txt or job is None or job.script != "create_project.py" \
            or Path(cible_txt).parent != racine \
            or str(Path(cible_txt).resolve()) not in job.argv:
        return False
    cible = Path(cible_txt)
    if job.state == "succeeded":
        st.session_state.pop("_creation_en_cours", None)
        st.session_state.pop("_creation_recap", None)
        _apercus.clear()
        _ouvrir(cible)
    recap = st.session_state.get("_creation_recap") or {"nom": cible.name}
    if job.state == "running" or job.state == "failed":
        lignes = [("Nom", f'<span class="ef-mono">{escape(recap["nom"])}</span>'),
                  ("Animaux par vidéo", "Plusieurs (arènes)" if recap.get("kind") == "multi" else "Un seul"),
                  ("Modèle DLC", escape(recap.get("modele") or "plus tard"))]
        st.markdown('<p class="ef-panneau-titre">Nouveau projet</p>'
                    '<p class="ef-muted ef-panneau-sous">Créé dans l\'emplacement ci-contre.</p>'
                    '<div class="ef-recap">' + "".join(
                        f"<div><span>{k}</span><span>{v}</span></div>" for k, v in lignes) + "</div>",
                    unsafe_allow_html=True)
        journal = runner.read_log(racine, job.job_id)
        echec = job.state == "failed"
        with st.container(key="carte_creation_echec" if echec else "carte_creation"):
            with st.container(horizontal=True, vertical_alignment="center",
                              horizontal_alignment="distribute"):
                couleur = "#c9473f" if echec else "#c27a12"
                st.markdown(f'<p class="ef-creation-titre"><span class="ef-dot" style="background:'
                            f'{couleur}"></span>{"Création échouée" if echec else "Création en cours"}</p>',
                            unsafe_allow_html=True)
                if not echec and st.button("Annuler", key="ef_annuler_creation"):
                    runner.cancel(racine, job.job_id)
                    st.rerun()
            etapes = PR.sous_etapes_creation(journal, job.state)
            st.markdown('<div class="ef-sous-etapes">' + "".join(
                f'<div><span class="ef-dot" style="background:{PR.COULEURS[e]}"></span>'
                f'<span class="{"ef-todo" if e == "todo" else ""}">{escape(t)}</span></div>'
                for t, e in etapes) + "</div>", unsafe_allow_html=True)
            if echec:
                st.code(runner.read_log(racine, job.job_id, tail=8) or "(pas de sortie)",
                        language="text")
                with st.container(horizontal=True):
                    if st.button("Modifier", key="btn_creation_modifier"):
                        st.session_state.pop("_creation_en_cours", None)
                        st.rerun()
                    if st.button("Réessayer", key="btn_creation_reessayer", type="primary"):
                        runner.restart(racine, job)
                        st.rerun()
        st.caption("Le projet s'ouvre tout seul à la fin. En cas d'échec, la cause "
                   "s'affiche ici avec le journal.")
        if job.state == "running":
            st.fragment(run_every=2)(lambda: _rafraichir_si_fini(racine, job.job_id))()
        return True
    # Annulée ou interrompue : bandeau ambre, puis le formulaire.
    st.session_state.pop("_creation_en_cours", None)
    mot = "annulée" if job.state == "cancelled" else "interrompue"
    st.warning(f"Création de {cible.name} {mot}.")
    return False


def _rafraichir_si_fini(racine: Path, job_id: str) -> None:
    job = runner.current(racine)
    if job is None or job.job_id != job_id or job.state != "running":
        st.rerun()


def _panneau_nouveau(racine: Path | None) -> None:
    with st.container(key="ef_nouveau"):
        if racine is None or not _creation_en_cours(racine):
            _formulaire(racine)


# ============================================================
# Entrée
# ============================================================

def render() -> None:
    st.markdown('<h1 class="ef-titre-etape">Projets</h1><p class="ef-description-etape">'
                "Ouvre un projet pour reprendre là où tu t'es arrêté.</p>",
                unsafe_allow_html=True)
    racine = _barre_emplacement()
    if racine is not None:
        _confirmation_suppression(racine)
        _liste(racine)
    _panneau_nouveau(racine)
