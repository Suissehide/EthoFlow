"""Page Visualisations — README étape 9, section « Visualisations
optionnelles » : les rendus qui finissent dans un papier ou un poster.

Facultatifs (le pipeline tourne sans), les trois scripts derrière cette
page tournent dans deux envs différents :

- `motif_gif.py` (bande de motifs) : env `ethoflow`.
- `behavior_structure_gif.py` (manifold) et `community_dendrogram.py`
  (dendrogramme) : env `vame` (umap/sklearn/scipy) — voir `SCRIPT_ENVS`
  dans `lib/pipeline.py`, jamais modifié ici.

`--labels` est pré-rempli depuis `motif_labels.csv` quand il existe
(`lib.motif_labels.exists`/`path`) : sans lui, les trois rendus affichent
`motif_0`, `motif_1`, … au lieu des noms de comportements — des heures
d'annotation (page **Motifs**) ignorées en silence sinon.

Aucune commande n'est exécutée ici : `lib.pipeline` construit,
`views._job.bouton_lancer` lance via `lib.runner`. La découverte des
rendus déjà produits et le recouvrement de leurs paramètres (l'`argv` du
job qui les a générés) vivent dans `lib.renders`, testable sans Streamlit
— voir ce module pour le détail de pourquoi le nom de fichier seul ne
suffit pas à retrouver les paramètres (suffixe `_sidebyside` conditionné
au succès réel de l'ouverture de la vidéo, pas au flag demandé).
"""
from __future__ import annotations

from pathlib import Path

import streamlit as st

import lib.motif_labels as ML
import lib.pipeline as PL
import lib.renders as RD
import lib.vame as VA
from lib.config import require_project
from lib.icons import lucide_title
from views import _job
from views import _widgets

_ALGOS_PAR_DEFAUT = ["hmm", "kmeans"]


def _algos_presents(projet: Path) -> list[str]:
    """Algos réellement segmentés pour ce projet — mêmes deux lignes que
    `views/analyses.py` (`_section_options_avancees`), reprises ici pour
    ne pas coupler les deux pages par un import croisé."""
    presents = sorted({VA.parse_algo_n(a)[0] for a in VA.list_algos(projet)})
    return presents or _ALGOS_PAR_DEFAUT


def _champ_labels(projet: Path, *, cle: str) -> str | None:
    defaut = str(ML.path(projet)) if ML.exists(projet) else ""
    valeur = st.text_input(
        "Fiche des motifs", value=defaut, key=cle,
        help="Trouvée automatiquement dans le projet ; à changer seulement "
             "pour utiliser une autre fiche.",
    )
    if not valeur:
        st.caption(
            "Pas encore de noms de motifs : ce rendu affichera « motif_0, "
            "motif_1… ». Remplis-les dans la page **Motifs** avant une figure "
            "de publication."
        )
    return valeur or None


# ============================================================
# Affichage des rendus déjà produits — commun aux trois onglets
# ============================================================

def _affiche_rendu(rendu: RD.Render, *, cle: str) -> None:
    chemin = rendu.path
    with st.container(border=True):
        col_apercu, col_info = st.columns([2, 3])
        with col_apercu:
            if chemin.suffix.lower() == ".mp4":
                st.video(str(chemin))
            else:
                st.image(str(chemin), width="stretch")
        with col_info:
            st.caption(f"`{chemin.name}`")
            if rendu.params is not None:
                for ligne in _lignes_params(rendu.params):
                    st.caption(ligne)
            else:
                st.caption(
                    "Paramètres non retrouvés — le job qui a produit ce "
                    "fichier n'apparaît plus dans l'historique du projet."
                )
            try:
                contenu = chemin.read_bytes()
            except OSError:
                contenu = None
            if contenu is not None:
                st.download_button(
                    "Télécharger", data=contenu, file_name=chemin.name,
                    key=f"{cle}_dl_{chemin.name}",
                )


def _lignes_params(params: dict) -> list[str]:
    """Format lisible des kwargs recouvrés — une ligne par render."""
    if "projection" in params:      # manifold
        morceaux = [f"session {params['session']}", f"méthode {params['algo']}",
                   f"projection {params['projection']}"]
        if params.get("start"):
            morceaux.append(f"début {params['start']:g}s")
        if params.get("duration"):
            morceaux.append(f"durée {params['duration']:g}s")
        morceaux.append(f"format {params['output_format']}")
        if params.get("with_video"):
            morceaux.append("avec vidéo")
        if params.get("pool_all_sessions"):
            morceaux.append("fond commun à toutes les sessions")
    elif "session" in params:       # bande de motifs
        morceaux = [f"session {params['session']}", f"méthode {params['algo']}"]
        if params.get("start"):
            morceaux.append(f"début {params['start']:g}s")
        if params.get("duration"):
            morceaux.append(f"durée {params['duration']:g}s")
        morceaux.append(f"format {params['output_format']}")
    else:                           # dendrogramme
        morceaux = [f"méthode {params['algo']}", f"liaison {params['linkage']}"]
        if params.get("group"):
            morceaux.append(f"groupe {params['group']}")
    return [
        " · ".join(morceaux),
        "avec les noms des motifs" if params.get("labels") else "sans noms (motif_0, motif_1…)",
    ]


def _section_rendus(projet: Path, rendus: list[RD.Render], *, cle: str) -> None:
    st.markdown("**Rendus précédents** (du plus récent au plus ancien)")
    if not rendus:
        st.caption("Aucun rendu pour l'instant.")
        return
    for i, rendu in enumerate(rendus):
        _affiche_rendu(rendu, cle=f"{cle}_{i}")


# ============================================================
# Onglet 1 : Bande de motifs (motif_gif.py, env ethoflow)
# ============================================================

def _tab_motif_gif(projet: Path, sessions: list[str]) -> None:
    st.caption(
        "La vidéo de la session, avec sous l'image un bandeau coloré qui "
        "indique le motif en cours à chaque instant."
    )

    session = st.selectbox("Session", options=sessions, key="visu_motif_session")
    algo = st.selectbox("Méthode de découpage", options=_algos_presents(projet),
                        key="visu_motif_algo")

    limiter = st.checkbox("Limiter la durée", value=False, key="visu_motif_limiter")
    col_start, col_duration = st.columns(2)
    with col_start:
        start = st.number_input("Début (s)", min_value=0.0, value=0.0,
                                step=1.0, key="visu_motif_start")
    duration = None
    if limiter:
        with col_duration:
            duration = st.number_input("Durée (s)", min_value=0.1, value=30.0,
                                       step=1.0, key="visu_motif_duration")

    if not limiter:
        st.caption(
            "Sans durée limitée, ce rendu **remplace** le précédent rendu "
            "de la même session, quel que soit son début. Limite la durée "
            "pour garder plusieurs rendus à comparer."
        )

    output_format = st.radio(
        "Format", options=["mp4", "gif"], horizontal=True,
        key="visu_motif_format",
        help="mp4 pour archiver ; gif pour partager (moins de 2 minutes "
             "pour un fichier raisonnable).",
    )
    labels = _champ_labels(projet, cle="visu_motif_labels")

    cmd = PL.motif_gif(
        projet, session=session, algo=algo, start=float(start),
        duration=float(duration) if duration else None,
        output_format=output_format, labels=labels,
    )
    _job.bouton_lancer(
        projet, f"Générer la bande de motifs — {session}", cmd,
        cle="btn_visu_motif_gif",
    )

    st.divider()
    _section_rendus(projet, RD.motif_gif_renders(projet), cle="visu_motif")


# ============================================================
# Onglet 2 : Manifold (behavior_structure_gif.py, env vame)
# ============================================================

def _tab_manifold(projet: Path, sessions: list[str]) -> None:
    st.caption(
        "La trajectoire animée de la session dans l'espace des "
        "comportements appris par VAME, avec en option la vidéo à côté."
    )

    session = st.selectbox("Session", options=sessions, key="visu_manifold_session")
    col_algo, col_proj = st.columns(2)
    with col_algo:
        algo = st.selectbox("Méthode de découpage", options=_algos_presents(projet),
                            key="visu_manifold_algo")
    with col_proj:
        projection = st.radio(
            "Projection", options=["umap", "pca"], horizontal=True,
            key="visu_manifold_projection",
            help="umap par défaut ; pca si umap n'est pas installé.",
        )

    limiter = st.checkbox("Limiter la durée", value=False, key="visu_manifold_limiter")
    col_start, col_duration = st.columns(2)
    with col_start:
        start = st.number_input("Début (s)", min_value=0.0, value=0.0,
                                step=1.0, key="visu_manifold_start")
    duration = None
    if limiter:
        with col_duration:
            duration = st.number_input("Durée (s)", min_value=0.1, value=30.0,
                                       step=1.0, key="visu_manifold_duration")

    output_format = st.radio(
        "Format", options=["gif", "mp4"], horizontal=True,
        key="visu_manifold_format",
    )
    with_video = st.checkbox(
        "Vidéo de la session à côté", value=False,
        key="visu_manifold_with_video",
        help="Il faut que la vidéo de la session soit trouvée. Ralentit un "
             "peu le rendu.",
    )
    labeled_video = st.checkbox(
        "Vidéo annotée (points du corps dessinés) à la place",
        value=False, key="visu_manifold_labeled",
        help="Utilise la vidéo annotée par la pose, si elle existe.",
    )
    labeled_pcutoff = None
    if labeled_video:
        labeled_pcutoff = st.number_input(
            "Seuil de confiance de la vidéo annotée (vide = la plus récente)",
            min_value=0.0, max_value=1.0, value=None, step=0.05,
            key="visu_manifold_labeled_pcutoff", width=320,
            help="Avec un seuil : exactement la vidéo annotée à ce seuil, "
                 "sans en prendre une autre.",
        )
    pool_all_sessions = st.checkbox(
        "Fond commun à toutes les sessions (rendus comparables entre eux)",
        value=False, key="visu_manifold_pool",
    )
    if pool_all_sessions:
        st.warning(
            "**Lent, et peut se bloquer** : le calcul porte alors sur toutes "
            "les sessions du projet, jusqu'à plusieurs dizaines de minutes "
            "sur un gros projet. C'est ce qui rend les rendus comparables "
            "entre eux (même fond)."
        )

    labels = _champ_labels(projet, cle="visu_manifold_labels")

    cmd = PL.behavior_structure_gif(
        projet, session=session, algo=algo, projection=projection,
        start=float(start), duration=float(duration) if duration else None,
        output_format=output_format, with_video=with_video,
        pool_all_sessions=pool_all_sessions, labels=labels,
        labeled_video=labeled_video,
        labeled_pcutoff=float(labeled_pcutoff) if labeled_pcutoff is not None else None,
    )
    _job.bouton_lancer(
        projet, f"Générer le manifold — {session}", cmd,
        cle="btn_visu_manifold",
    )

    st.divider()
    _section_rendus(projet, RD.manifold_renders(projet), cle="visu_manifold_rendus")


# ============================================================
# Onglet 3 : Dendrogramme (community_dendrogram.py, env vame)
# ============================================================

def _tab_dendrogramme(projet: Path) -> None:
    st.caption(
        "L'arbre de parenté des motifs, avec leurs noms plutôt que leurs "
        "numéros."
    )

    algo = st.selectbox("Méthode de découpage", options=_algos_presents(projet),
                        key="visu_dendro_algo", width=320)
    group = st.text_input(
        "Groupe (facultatif, ex : MCCiECKO)", value="", key="visu_dendro_group", width=320,
    )
    linkage = st.selectbox(
        "Méthode de liaison", options=["ward", "average", "complete", "single"],
        key="visu_dendro_linkage", width=320,
    )
    labels = _champ_labels(projet, cle="visu_dendro_labels")

    cmd = PL.community_dendrogram(
        projet, algo=algo, group=group.strip() or None, linkage=linkage, labels=labels,
    )
    _job.bouton_lancer(
        projet, "Générer le dendrogramme", cmd, cle="btn_visu_dendro",
    )

    st.divider()
    _section_rendus(projet, RD.dendrogram_renders(projet), cle="visu_dendro_rendus")


# ============================================================
# Entrée
# ============================================================

def render() -> None:
    projet = require_project()

    _widgets.entete_etape(projet)

    if not VA.list_algos(projet):
        st.info(
            "Aucune segmentation VAME trouvée pour ce projet — les trois "
            "rendus de cette page lisent les motifs produits par l'étape "
            "« segment » de VAME (page **VAME**). Lance-la d'abord."
        )
        return

    sessions = VA.list_sessions(projet)

    _job.panneau(projet)

    onglet_motif, onglet_manifold, onglet_dendro = st.tabs([
        "Bande de motifs", "Manifold", "Dendrogramme",
    ])
    with onglet_motif:
        st.markdown(lucide_title("clapperboard", "Bande de motifs"), unsafe_allow_html=True)
        if not sessions:
            st.caption("Aucune session segmentée trouvée.")
        else:
            _tab_motif_gif(projet, sessions)
    with onglet_manifold:
        st.markdown(lucide_title("clapperboard", "Manifold"), unsafe_allow_html=True)
        if not sessions:
            st.caption("Aucune session segmentée trouvée.")
        else:
            _tab_manifold(projet, sessions)
    with onglet_dendro:
        st.markdown(lucide_title("clapperboard", "Dendrogramme"), unsafe_allow_html=True)
        _tab_dendrogramme(projet)

    st.divider()
    _job.historique(projet)
