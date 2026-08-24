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
        "`--labels`", value=defaut, key=cle,
        help="motif_labels.csv (auto : <vame>/motif_labels.csv).",
    )
    if not valeur:
        st.caption(
            "Pas de `motif_labels.csv` pour ce projet — sans lui, ce "
            "rendu affichera `motif_0`, `motif_1`, … au lieu des noms de "
            "comportements. Remplis-le dans la page **Motifs** avant de "
            "produire une figure destinée à une publication."
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
                st.image(str(chemin), use_container_width=True)
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
        morceaux = [f"session `{params['session']}`", f"algo `{params['algo']}`",
                   f"projection `{params['projection']}`"]
        if params.get("start"):
            morceaux.append(f"début {params['start']:g}s")
        if params.get("duration"):
            morceaux.append(f"durée {params['duration']:g}s")
        morceaux.append(f"format `{params['output_format']}`")
        if params.get("with_video"):
            morceaux.append("avec vidéo")
        if params.get("pool_all_sessions"):
            morceaux.append("pool-all-sessions")
    elif "session" in params:       # bande de motifs
        morceaux = [f"session `{params['session']}`", f"algo `{params['algo']}`"]
        if params.get("start"):
            morceaux.append(f"début {params['start']:g}s")
        if params.get("duration"):
            morceaux.append(f"durée {params['duration']:g}s")
        morceaux.append(f"format `{params['output_format']}`")
    else:                           # dendrogramme
        morceaux = [f"algo `{params['algo']}`", f"linkage `{params['linkage']}`"]
        if params.get("group"):
            morceaux.append(f"groupe `{params['group']}`")
    return [
        " · ".join(morceaux),
        "avec labels" if params.get("labels") else "sans labels (`motif_0`, `motif_1`, …)",
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
        "Vidéo (ou GIF) de la session, avec un bandeau color-codé sous "
        "l'image indiquant le motif VAME actif à chaque instant. Env "
        "`ethoflow` — `motif_gif.py`."
    )

    session = st.selectbox("Session", options=sessions, key="visu_motif_session")
    algo = st.selectbox("`--algo`", options=_algos_presents(projet), key="visu_motif_algo")

    col_start, col_duration = st.columns(2)
    with col_start:
        start = st.number_input("`--start` (s)", min_value=0.0, value=0.0,
                                step=1.0, key="visu_motif_start")
    with col_duration:
        limiter = st.checkbox("Limiter la durée (`--duration`)", value=False,
                              key="visu_motif_limiter")
        duration = None
        if limiter:
            duration = st.number_input("`--duration` (s)", min_value=0.1, value=30.0,
                                       step=1.0, key="visu_motif_duration")

    if not limiter:
        st.caption(
            "Sans `--duration`, `motif_gif.py` n'inscrit pas `--start` dans "
            "le nom de fichier (`<session>_annotated.mp4`, toujours le "
            "même) : ce rendu **remplacera** un précédent rendu de la même "
            "session, quel que soit son `--start`. Renseigne aussi une "
            "durée si tu veux garder plusieurs rendus distincts pour "
            "comparer."
        )

    output_format = st.radio(
        "`--output-format`", options=["mp4", "gif"], horizontal=True,
        key="visu_motif_format",
        help="mp4 pour archivage/analyse ; gif pour partage web (garde "
             "`--duration` < 120 s pour un gif raisonnable).",
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
        "Reproduit la visu « behavior manifold » du README VAME : la "
        "trajectoire de la session dans l'espace latent, animée — avec "
        "en option la vraie vidéo à côté. Env `vame` (umap + sklearn) — "
        "`behavior_structure_gif.py`."
    )

    session = st.selectbox("Session", options=sessions, key="visu_manifold_session")
    col_algo, col_proj = st.columns(2)
    with col_algo:
        algo = st.selectbox("`--algo`", options=_algos_presents(projet),
                            key="visu_manifold_algo")
    with col_proj:
        projection = st.radio(
            "`--projection`", options=["umap", "pca"], horizontal=True,
            key="visu_manifold_projection",
            help="umap par défaut (bascule automatiquement sur pca si "
                 "umap-learn est absent de l'env).",
        )

    col_start, col_duration = st.columns(2)
    with col_start:
        start = st.number_input("`--start` (s)", min_value=0.0, value=0.0,
                                step=1.0, key="visu_manifold_start")
    with col_duration:
        limiter = st.checkbox("Limiter la durée (`--duration`)", value=False,
                              key="visu_manifold_limiter")
        duration = None
        if limiter:
            duration = st.number_input("`--duration` (s)", min_value=0.1, value=30.0,
                                       step=1.0, key="visu_manifold_duration")

    output_format = st.radio(
        "`--output-format`", options=["gif", "mp4"], horizontal=True,
        key="visu_manifold_format",
    )
    with_video = st.checkbox(
        "`--with-video` — panneau vidéo à côté du manifold", value=False,
        key="visu_manifold_with_video",
        help="Nécessite OpenCV et un `source_video` valide dans "
             "`metadata.yaml`. Ralentit un peu le rendu.",
    )
    pool_all_sessions = st.checkbox(
        "`--pool-all-sessions` — référentiel commun à toutes les sessions",
        value=False, key="visu_manifold_pool",
    )
    if pool_all_sessions:
        st.warning(
            "**Lent, et documenté comme pouvant se bloquer** (voir la "
            "section Troubleshooting du README, « Rendu "
            "`behavior_structure_gif` bloqué en mode `--pool-all-sessions` "
            "»). UMAP tourne alors sur les points de TOUTES les sessions "
            "du projet, pas seulement celle-ci — jusqu'à plusieurs "
            "dizaines de minutes sur un gros projet. C'est en échange ce "
            "qui rend les rendus comparables entre eux (même fond "
            "commun) — mieux vaut le savoir avant de lancer plutôt que de "
            "le découvrir après vingt minutes d'attente."
        )

    labels = _champ_labels(projet, cle="visu_manifold_labels")

    cmd = PL.behavior_structure_gif(
        projet, session=session, algo=algo, projection=projection,
        start=float(start), duration=float(duration) if duration else None,
        output_format=output_format, with_video=with_video,
        pool_all_sessions=pool_all_sessions, labels=labels,
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
        "Refait la dendrogramme des communautés de motifs VAME, avec les "
        "noms du CSV de labels au lieu des identifiants numériques. Env "
        "`vame` (scipy) — `community_dendrogram.py`."
    )

    algo = st.selectbox("`--algo`", options=_algos_presents(projet), key="visu_dendro_algo")
    group = st.text_input(
        "`--group` (optionnel — filtre par groupe, ex : `MCCiECKO`)",
        value="", key="visu_dendro_group",
    )
    linkage = st.selectbox(
        "`--linkage`", options=["ward", "average", "complete", "single"],
        key="visu_dendro_linkage",
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

    st.title("Visualisations")
    st.caption(
        "Rendus optionnels du README (étape 9) — le pipeline tourne sans, "
        "mais ce sont ceux qui finissent dans un papier ou un poster."
    )

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
