"""Page Pose (DLC) — étape 5 du pipeline.

`run_dlc_inference.py` tourne dans l'env conda `dlc` et transforme chaque
vidéo de session en `.h5` de coordonnées de points-clés. Trois modes
existent (voir README §Étape 5) et se tromper coûte des heures de GPU :

- `custom` — un modèle DLC entraîné maison, désigné par
  `dlc_project_config` dans `configs/pipeline_config.yaml`. Le cas typique
  pour 1 animal par vidéo.
- `superanimal` — le modèle SuperAnimal multi-animal pré-entraîné, sur la
  vidéo entière. Le défaut pour N animaux par vidéo sans modèle custom
  (voie A de l'étape 4).
- `single-animal` — SuperAnimal contraint à un seul individu, sur des
  vidéos déjà croppées par arène (voie B de l'étape 4).

Le mode par défaut suit le projet plutôt qu'une constante : voir
`_mode_par_defaut`. Aucune commande n'est exécutée ici — `lib.pipeline`
construit, `views._job.bouton_lancer` lance via `lib.runner`.
"""
from __future__ import annotations

from pathlib import Path

import streamlit as st

import lib.pipeline as PL
from lib.config import (
    cropped_videos_exist,
    dlc_config_status,
    dlc_output_dir,
    project_kind,
    require_project,
)
from lib.icons import lucide_title
from views import _job, _widgets

# Options du mode d'inférence, dans l'ordre où le README §Étape 5 les
# présente : custom (1 animal, modèle maison), superanimal (N animaux,
# voie A), single-animal (N animaux déjà croppés, voie B).
_MODES = ["custom", "superanimal", "single-animal"]

_EXPLICATIONS = {
    "custom": (
        "Un modèle DLC que tu as entraîné toi-même, désigné par "
        "`dlc_project_config`. Le cas typique pour **1 animal par vidéo** "
        "(ex. bottom-view)."
    ),
    "superanimal": (
        "Le modèle SuperAnimal multi-animal pré-entraîné, lancé sur la "
        "vidéo entière — pas de crop. Le défaut pour **N animaux par "
        "vidéo** sans modèle custom (voie A de l'étape 4)."
    ),
    "single-animal": (
        "SuperAnimal contraint à un seul individu, lancé sur des vidéos "
        "**déjà croppées par arène** (voie B de l'étape 4, via "
        "`crop_arenes.py`)."
    ),
}


def _mode_par_defaut(projet: Path) -> str:
    """Le mode par défaut suit le projet, jamais une constante figée.

    Ordre de préférence :
    1. `custom` si un modèle DLC configuré existe réellement sur disque
       (`dlc_config_status` == "ok") — l'emporte toujours, même en `multi`,
       si le chercheur a pris la peine de désigner un modèle valide. Un
       modèle configuré mais introuvable (déplacé/supprimé) ne compte PAS
       comme "ok" : proposer `custom` par défaut dans ce cas mènerait
       droit à l'échec `--no-prompt` (ruling R12.1).
    2. `single-animal` si des vidéos croppées existent déjà (étape 4,
       voie B) : la suite logique est de les traiter, pas de relancer
       SuperAnimal sur la vidéo entière.
    3. `superanimal` sinon — fonctionne sans modèle custom, en single
       comme en multi-animal.
    """
    statut, _ = dlc_config_status(projet)
    if statut == "ok":
        return "custom"
    if cropped_videos_exist(projet):
        return "single-animal"
    return "superanimal"


def _section_mode(projet: Path) -> str:
    st.markdown(lucide_title("brain", "Mode d'inférence"), unsafe_allow_html=True)

    defaut = _mode_par_defaut(projet)
    mode = st.radio(
        "Mode",
        _MODES,
        index=_MODES.index(defaut),
        format_func=lambda m: {
            "custom": "custom — modèle DLC maison",
            "superanimal": "superanimal — SuperAnimal multi-animal (vidéo entière)",
            "single-animal": "single-animal — SuperAnimal sur vidéos croppées",
        }[m],
        key="pose_mode",
    )
    st.caption(_EXPLICATIONS[mode])

    if mode == "custom":
        statut, chemin = dlc_config_status(projet)
        if statut == "absent":
            st.warning(
                "Aucun modèle DLC n'est configuré pour ce projet. Avec "
                "`--no-prompt`, `run_dlc_inference.py --mode custom` "
                "échouerait immédiatement au lieu de demander un modèle à "
                "l'invite — désigne d'abord un modèle dans la section "
                "**Modèle DLC** de la page **Projet**."
            )
        elif statut == "introuvable":
            st.warning(
                f"Le modèle configuré (`{chemin}`) est introuvable — déplacé "
                "ou supprimé depuis. `run_dlc_inference.py --mode custom` "
                "échouerait immédiatement en `--no-prompt`. Va dans "
                "**Projet → Modèle DLC → Diagnostiquer** pour réparer, ou "
                "redésigne un modèle valide."
            )

    return mode


def _section_options(projet: Path, mode: str) -> tuple[list[str], bool, bool, bool, int]:
    st.markdown(lucide_title("settings", "Options"), unsafe_allow_html=True)

    choisies, tout = _widgets.selecteur_sessions(projet, cle="pose")

    col_skip, col_adapt = st.columns(2)
    with col_skip:
        skip_existing = st.checkbox(
            "Sauter les sessions déjà traitées (`--skip-existing`)",
            value=True, key="pose_skip_existing",
        )
    with col_adapt:
        video_adapt = st.checkbox(
            "Adapter le modèle aux vidéos (`--video-adapt`)",
            value=False, key="pose_video_adapt",
            help="Fine-tuning court de SuperAnimal sur les statistiques de "
                 "tes propres vidéos — plus lent, améliore la précision sur "
                 "des vidéos assez différentes du jeu d'entraînement.",
        )

    video_adapt_batch_size = 2
    if video_adapt:
        video_adapt_batch_size = st.number_input(
            "`--video-adapt-batch-size`",
            min_value=1, max_value=32, value=2, step=1,
            key="pose_video_adapt_batch_size",
            help="2 sur GPU 16 Go, 4 à 8 sur GPU 24 Go. Le défaut du script "
                 "(8) déborde la VRAM d'un RTX 4080/5080.",
        )

    return choisies, tout, skip_existing, video_adapt, int(video_adapt_batch_size)


def _section_lancement(
    projet: Path, mode: str, choisies: list[str], tout: bool,
    skip_existing: bool, video_adapt: bool, video_adapt_batch_size: int,
) -> None:
    # "absent" ET "introuvable" désactivent le bouton : dans les deux cas
    # `--mode custom --no-prompt` échouerait à coup sûr (ruling R12.1),
    # lancer un job qui va échouer n'aide personne.
    statut, _ = dlc_config_status(projet)
    modele_manquant = mode == "custom" and statut != "ok"
    pas_de_sessions = not tout and not choisies

    aide = None
    if modele_manquant:
        aide = "Modèle DLC absent ou introuvable — voir l'avertissement ci-dessus."
    elif pas_de_sessions:
        aide = "Sélectionne au moins une session, ou coche « Toutes les sessions »."

    cmd = PL.run_dlc_inference(
        projet, mode=mode,
        sessions=choisies, all_sessions=tout,
        skip_existing=skip_existing,
        video_adapt=video_adapt,
        video_adapt_batch_size=video_adapt_batch_size,
    )
    _job.bouton_lancer(
        projet, "Lancer l'inférence DLC", cmd,
        cle="btn_pose_lancer",
        disabled=modele_manquant or pas_de_sessions,
        help=aide,
    )

    _job.panneau(projet)
    _job.historique(projet)


def _section_artefacts(projet: Path) -> None:
    from lib import runner

    job = runner.current(projet)
    if job is None or job.script != "run_dlc_inference.py" or job.state != "succeeded":
        return

    st.markdown(lucide_title("circle-check", "Artefacts produits"), unsafe_allow_html=True)

    racine = dlc_output_dir(projet)
    if not racine.is_dir():
        st.info("Aucune sortie trouvée sous `data/dlc-output/`.")
        return

    trouve = False
    for session_dir in sorted(racine.iterdir()):
        if not session_dir.is_dir():
            continue
        # Récursif, pas un glob à plat : en mode single-animal,
        # run_superanimal_cropped écrit ses _labeled.mp4 dans un sous-dossier
        # scratch dlc-output/<session>/cropped-raw/, pas directement dans
        # dlc-output/<session>/ — un glob() plat les raterait précisément
        # dans le mode où le QC visuel compte le plus (ruling R12.2). En
        # mode custom, DLC (`analyze_videos`) n'écrit jamais de
        # _labeled.mp4 : une liste vide dans ce mode est normale, pas un
        # signe de problème.
        h5 = sorted(session_dir.rglob("*.h5"))
        labeled = sorted(session_dir.rglob("*_labeled.mp4"))
        if not h5 and not labeled:
            continue
        trouve = True
        with st.expander(session_dir.name, expanded=False):
            for f in h5 + labeled:
                st.caption(f"`{f.relative_to(projet)}`")

    if not trouve:
        st.info("Aucun `.h5` ni `_labeled.mp4` trouvé sous `data/dlc-output/`.")
    else:
        st.caption(
            "Pour un contrôle qualité visuel des `_labeled.mp4`, voir la "
            "page **Vidéos & calibration**."
        )


def render() -> None:
    projet = require_project()

    st.title("Pose (DLC)")
    st.caption(
        "Étape 5 du pipeline : inférence DeepLabCut sur la vidéo source de "
        "chaque session. Tourne dans l'env conda `dlc` — nécessite un GPU."
    )
    st.caption(f"Projet détecté comme `{project_kind(projet)}`.")

    mode = _section_mode(projet)
    st.divider()
    choisies, tout, skip_existing, video_adapt, video_adapt_batch_size = _section_options(projet, mode)
    st.divider()
    _section_lancement(
        projet, mode, choisies, tout,
        skip_existing, video_adapt, video_adapt_batch_size,
    )
    st.divider()
    _section_artefacts(projet)
