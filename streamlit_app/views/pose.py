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
    require_project,
)
from lib.icons import lucide_title
from views import _job, _widgets

# Options du mode d'inférence, dans l'ordre où le README §Étape 5 les
# présente : custom (1 animal, modèle maison), superanimal (N animaux,
# voie A), single-animal (N animaux déjà croppés, voie B).
_MODES = ["custom", "superanimal", "single-animal"]

# Libellés et descriptions en clair (handoff) ; le nom du mode reste dans
# « Détails techniques ».
_TITRES = {
    "custom": "Mon modèle entraîné",
    "superanimal": "SuperAnimal, vidéo entière",
    "single-animal": "SuperAnimal, vidéos découpées",
}
_DESCRIPTIONS = {
    "custom": "Le modèle DLC désigné pour ce projet{nom}. Le bon choix pour 1 animal par vidéo.",
    "superanimal": "Modèle pré-entraîné, plusieurs animaux à la fois. Sans entraînement préalable.",
    "single-animal": "Un animal par vidéo, sur des vidéos déjà découpées par arène.",
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
    st.markdown(lucide_title("brain", "Modèle à utiliser"), unsafe_allow_html=True)
    st.caption("Un modèle entraîné pour ton setup caméra est en général plus précis, "
               "et sert à tous les projets filmés avec ce même setup. SuperAnimal "
               "marche tout de suite, sans entraînement.")

    defaut = _mode_par_defaut(projet)
    statut, chemin = dlc_config_status(projet)
    nom = f" ({Path(chemin).parent.name})" if statut == "ok" else ""
    mode = st.radio(
        "Modèle à utiliser", _MODES, index=_MODES.index(defaut),
        # Le badge reste sur « Mon modèle entraîné », même sans modèle
        # désigné : c'est la voie recommandée, la carte ci-dessous aide à y aller.
        format_func=lambda m: _TITRES[m] + (
            " :violet-badge[Recommandé pour ce projet]" if m == "custom" else ""),
        captions=[_DESCRIPTIONS[m].format(nom=nom) for m in _MODES],
        key="pose_mode", label_visibility="collapsed", width="stretch",
    )

    if mode == "custom" and statut != "ok":
        # Désigner le modèle ici même, sans quitter la page (au lieu d'un
        # message qui renvoie vers le menu du projet).
        with st.container(key="carte_modele"):
            st.markdown('<p class="ef-carte-titre" style="margin:0">'
                        "Choisir le modèle DLC du projet</p>", unsafe_allow_html=True)
            if statut == "introuvable":
                st.warning(f"Le modèle désigné est introuvable (déplacé ou supprimé) : "
                           f"{chemin}. Choisis-le à son nouvel emplacement.")
            else:
                st.caption("Aucun modèle n'est encore désigné pour ce projet. Choisis le "
                           "dossier du modèle entraîné : il sert pour toutes les sessions. "
                           "Pas encore de modèle ? Prends SuperAnimal ci-dessus, ou "
                           "entraînes-en un (Parcours B du README, au terminal).")
            _widgets.choix_modele_dlc(projet, cle="pose_modele")
    return mode


def _section_options(projet: Path, mode: str) -> dict:
    choisies, tout = _widgets.selecteur_sessions(projet, cle="pose", colonne="DLC")

    o = {"choisies": choisies, "tout": tout, "batch_size": None, "fp16": False,
         "video_adapt_batch_size": 2}
    o["skip_existing"] = st.checkbox("Ignorer les sessions déjà traitées",
                                     value=True, key="pose_skip_existing")
    # L'adaptation ne concerne que SuperAnimal : le script l'ignore en mode custom.
    o["video_adapt"] = mode != "custom" and st.checkbox(
        "Adapter le modèle à mes vidéos :gray[(plus lent, plus précis)]",
        value=False, key="pose_video_adapt",
        help="Court ré-entraînement de SuperAnimal sur tes propres vidéos : "
             "utile quand elles diffèrent beaucoup de celles d'origine.")

    with st.expander("Réglages avancés"):
        if o["video_adapt"]:
            o["video_adapt_batch_size"] = int(st.number_input(
                "Lot d'adaptation", min_value=1, max_value=32, value=2,
                step=1, key="pose_video_adapt_batch_size", width=220,
                help="2 sur une carte graphique de 16 Go, 4 à 8 sur 24 Go."))
        if mode == "custom":
            o["batch_size"] = int(st.number_input(
                "Images par lot", min_value=1,
                max_value=128, value=16, step=1, key="pose_batch_size", width=220,
                help="Images traitées ensemble par la carte graphique. 16 est un bon "
                     "départ sur une carte de 16 Go ; baisse si sa mémoire sature."))
            o["fp16"] = st.checkbox(
                "Calcul en demi-précision :gray[(≈ 2× plus rapide, écart sous le pixel)]",
                value=False, key="pose_fp16",
                help="Utilise mieux les cartes graphiques récentes. L'écart sur les "
                     "positions reste sous le pixel, bien en dessous de la précision "
                     "du modèle (2 à 3 pixels).")
        if not o["video_adapt"] and mode != "custom":
            st.caption("Aucun réglage avancé pour ce modèle.")
    return o


def _section_lancement(projet: Path, mode: str, o: dict) -> None:
    # Modèle "absent" ET "introuvable" désactivent le bouton : dans les deux
    # cas `--mode custom --no-prompt` échouerait à coup sûr (ruling R12.1).
    statut, _ = dlc_config_status(projet)
    modele_manquant = mode == "custom" and statut != "ok"
    n = _widgets.nb_sessions(projet, o["choisies"], o["tout"], colonne="DLC",
                             ignorer_faites=o["skip_existing"])

    aide = None
    if modele_manquant:
        aide = "Désigne d'abord un modèle DLC (voir l'avertissement ci-dessus)."
    elif n == 0:
        aide = ("Toutes ces sessions sont déjà traitées." if o["skip_existing"]
                else "Choisis au moins une session.")

    # `--all` ne prend que les sessions non traitées : pour tout refaire, on
    # passe la liste complète.
    sessions, tout = o["choisies"], o["tout"]
    if tout and not o["skip_existing"]:
        from lib.sessions import list_sessions
        sessions, tout = list(list_sessions(projet)["session_id"]), False
    cmd = PL.run_dlc_inference(
        projet, mode=mode, sessions=sessions, all_sessions=tout,
        skip_existing=o["skip_existing"], video_adapt=o["video_adapt"],
        video_adapt_batch_size=o["video_adapt_batch_size"],
        batch_size=o["batch_size"], fp16=o["fp16"],
    )
    st.caption("Avec une carte graphique, compte quelques dizaines de minutes par "
               "vidéo de 20 min, plus si tu adaptes le modèle. Sans carte graphique, "
               "ça marche aussi, mais 10 à 50 fois plus lentement.")
    _job.bouton_lancer(
        projet, _widgets.libelle_lancer(n), cmd,
        cle="btn_pose_lancer", disabled=modele_manquant or n == 0, help=aide,
    )
    _job.panneau(projet)
    with st.expander("Vérifier le résultat avant le nettoyage"):
        st.markdown(
            "- Avec « Mon modèle entraîné », aucune vidéo annotée n'est produite. "
            "Pour voir les points sur la vidéo, il faut les redessiner à partir du "
            "fichier de poses (quelques dizaines de secondes, sans refaire la pose : "
            "voir **Détails techniques**).\n"
            "- Deux vidéos sortent : seuil 0,3 et seuil 0,6. La 0,6 efface les points "
            "dont le modèle doute, donc elle paraît propre et cache les hésitations. "
            "Compare les deux.\n"
            "- Sauts rares et brefs : le nettoyage les répare. Sauts fréquents, ou "
            "pattes gauche et droite inversées tout le temps : c'est le modèle qu'il "
            "faut reprendre (Parcours B du README)."
        )
    _widgets.details_techniques(cmd, notes=[
        f"Mode d'inférence : `{mode}`.",
        "Sorties : `data/dlc-output/<session>/` (un `.h5` de coordonnées par "
        "session ; vidéos annotées `_labeled.mp4` hors mode `custom`).",
        "Redessiner les points (mode `custom` seulement), env `dlc` : "
        "`python scripts/relabel_video.py --session <session>` → "
        "`<stem>_labeled_p30.mp4` et `<stem>_labeled_p60.mp4`.",
        "Erreur « Could not find a shuffle… » : bouton **Diagnostiquer** dans la "
        "page **Projet** (souvent un modèle déplacé, réparable).",
        "Pose très lente : la carte graphique n'est peut-être pas vue. Diagnostic "
        "dans l'env `dlc` : `python scripts/diagnose_gpu.py`.",
    ])


def render() -> None:
    projet = require_project()
    _widgets.entete_etape(projet)
    mode = _section_mode(projet)
    options = _section_options(projet, mode)
    _section_lancement(projet, mode, options)
