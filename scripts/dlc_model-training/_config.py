"""Configuration centralisée pour les scripts d'entraînement DLC.

Les scripts numérotés 01 → 06 importent depuis ici. Édite ce fichier
une seule fois par projet DLC — pas de chemins absolus disséminés dans
les autres scripts.

Ce module marche pour les deux vues (top et bottom) : les scripts en
aval sont agnostiques, seul le `SUPERANIMAL_NAME` choisi ci-dessous
change entre les setups.

Conventions :
- Les chemins Windows utilisent des raw strings (r"...") pour éviter
  de doubler les backslashes.
- `CONFIG` est un str (pas Path) car DeepLabCut l'attend en str.
"""
from __future__ import annotations

from pathlib import Path


# ----------------------------------------------------------------------
# Projet DLC
# ----------------------------------------------------------------------

# Ces trois valeurs déterminent le nom du dossier de projet DLC :
#   <WORKDIR>/<PROJECT_NAME>-<EXPERIMENTER>-<YYYY-MM-DD>/
# Le dossier est créé par `01_setup_project.py` avec la date du jour.
PROJECT_NAME = "souris-bottomview"
EXPERIMENTER = "labo"
WORKDIR = Path(r"D:\EthoFlow\models")

# Le projet DLC vit à `<WORKDIR>/<PROJECT_NAME>/` — même dossier que
# ce _config.py. DLC ajoute par défaut un suffixe `-<EXPERIMENTER>-<date>`
# au nom du dossier qu'il crée, mais 01_setup_project.py merge tout de
# suite ce contenu dans <WORKDIR>/<PROJECT_NAME>/ (où vit déjà ton
# _config.py écrit par le wizard) et supprime le dossier daté vide.
# Résultat : un seul dossier propre par projet, pas d'édition manuelle.
PROJECT_DIR = WORKDIR / PROJECT_NAME
CONFIG = str(PROJECT_DIR / "config.yaml")


# ----------------------------------------------------------------------
# Vidéos
# ----------------------------------------------------------------------

# Vidéo principale utilisée pour le pilote (première itération de
# labellisation). Choisis une vidéo représentative : bonne qualité,
# comportements variés, souris qui bouge suffisamment.
PILOT_VIDEO = Path(r"D:\path\to\pilot_video.mp4")

# Liste des vidéos à analyser par `03_apply.py`. Peut contenir uniquement
# PILOT_VIDEO au début, puis s'enrichir des vidéos ajoutées via 04.
VIDEOS_TO_ANALYZE: list[Path] = [
    PILOT_VIDEO,
    # Path(r"D:\path\to\other_video.mp4"),
]


# ----------------------------------------------------------------------
# Sortie
# ----------------------------------------------------------------------

# Un dossier par vidéo source dans <PROJECT_DIR>/result-videos/<stem>/
# → pas de mélange entre runs.
RESULTS_DIR = PROJECT_DIR / "result-videos"


# ----------------------------------------------------------------------
# Extraction de frames (01_setup_project.py + 04_add_videos.py)
# ----------------------------------------------------------------------

# Nombre de frames extraites automatiquement par k-means au setup.
# Recommandation Tony (VAME/LIN) pour un vrai premier entraînement :
#   - viser 200-300 frames de training total
#   - dont 100-150 par k-means (couvre la diversité globale)
#   - dont 50-150 sélectionnées MANUELLEMENT dans les situations
#     difficiles (rearing, occlusions, pattes floues) — cf. Parcours B
#     du README pour la stratégie de sélection.
# Valeur par défaut : 120 pour rester dans le range k-means, à monter
# si tu as peu de vidéos ou beaucoup de variabilité inter-individu.
N_AUTO_FRAMES = 120


# ----------------------------------------------------------------------
# Anatomie souris (bodyparts + skeleton, écrits dans le config.yaml DLC)
# ----------------------------------------------------------------------

# Deux jeux de 12 keypoints, un par vue caméra — c'est `SUPERANIMAL_NAME`
# plus bas qui décide lequel devient DEFAULT_BODYPARTS/DEFAULT_SKELETON.
# Un seul jeu servi aux deux vues labelliserait des points invisibles :
# par en dessous on ne voit pas les oreilles, par au-dessus on ne voit ni
# le menton ni le ventre. Et un keypoint qu'on ne voit pas, on le place au
# jugé — c'est du bruit d'apprentissage payé au prix fort, une frame à la
# main à la fois.

# Bottom-view (souris filmée par en dessous, à travers un plancher
# transparent) : la ligne médiane du dessous est visible sur toute la
# longueur, les quatre pattes aussi.
BODYPARTS_BOTTOMVIEW: list[str] = [
    "nose",
    "chin",
    "head_center",
    "chest_center",
    "belly_center",
    "tail_base",
    "tail_mid",
    "tail_tip",
    "front_paw_left",
    "front_paw_right",
    "hind_paw_left",
    "hind_paw_right",
]

# Skeleton = liaisons anatomiques entre keypoints. Sert au regroupement
# multi-animal DLC et au tracking. Format : liste de [kp_A, kp_B].
# Ici : l'axe du corps du nez à la queue, les pattes avant rattachées au
# poitrail, les pattes arrière au ventre.
SKELETON_BOTTOMVIEW: list[list[str]] = [
    ["nose", "chin"],
    ["chin", "head_center"],
    ["head_center", "chest_center"],
    ["chest_center", "belly_center"],
    ["belly_center", "tail_base"],
    ["tail_base", "tail_mid"],
    ["tail_mid", "tail_tip"],
    ["chest_center", "front_paw_left"],
    ["chest_center", "front_paw_right"],
    ["belly_center", "hind_paw_left"],
    ["belly_center", "hind_paw_right"],
]

# Top-view (arène ouverte filmée de dessus) : les oreilles servent de
# repère d'orientation de la tête, les pattes ne sont visibles que par
# intermittence — on les garde, DLC gère les points occultés par la
# likelihood, et le nettoyage de l'étape 6b s'en occupe ensuite.
BODYPARTS_TOPVIEW: list[str] = [
    "nose",
    "left_ear",
    "right_ear",
    "center",
    "left_flank",
    "tail_base",
    "tail_mid",
    "tail_tip",
    "front_paw_left",
    "front_paw_right",
    "hind_paw_left",
    "hind_paw_right",
]

SKELETON_TOPVIEW: list[list[str]] = [
    ["nose", "left_ear"],
    ["nose", "right_ear"],
    ["left_ear", "right_ear"],
    ["nose", "center"],
    ["center", "left_flank"],
    ["center", "front_paw_left"],
    ["center", "front_paw_right"],
    ["center", "hind_paw_left"],
    ["center", "hind_paw_right"],
    ["center", "tail_base"],
    ["tail_base", "tail_mid"],
    ["tail_mid", "tail_tip"],
]

KEYPOINTS_PAR_VUE: dict[str, tuple[list[str], list[list[str]]]] = {
    "superanimal_quadruped": (BODYPARTS_BOTTOMVIEW, SKELETON_BOTTOMVIEW),
    "superanimal_topviewmouse": (BODYPARTS_TOPVIEW, SKELETON_TOPVIEW),
}


# ----------------------------------------------------------------------
# Transfer learning (02_train.py)
# ----------------------------------------------------------------------

# SuperAnimal à utiliser comme base de transfer learning.
#   - "superanimal_quadruped"      : vue générique quadrupède (par
#     défaut). Recommandé pour bottomview et pour tout setup où les
#     pattes sont visibles.
#   - "superanimal_topviewmouse"   : spécifique top-view rongeurs.
#     À privilégier pour un setup vue de dessus où les pattes ne sont
#     jamais visibles (arène ouverte classique).
SUPERANIMAL_NAME = "superanimal_quadruped"

# Jeu de keypoints correspondant à la vue choisie ci-dessus. C'est ce que
# 01_setup_project écrit dans le config.yaml DLC. Pour une anatomie
# maison, édite le jeu concerné plus haut — ou écrase les deux lignes
# ci-dessous par tes propres listes.
try:
    DEFAULT_BODYPARTS, DEFAULT_SKELETON = KEYPOINTS_PAR_VUE[SUPERANIMAL_NAME]
except KeyError:
    raise ValueError(
        f"SUPERANIMAL_NAME inconnu : {SUPERANIMAL_NAME!r}. "
        f"Attendus : {', '.join(KEYPOINTS_PAR_VUE)}."
    ) from None

MODEL_NAME = "hrnet_w32"
DETECTOR_NAME = "fasterrcnn_resnet50_fpn_v2"

# Architecture du modèle local. DOIT matcher MODEL_NAME, sinon size
# mismatch au chargement des poids pré-entraînés.
NET_TYPE = "hrnet_w32"


# ----------------------------------------------------------------------
# Training (02_train.py)
# ----------------------------------------------------------------------

# Standard settings — recommandation Tony : ne pas modifier les
# hyperparamètres d'entraînement, la tâche n'est pas assez spécifique
# pour justifier un tuning au-delà des défauts.
EPOCHS = 50


# ----------------------------------------------------------------------
# Inférence et visualisation (03_apply.py)
# ----------------------------------------------------------------------

# Seuil de confiance pour l'affichage des keypoints dans la vidéo
# annotée.
#   - 0.6 = défaut DLC, propre pour un modèle qui marche bien
#   - 0.3 = mode diagnostic, montre les prédictions incertaines
#     (utile pour voir où le modèle hésite entre plusieurs candidats)
#   - 0.1 = mode debug extrême, tout est affiché même les prédictions
#     à peu près aléatoires
LABELED_VIDEO_PCUTOFF = 0.6

# Génère ou non la vidéo annotée pendant l'inférence.
#   - True  : phase pilote / debug → indispensable pour le QC visuel
#   - False : production (juste produire les .h5) → gagne ~1× la durée
#     de la vidéo en calcul
MAKE_LABELED_VIDEO = True


# ----------------------------------------------------------------------
# Phase 2 — ajout de vidéos cross-mouse (04_add_videos.py)
# ----------------------------------------------------------------------

# Vidéos supplémentaires à ajouter au training set après le pilote,
# pour que le modèle apprenne des features anatomiques génériques au
# lieu de raccourcis spécifiques à la vidéo pilote.
#
# Recommandation Tony : « prendre autant d'animaux différents que
# possible ». L'objectif est de présenter au réseau la plus large
# variété de situations. Sur un projet à ~40 souris, viser 6-10
# animaux différents dans le training set final.
ADDITIONAL_VIDEOS: list[Path] = [
    # Path(r"D:\path\to\other_mouse_02.mp4"),
    # Path(r"D:\path\to\other_mouse_03.mp4"),
]

# Nombre de frames à extraire par nouvelle vidéo (k-means). Plus petit
# que N_AUTO_FRAMES parce qu'on a déjà la diversité posturale du pilote
# ici l'objectif est la diversité INTER-individus, pas intra.
NEW_VIDEO_FRAMES = 20


# ----------------------------------------------------------------------
# Phase 3 — refine outliers (05_refine_outliers.py)
# ----------------------------------------------------------------------

# Vidéos sur lesquelles chercher des outliers à re-labelliser. Doivent
# avoir été analysées via 03_apply.py au préalable (un .h5 doit exister
# dans <PROJECT_DIR>/result-videos/<stem>/).
#
# Recommandation Tony : « après le premier training, extraire les
# outliers MANUELLEMENT — pas via l'auto-detect. Tu vois exactement où
# le réseau échoue, tu peux choisir les frames les plus utiles. Vise
# 50-100 frames AU TOTAL (pas par situation), réparties entre les
# situations problématiques que tu as identifiées (rearing, occlusion,
# ambiguïté L/R...). Le nombre dépend de combien de situations
# distinctes posent problème. »
# Ce script sert de béquille pour attraper les cas évidents, mais le
# vrai gain vient d'une passe manuelle dans la GUI DLC.
TRAINING_VIDEOS_FOR_REFINE: list[Path] = [
    PILOT_VIDEO,
]

# Algorithme de détection d'outliers :
#   - "uncertain" : frames à likelihood basse — idéal pour cibler les
#     pattes occultées à leur émergence (recommandé si le failure mode
#     principal est un manque de couverture posturale)
#   - "jump"      : frames avec sauts inter-frame anormaux d'un
#     keypoint — utile pour les téléportations gauche/droite
#   - "fitting"   : modèle ARIMA (plus coûteux, moins ciblé)
OUTLIER_ALGORITHM = "uncertain"

# Seuil de "jump" en pixels au-dessus duquel un mouvement inter-frame
# est considéré anormal. Sur 1024×1080, 50 px est un compromis.
OUTLIER_EPSILON = 50

# Nombre max de frames extraites PAR VIDÉO. Garde-fou pour éviter
# d'extraire des centaines de frames à la volée.
OUTLIER_NUMFRAMES = 30


# ----------------------------------------------------------------------
# (Optionnel) Preprocessing MOG2 + CLAHE
# ----------------------------------------------------------------------

# Si tes vidéos sont bruitées (reflets IR, ombres dynamiques), tu peux
# créer un projet DLC parallèle qui consomme des vidéos pré-traitées.
# Ces valeurs ne sont utilisées que si tu passes par ce flow.
PREPROCESSED_VIDEO_DIR = Path(r"E:\path\to\preprocessed_videos")
PREPROCESSED_PROJECT_DIR = WORKDIR / f"{PROJECT_NAME}-{EXPERIMENTER}-preproc"

# Paramètres MOG2 (background subtraction adaptative)
MOG2_HISTORY = 500
MOG2_VAR_THRESHOLD = 16.0
MOG2_WARMUP = 1000
MOG2_LEARNING_RATE = 0.0001

# Paramètres CLAHE (contrast enhancement)
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_SIZE = 8
