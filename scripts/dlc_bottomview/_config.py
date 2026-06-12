"""Configuration centralisée pour les scripts DLC bottom-view.

Modifie ce fichier une seule fois — les scripts 01, 02, 03 et les éventuels
scripts d'analyse à venir importent depuis ici. Ça évite de copier-coller des
chemins absolus partout.

Conventions :
- Les chemins Windows utilisent des raw strings (r"...") pour éviter de
  doubler les backslashes.
- `CONFIG` est un str (et pas un Path) parce que DeepLabCut l'attend en str.
"""
from __future__ import annotations

from pathlib import Path


# ----------------------------------------------------------------------
# Projet DLC
# ----------------------------------------------------------------------

PROJECT_NAME = "souris-bottomview"
EXPERIMENTER = "Leo"
WORKDIR = Path(r"E:\LEO\dlc-projects")

# Dossier du projet une fois `01_setup_project.py` exécuté.
# DLC ajoute le nom de l'expérimentateur et la date à la création.
# Mets à jour ce nom après le premier setup.
PROJECT_DIR = WORKDIR / "souris-bottomview-Leo-2026-06-05"
CONFIG = str(PROJECT_DIR / "config.yaml")


# ----------------------------------------------------------------------
# Vidéos
# ----------------------------------------------------------------------

# Vidéo principale utilisée pour le pilote et les premiers tests.
PILOT_VIDEO = Path(
    r"D:\ETHOVISION\202606005-MCCfemellescapto-bottomIR\Media Files\970.mp4"
)

# Liste des vidéos à analyser par `03_apply.py`. Peut contenir PILOT_VIDEO
# seul au début, puis s'enrichir des autres souris quand tu les auras.
VIDEOS_TO_ANALYZE: list[Path] = [
    PILOT_VIDEO,
    # Ajoute les autres vidéos ici, par exemple :
    # Path(r"D:\ETHOVISION\...\autre_video.mp4"),
]


# ----------------------------------------------------------------------
# Sortie
# ----------------------------------------------------------------------

# Les outputs d'inférence (h5, csv, vidéo annotée) atterrissent dans
# <PROJECT_DIR>/result-videos/<nom_vidéo>/. Un dossier par vidéo source =
# pas de mélange entre runs.
RESULTS_DIR = PROJECT_DIR / "result-videos"


# ----------------------------------------------------------------------
# Extraction de frames (01_setup_project.py)
# ----------------------------------------------------------------------

# Nombre de frames extraites automatiquement par kmeans.
# 60 = sweet spot pour un pilote single-video. À monter si tu ajoutes
# plusieurs souris au projet (15-20 frames par souris supplémentaire).
N_AUTO_FRAMES = 60


# ----------------------------------------------------------------------
# Transfer learning (02_train.py)
# ----------------------------------------------------------------------

# Pour la souris bottom-view, on transfer learning depuis Quadruped
# qui voit les pattes pendant son entraînement (contrairement à
# TopViewMouse qui ne les voit jamais).
SUPERANIMAL_NAME = "superanimal_quadruped"
MODEL_NAME = "hrnet_w32"
DETECTOR_NAME = "fasterrcnn_resnet50_fpn_v2"

# Architecture du modèle local. DOIT matcher MODEL_NAME, sinon size mismatch
# au chargement des poids (cf. bug ResNet50 vs HRNet rencontré au 1er run).
NET_TYPE = "hrnet_w32"


# ----------------------------------------------------------------------
# Training (02_train.py)
# ----------------------------------------------------------------------

EPOCHS = 50


# ----------------------------------------------------------------------
# Inférence et visualisation (03_apply.py)
# ----------------------------------------------------------------------

# Seuil de confiance pour l'affichage des keypoints dans la vidéo annotée.
# - 0.6  = défaut DLC, propre pour un modèle qui marche bien
# - 0.1  = mode diagnostic, affiche TOUTES les prédictions même non confiantes
#   (utile quand le modèle semble « ne rien détecter » : tu vois s'il prédit
#    quelque chose au mauvais endroit ou rien du tout)
LABELED_VIDEO_PCUTOFF = 0.6

# Génère ou non la vidéo annotée.
# - True  = phase pilote / debug → INDISPENSABLE pour le QC visuel
# - False = production (juste produire les .h5 pour VAME) → gagne ~1× la
#           durée de la vidéo en calcul
MAKE_LABELED_VIDEO = True


# ----------------------------------------------------------------------
# Phase 2 — ajout de vidéos cross-mouse (04_add_videos.py)
# ----------------------------------------------------------------------

# Vidéos supplémentaires à ajouter au training set après le pilote, pour
# que le modèle apprenne des features anatomiques de la souris au lieu
# de raccourcis spécifiques au décor de la vidéo pilote.
#
# 4-6 souris différentes suffisent largement pour le pilote phase 2,
# pas la peine de tout mettre d'un coup. Tu peux re-lancer 04_add_videos
# plus tard avec d'autres souris si besoin.
ADDITIONAL_VIDEOS: list[Path] = [
    # Path(r"D:\ETHOVISION\...\souris02.mp4"),
    # Path(r"D:\ETHOVISION\...\souris03.mp4"),
    # Path(r"D:\ETHOVISION\...\souris04.mp4"),
    # Path(r"D:\ETHOVISION\...\souris05.mp4"),
]

# Nombre de frames à extraire par nouvelle vidéo. Plus petit que
# N_AUTO_FRAMES (60) parce qu'on a déjà la diversité posturale du pilote ;
# ici l'objectif est la diversité INTER-individus, pas intra.
NEW_VIDEO_FRAMES = 20


# ----------------------------------------------------------------------
# Phase 3 — refine outliers (05_refine_outliers.py)
# ----------------------------------------------------------------------

# Vidéos sur lesquelles chercher des outliers à re-labelliser. Doivent
# avoir été analysées via 03_apply.py au préalable (un .h5 doit exister
# dans <PROJECT_DIR>/result-videos/<stem>/). En général ce sont les
# vidéos déjà au training set, pour cibler les pattes que le modèle
# galère à placer SUR DES DONNÉES QU'IL CONNAÎT (signe clair d'un manque
# d'exemples plutôt que d'un manque de généralisation).
TRAINING_VIDEOS_FOR_REFINE: list[Path] = [
    PILOT_VIDEO,
    # Ajoute ici les vidéos de phase 2 (971-980) une fois analysées :
    # Path(r"D:\ETHOVISION\202606005-MCCfemellescapto-bottomIR\Media Files\971.mp4"),
    # ...
]

# Algorithme de détection d'outliers :
# - "jump" : détecte les sauts inter-frame d'un keypoint (très efficace
#   pour les pattes qui flickent — exactement ce qu'on veut corriger).
# - "fitting" : modèle ARIMA, plus coûteux et moins ciblé pour notre cas.
# - "uncertain" : prend les frames à likelihood basse (utile mais souvent
#   redondant avec "jump").
OUTLIER_ALGORITHM = "jump"

# Seuil de "jump" en pixels au-dessus duquel un mouvement inter-frame
# est considéré anormal. Sur 1024×1080, 50 px est un compromis : assez
# bas pour attraper les pattes qui sautent, assez haut pour ignorer les
# mouvements légitimes du nez/tête.
OUTLIER_EPSILON = 50

# Nombre max de frames extraites PAR VIDÉO. Garde-fou : sinon DLC peut
# en extraire des centaines. 30-40 / vidéo × 6 vidéos = 180-240 frames à
# corriger, ~30-45 min de boulot dans la GUI.
OUTLIER_NUMFRAMES = 30
