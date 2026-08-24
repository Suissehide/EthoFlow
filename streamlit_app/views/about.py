"""Page À propos."""
from __future__ import annotations

import streamlit as st

from lib.project import ROOT


def render() -> None:
    st.title("À propos")
    st.markdown(
        """
        **EthoFlow** est une interface web légère pour orchestrer l'analyse
        comportementale de souris avec **DeepLabCut** et **VAME**. Elle pilote
        les scripts de `scripts/` (chacun tournant dans son propre
        environnement conda — `dlc`, `pipeline` ou `vame`) sans dupliquer leur
        logique : l'app construit et lance la commande, affiche le suivi de
        job, et lit les fichiers produits.

        ## Les neuf pages du pipeline

        - **Projet** — ouvrir ou créer un projet, désigner le modèle DLC à
          utiliser, suivre l'avancement global.
        - **Données** — étapes 2-3 : synchroniser l'Excel maître et générer
          `metadata.yaml` par session.
        - **Vidéos & calibration** — étapes 4 et 6a : localiser/recadrer les
          vidéos, calibrer l'échelle px/cm et les arènes, QC visuel des
          trajectoires.
        - **Pose (DLC)** — étape 5 : lance `run_dlc_inference.py` (env `dlc`)
          pour produire les `.h5` de points-clés par session.
        - **Nettoyage** — étape 6b : post-traitement des trajectoires
          (interpolation, filtrage de vitesse, détection de blocages).
        - **VAME** — étape 7, présentée en stepper : `setup`, `align`,
          `trainset`, `train`, `evaluate`, `segment` (env `vame`).
        - **Motifs** — étape 8 : visionne les clips de chaque motif VAME et
          nomme les comportements. Voir « Où vivent les labels » ci-dessous.
        - **Analyses** — étape 9 : des motifs segmentés aux statistiques
          (`analyze_vame.py`), tableaux et figures affichés inline.
        - **Visualisations** — étape 9, rendus optionnels (motif_gif,
          manifold, dendrogramme de communautés) pour un papier ou un poster.

        Deux pages système complètent la navigation : **Configuration**
        (chemins de préférence, santé des environnements conda) et **À
        propos** (cette page).

        ## Où vivent les labels de motifs

        La page **Motifs** lit et écrit exclusivement
        `<projet>/data/vame/motif_labels.csv` (séparateur `;`, encodage
        `utf-8-sig`), généré par `run_vame.py motif-videos` /
        `motif-labels` et consommé par `analyze_vame.py`. Sans ce fichier,
        les figures affichent `motif_0`, `motif_1`, etc. au lieu d'un vrai
        nom de comportement.

        Une ancienne version de l'app écrivait les labels dans
        `<projet>/analysis/motif_labels_<algo>.yaml` — un fichier que rien
        en aval ne lisait, donc invisible pour `analyze_vame.py`. Ce format
        n'est plus produit ; s'il en reste sur un vieux projet, la page
        **Motifs** propose de reprendre la colonne `label` dans le nouveau
        CSV (rien n'est écrasé sans confirmation explicite).

        ## Entraîner un nouveau modèle DeepLabCut

        Pour entraîner un nouveau modèle DeepLabCut (Parcours B du README),
        tu n'utilises pas cette interface : la formation se fait **au terminal**
        via `scripts/dlc_model-training/`. L'app EthoFlow n'importe et utilise
        que des modèles préalablement entraînés.

        Consulte le [README](README.md) pour le détail complet du pipeline et
        [`docs/ETHOFLOW.md`](docs/ETHOFLOW.md) pour la documentation
        technique approfondie.

        ## Stack

        Python · Streamlit · OpenCV · DeepLabCut · VAME
        """
    )
    st.caption(f"Lancé depuis : `{ROOT}`")
