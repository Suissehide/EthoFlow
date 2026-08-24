"""Page À propos."""
from __future__ import annotations

import streamlit as st

from lib.project import ROOT


def render() -> None:
    st.title("À propos")
    st.markdown(
        """
        **EthoFlow** est une interface web légère pour orchestrer l'analyse
        comportementale de souris avec **DeepLabCut** et **VAME**.

        ## Nouveautés v2

        - **Pipeline structuré** : la page *Lancer pipeline* est segmentée en
          DLC / post-DLC / VAME / Analyse, avec un bouton dédié par sous-commande.
        - **Labellisation des motifs** : nouvelle page pour visionner chaque
          `cluster_video` et attribuer un label éthologique, persisté dans
          `<projet>/analysis/motif_labels_<algo>.yaml` (consommé par
          `analyze_vame.py --labels`).
        - **Support empty-arena** : intégration des options `--validity-source`,
          `--mask-empty`, `--min-edge-frames` et du script `trim_empty_arena.py`
          (voir ETHOFLOW.md §6.6).
        - **Page Résultats refondue** : affiche inline la heatmap, les plots
          par condition et les CSV produits par `analyze_vame.py`.
        - **Sélecteur de projet VAME** : tu peux pointer plusieurs projets dans
          `~/Inserm/vame-projects/` (configurable dans la sidebar) et basculer.

        ## Sources et documentation

        - Documentation complète : [`docs/ETHOFLOW.md`](docs/ETHOFLOW.md)
          - §6.6 — Détection et exclusion des artefacts empty-arena
          - §6.7 — Labellisation des motifs (vocabulaire standard)
          - §6.8 — Notes hyperparamètres VAME (overfit, régularisation)
        - Source de vérité expérimentale : fichier Excel maître
          (`OpenField_trials_*.xlsx`)
        - Workflow : Excel → sync → metadata.yaml → DLC → post-DLC → VAME → analyse

        ## Stack

        Python · Streamlit · OpenCV · DeepLabCut · VAME
        """
    )
    st.caption(f"Lancé depuis : `{ROOT}`")
