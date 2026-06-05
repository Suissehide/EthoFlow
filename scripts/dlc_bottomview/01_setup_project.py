"""Crée le projet DLC bottom-view et extrait les frames à labelliser.

Workflow :
    1. Édite les constantes VIDEO_PILOTE et WORKDIR ci-dessous
    2. (env: dlc) python scripts/dlc_bottomview/01_setup_project.py
    3. Édite manuellement le config.yaml généré pour mettre la liste des
       12 bodyparts et le skeleton (voir docs/ETHOFLOW.md §10)
    4. Reprends ce script pour les extract_frames (ou décommente les lignes
       à la fin et relance)
    5. dlc.label_frames(CONFIG) dans une session Python pour labelliser

Pré-requis :
    - conda activate dlc
    - DeepLabCut 3.x avec PyTorch
    - Une vidéo pilote bottom-view au format mp4

Le projet créé peut être versionné dans Git séparément (voir
configs/pipeline_config.yaml clé `dlc_project_config` pour le câbler
au pipeline EthoFlow via `run_dlc_inference.py --mode custom`).
"""
from __future__ import annotations

from pathlib import Path

import deeplabcut as dlc


# ----------------------------------------------------------------------
# À ÉDITER
# ----------------------------------------------------------------------

VIDEO_PILOTE = Path("/chemin/vers/ta_video_pilote.mp4")
WORKDIR = Path.home() / "Inserm" / "dlc-projects"
PROJECT_NAME = "souris-bottomview"
EXPERIMENTER = "Leo"

# Nombre de frames extraites automatiquement (kmeans pour la diversité posturale).
# Édite aussi le `numframes2pick` dans config.yaml si tu veux > 30.
N_AUTO_FRAMES = 30


# ----------------------------------------------------------------------
# 1. Création du projet
# ----------------------------------------------------------------------

def main() -> None:
    if not VIDEO_PILOTE.exists():
        raise FileNotFoundError(
            f"Vidéo pilote introuvable : {VIDEO_PILOTE}\n"
            "Édite la constante VIDEO_PILOTE en tête de ce script."
        )

    print(f"Création du projet '{PROJECT_NAME}' dans {WORKDIR}/")
    config_path = dlc.create_new_project(
        project=PROJECT_NAME,
        experimenter=EXPERIMENTER,
        videos=[str(VIDEO_PILOTE)],
        working_directory=str(WORKDIR),
        copy_videos=True,
    )
    print(f"\n✅ Projet créé : {config_path}\n")

    print(
        "Étape suivante MANUELLE avant extract_frames :\n"
        f"  1. Ouvre {config_path}\n"
        "  2. Remplace la section `bodyparts:` par les 12 keypoints "
        "bottom-view (voir docs/ETHOFLOW.md §10)\n"
        "  3. Remplace la section `skeleton:` par les liaisons correspondantes\n"
        f"  4. (Optionnel) règle `numframes2pick: {N_AUTO_FRAMES}`\n"
    )

    # ------------------------------------------------------------------
    # 2. Extraction automatique des frames (kmeans → diversité)
    # ------------------------------------------------------------------
    # Décommente les lignes ci-dessous APRÈS avoir édité le config.yaml :
    #
    # dlc.extract_frames(
    #     config_path,
    #     mode="automatic",
    #     algo="kmeans",
    #     crop=False,
    #     userfeedback=False,
    # )
    # print(f"✅ {N_AUTO_FRAMES} frames extraites automatiquement.\n")
    #
    # # Phase manuelle pour ajouter ~15 frames de rearing (une GUI s'ouvre)
    # dlc.extract_frames(config_path, mode="manual")


if __name__ == "__main__":
    main()
