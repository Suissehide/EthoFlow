"""Crée le projet DLC bottom-view et extrait les frames à labelliser.

Workflow :
    1. Édite les constantes dans `_config.py` (PILOT_VIDEO, WORKDIR, etc.)
    2. (env: dlc) python scripts/dlc_model-training/01_setup_project.py
    3. Édite manuellement le config.yaml généré pour mettre la liste des
       12 bodyparts et le skeleton (voir docs/ETHOFLOW.md §10)
    4. Mets à jour `PROJECT_DIR` dans _config.py avec le nom exact du
       projet créé (DLC ajoute la date)
    5. Décommente les extract_frames à la fin du script et relance,
       OU lance-les en interactif depuis Python
    6. dlc.label_frames(CONFIG) dans une session Python pour labelliser

Pré-requis : conda activate dlc, DeepLabCut 3.x + PyTorch, vidéo pilote mp4.
"""
from __future__ import annotations

import sys
from pathlib import Path

import deeplabcut as dlc

# Import du config centralisé (situé dans le même dossier)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _config import EXPERIMENTER, N_AUTO_FRAMES, PILOT_VIDEO, PROJECT_NAME, WORKDIR  # noqa: E402


def main() -> None:
    if not PILOT_VIDEO.exists():
        raise FileNotFoundError(
            f"Vidéo pilote introuvable : {PILOT_VIDEO}\n"
            "Édite la constante PILOT_VIDEO dans _config.py."
        )

    print(f"Création du projet '{PROJECT_NAME}' dans {WORKDIR}/")
    config_path = dlc.create_new_project(
        project=PROJECT_NAME,
        experimenter=EXPERIMENTER,
        videos=[str(PILOT_VIDEO)],
        working_directory=str(WORKDIR),
        copy_videos=True,
    )
    print(f"\n✅ Projet créé : {config_path}\n")

    print(
        "Étapes MANUELLES avant extract_frames :\n"
        f"  1. Ouvre {config_path}\n"
        "  2. Remplace `bodyparts:` par les 12 keypoints bottom-view\n"
        "     (cf. docs/ETHOFLOW.md §10)\n"
        "  3. Remplace `skeleton:` par les 11 liaisons correspondantes\n"
        f"  4. Règle `numframes2pick: {N_AUTO_FRAMES}` si tu veux plus que 20 par défaut\n"
        "  5. Mets à jour `PROJECT_DIR` dans `_config.py` avec le nom\n"
        f"     exact du dossier créé : {Path(config_path).parent.name}\n"
    )

    # ------------------------------------------------------------------
    # Extraction automatique des frames (kmeans → diversité)
    # ------------------------------------------------------------------
    # Décommente APRÈS avoir édité config.yaml :
    #
    # dlc.extract_frames(
    #     config_path,
    #     mode="automatic",
    #     algo="kmeans",
    #     crop=False,
    #     userfeedback=False,
    # )
    # print(f"✅ {N_AUTO_FRAMES} frames extraites automatiquement.")


if __name__ == "__main__":
    main()
