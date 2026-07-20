"""Ajoute des vidéos supplémentaires au projet DLC (phase 2 cross-mouse).

But : enrichir le dataset d'entraînement avec d'autres souris pour que le
modèle apprenne à généraliser. Sur un seul pilote de 60 frames, DLC tend
à mémoriser des features parasites du décor (coins, reflets IR fixes…)
parce que tout l'arrière-plan est identique d'une frame à l'autre. Avec
4-6 souris différentes en plus, les éléments parasites changent ou
disparaissent et le modèle est forcé d'apprendre la silhouette de la
souris elle-même.

Workflow phase 2 :

  1. Édite ADDITIONAL_VIDEOS dans _config.py avec les nouvelles vidéos.
  2. (Optionnel) règle NEW_VIDEO_FRAMES dans _config.py (défaut 20).
  3. AVANT de labelliser : vérifie ta convention L/R sur la vidéo pilote
     (dlc.check_labels(CONFIG) et inspection visuelle). Si tu as inversé
     L/R sur certaines frames de la vidéo 1, corrige AVANT d'en accumuler
     de nouvelles sur le même biais.
  4. (env: dlc) python scripts/dlc_model-training/04_add_videos.py
  5. Labellise les nouvelles frames :
         import deeplabcut as dlc
         from _config import CONFIG
         dlc.label_frames(CONFIG)
  6. Relance python scripts/dlc_model-training/02_train.py

Ce script :
  - met à jour `numframes2pick` dans le config.yaml DU PROJET DLC pour
    que extract_frames sorte NEW_VIDEO_FRAMES frames par nouvelle vidéo
    (sans ça, il piocherait N_AUTO_FRAMES = 60 sur chaque, ce qui est
    surdimensionné en phase 2)
  - appelle dlc.add_new_videos pour registrer les vidéos dans le projet
  - appelle dlc.extract_frames en mode kmeans automatique
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

# Insère le dossier du script en tête de sys.path pour trouver _load_config
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _load_config import add_config_dir_arg, load_config  # noqa: E402


def update_numframes2pick(project_config_path: Path, n: int) -> int:
    """Met à jour `numframes2pick` dans le config.yaml du projet DLC.

    Retourne l'ancienne valeur (utile pour info ou restore manuel).
    """
    with open(project_config_path) as f:
        cfg = yaml.safe_load(f)
    old = cfg.get("numframes2pick", 20)
    if old == n:
        return old
    cfg["numframes2pick"] = n
    with open(project_config_path, "w") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
    return old


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    add_config_dir_arg(parser)
    args = parser.parse_args()
    load_config(args)

    import deeplabcut as dlc  # noqa: E402
    from _config import (  # noqa: E402
        ADDITIONAL_VIDEOS, CONFIG, NEW_VIDEO_FRAMES, PROJECT_DIR,
    )

    if not ADDITIONAL_VIDEOS:
        print(
            "⚠ ADDITIONAL_VIDEOS est vide dans _config.py.\n"
            "Édite la liste avec 4-6 nouvelles vidéos de souris différentes,\n"
            "puis relance ce script."
        )
        return

    # Vérifie l'existence des vidéos AVANT toute modification du projet
    missing = [v for v in ADDITIONAL_VIDEOS if not v.exists()]
    if missing:
        print("❌ Vidéos introuvables (corrige les chemins dans _config.py) :")
        for v in missing:
            print(f"   - {v}")
        sys.exit(1)

    project_config = Path(CONFIG)
    print(f"Projet DLC : {PROJECT_DIR}")
    print(f"Vidéos à ajouter ({len(ADDITIONAL_VIDEOS)}) :")
    for v in ADDITIONAL_VIDEOS:
        print(f"   - {v.name}")
    print(f"Frames à extraire par vidéo : {NEW_VIDEO_FRAMES}\n")

    # 1. Met à jour numframes2pick dans le config du projet DLC
    print("Mise à jour de numframes2pick dans config.yaml...")
    old = update_numframes2pick(project_config, NEW_VIDEO_FRAMES)
    if old == NEW_VIDEO_FRAMES:
        print(f"  numframes2pick déjà à {NEW_VIDEO_FRAMES}, pas de changement.\n")
    else:
        print(f"  numframes2pick : {old} → {NEW_VIDEO_FRAMES}\n")

    # 2. Ajoute les vidéos au projet (copie dans <projet>/videos/)
    print("Ajout des vidéos au projet (copie en cours)...")
    dlc.add_new_videos(
        CONFIG,
        [str(v) for v in ADDITIONAL_VIDEOS],
        copy_videos=True,
    )
    print("✅ Vidéos ajoutées au projet.\n")

    # 3. Extrait les frames automatiquement (kmeans)
    print(f"Extraction de {NEW_VIDEO_FRAMES} frames par nouvelle vidéo (kmeans)...")
    dlc.extract_frames(
        CONFIG,
        mode="automatic",
        algo="kmeans",
        crop=False,
        userfeedback=False,
    )
    print("✅ Frames extraites.\n")

    print(
        "Étapes suivantes :\n"
        "  1. Labellise les nouvelles frames (~30-40 min) :\n"
        "       conda activate dlc\n"
        "       python\n"
        "       >>> import deeplabcut as dlc\n"
        "       >>> import sys\n"
        "       >>> sys.path.insert(0, 'scripts/dlc_model-training')\n"
        "       >>> from _config import CONFIG\n"
        "       >>> dlc.label_frames(CONFIG)\n"
        "  2. Vérifie la convention L/R sur toutes les vidéos (Ctrl+S\n"
        "     régulier pendant la labellisation).\n"
        "  3. Relance le training sur le dataset enrichi :\n"
        "       python scripts/dlc_model-training/02_train.py\n"
    )


if __name__ == "__main__":
    main()
