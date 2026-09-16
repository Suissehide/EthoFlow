"""Crée le training dataset avec transfer learning depuis SuperAnimal-Quadruped
et entraîne le modèle DLC bottom-view.

Pré-requis :
    - 01_setup_project.py exécuté
    - config.yaml édité avec les 12 bodyparts + skeleton
    - PROJECT_DIR mis à jour dans `_config.py`
    - Frames labellisées via `dlc.label_frames(CONFIG)` ou la GUI napari
    - conda activate dlc

Pourquoi transfer learning depuis Quadruped (pas TopViewMouse) :
    Quadruped voit les pattes pendant son entraînement (vue latérale/oblique de
    quadrupèdes), TopViewMouse non. Pour bottom-view où les pattes sont les
    keypoints centraux, Quadruped donne un meilleur backbone de départ.
    `with_decoder=False` = on garde les features bas-niveau Quadruped mais on
    entraîne un nouveau décodeur pour NOS 12 keypoints custom.

Piège connu :
    `NET_TYPE` dans _config.py DOIT matcher `MODEL_NAME` (les deux à
    "hrnet_w32"), sinon size mismatch au chargement des poids.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Insère le dossier du script en tête de sys.path pour trouver _load_config
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _load_config import add_config_dir_arg, load_config  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    add_config_dir_arg(parser)
    parser.add_argument(
        "--eval-only", action="store_true",
        help="N'entraîne pas : évalue les snapshots déjà présents et "
             "produit les images annotées. Quelques minutes. À utiliser "
             "pour juger un modèle dont l'entraînement a planté en route, "
             "AVANT de relancer 200 epochs. Ne touche pas au training "
             "dataset (le recréer effacerait les snapshots).",
    )
    args = parser.parse_args()
    load_config(args)

    import deeplabcut as dlc  # noqa: E402 — après load_config
    from deeplabcut.modelzoo import build_weight_init  # noqa: E402
    from _dlc_patches import apply_patches  # noqa: E402
    from _config import (  # noqa: E402
        CONFIG, DETECTOR_NAME, EPOCHS, MODEL_NAME, NET_TYPE, SUPERANIMAL_NAME,
    )

    # Sans ça, l'évaluation saute TOUTES les images annotées sur un projet
    # mono-animal ("DataFrame reshape failed"). Détails dans _dlc_patches.
    apply_patches()

    if args.eval_only:
        # create_training_dataset régénère le shuffle et fait disparaître
        # les snapshots existants : on ne l'appelle surtout pas ici.
        print("Mode --eval-only : ni dataset ni entraînement, "
              "on évalue ce qui est déjà sur le disque.\n")
    else:
        print(f"Préparation des poids initiaux depuis {SUPERANIMAL_NAME}...")
        weight_init = build_weight_init(
            cfg=CONFIG,
            super_animal=SUPERANIMAL_NAME,
            model_name=MODEL_NAME,
            detector_name=DETECTOR_NAME,
            with_decoder=False,
        )

        print(f"Création du training dataset (architecture {NET_TYPE})...")
        dlc.create_training_dataset(
            CONFIG,
            weight_init=weight_init,
            net_type=NET_TYPE,  # IMPORTANT : doit matcher MODEL_NAME des poids
        )

        print(f"Entraînement ({EPOCHS} epochs, transfer learning actif)...")
        dlc.train_network(
            CONFIG,
            superanimal_name=SUPERANIMAL_NAME,
            superanimal_transfer_learning=True,
            epochs=EPOCHS,
        )
        print("✅ Entraînement terminé.\n")

    print("Évaluation du modèle (produit des images annotées)...")
    dlc.evaluate_network(CONFIG, plotting=True)
    print("✅ Évaluation terminée.\n")

    print(
        "Cible : test rmse_pcutoff < 8 px (sur 1024×1080, ~0.8 %).\n"
        "Si > 15 px, soit labellise plus de frames, soit revérifie la\n"
        "cohérence des annotations (utilise `dlc.check_labels(CONFIG)`)."
    )
    print(
        "\nrmse_pcutoff = nan signifie qu'AUCUNE prédiction n'a dépassé le\n"
        "seuil de confiance : le modèle n'est sûr de rien. Regarde les\n"
        "images annotées avant d'ajouter des epochs — elles sont dans\n"
        "<projet>/evaluation-results/iteration-0/.../LabeledImages_*/"
    )


if __name__ == "__main__":
    main()
