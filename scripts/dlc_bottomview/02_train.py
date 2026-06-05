"""Crée le training dataset avec transfer learning depuis SuperAnimal-Quadruped
et entraîne le modèle DLC bottom-view.

Pré-requis :
    - Avoir lancé 01_setup_project.py
    - Avoir édité config.yaml avec les 12 bodyparts
    - Avoir labellisé les frames via `dlc.label_frames(CONFIG)` dans une
      session Python (ou via la GUI : python -m deeplabcut)
    - conda activate dlc

Pourquoi transfer learning depuis Quadruped et pas TopViewMouse :
    Quadruped voit les pattes (vue latérale ou oblique de quadrupèdes), pas
    TopViewMouse. Pour un setup bottom-view où les pattes sont les keypoints
    centraux, Quadruped donne un meilleur backbone de départ.
    `with_decoder=False` = on garde les features bas-niveau Quadruped mais
    on entraîne un nouveau décodeur pour NOS 12 keypoints custom.
"""
from __future__ import annotations

import deeplabcut as dlc
from deeplabcut.modelzoo import build_weight_init


# ----------------------------------------------------------------------
# À ÉDITER
# ----------------------------------------------------------------------

# Chemin absolu vers le config.yaml généré par 01_setup_project.py
CONFIG = "/chemin/absolu/vers/souris-bottomview-Leo-2026-06-XX/config.yaml"

# 50 epochs suffisent grâce au transfer learning (vs ~200k iterations
# en from-scratch).
EPOCHS = 50


# ----------------------------------------------------------------------
# Pipeline
# ----------------------------------------------------------------------

def main() -> None:
    print("Préparation des poids initiaux depuis SuperAnimal-Quadruped...")
    weight_init = build_weight_init(
        cfg=CONFIG,
        super_animal="superanimal_quadruped",
        model_name="hrnet_w32",
        detector_name="fasterrcnn_resnet50_fpn_v2",
        with_decoder=False,
    )

    print("Création du training dataset...")
    dlc.create_training_dataset(CONFIG, weight_init=weight_init)

    print(f"Entraînement ({EPOCHS} epochs, transfer learning actif)...")
    dlc.train_network(
        CONFIG,
        superanimal_name="superanimal_quadruped",
        superanimal_transfer_learning=True,
        epochs=EPOCHS,
    )
    print("✅ Entraînement terminé.\n")

    print("Évaluation du modèle (produit des images annotées)...")
    dlc.evaluate_network(CONFIG, plotting=True)
    print("✅ Évaluation terminée.\n")

    print(
        "Cible : erreur test < 8 pixels (sur 1024×1080, ~0.8 %).\n"
        "Si > 15 pixels, soit labellise plus de frames, soit revérifie\n"
        "la cohérence des annotations (cf. LABELLING_GUIDE.md à créer)."
    )


if __name__ == "__main__":
    main()
