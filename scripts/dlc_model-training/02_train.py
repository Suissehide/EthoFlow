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


def reset_training_artifacts(project_dir: Path) -> None:
    """Supprime tout ce qu'un run produit, et rien de ce qu'il consomme.

    Ce qui part : `dlc-models-pytorch/` (snapshots + learning_stats.csv),
    `training-datasets/` (régénéré par create_training_dataset), et
    `evaluation-results/`.

    Ce qui reste : `labeled-data/` — tes annotations, le seul contenu
    vraiment coûteux à reproduire —, `config.yaml` et `videos/`.

    Motivation : après un run interrompu, les snapshots de l'ancien run
    cohabitent avec ceux du nouveau. `evaluate_network` les évalue tous
    (`snapshotindex: all`) et la table de résultats mélange deux runs.
    """
    import shutil

    # DLC suffixe ses dossiers selon le backend (`-pytorch` pour DLC 3.x),
    # donc on balaie les deux conventions plutôt que d'en supposer une.
    a_supprimer = [
        project_dir / "dlc-models-pytorch",
        project_dir / "dlc-models",
        project_dir / "training-datasets",
        *sorted(project_dir.glob("evaluation-results*")),
    ]
    preserves = project_dir / "labeled-data"
    n_labels = 0
    if preserves.exists():
        n_labels = sum(len(list(d.glob("CollectedData_*.h5")))
                       for d in preserves.iterdir() if d.is_dir())

    print("--reset : remise à zéro de l'entraînement")
    print(f"  conservé : labeled-data/ ({n_labels} fichier(s) "
          f"d'annotations), config.yaml, videos/")
    for p in a_supprimer:
        if not p.exists():
            continue
        shutil.rmtree(p)
        print(f"  supprimé : {p.name}/")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    add_config_dir_arg(parser)
    parser.add_argument(
        "--reset", action="store_true",
        help="Repart d'un entraînement propre : supprime les snapshots, "
             "learning_stats.csv et les résultats d'évaluation de "
             "l'itération courante avant de réentraîner. NE TOUCHE PAS à "
             "labeled-data/ (tes annotations), config.yaml ni videos/. "
             "À utiliser après un run interrompu, pour ne pas mélanger "
             "les snapshots de deux runs.",
    )
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

    if args.reset:
        if args.eval_only:
            print("❌ --reset et --eval-only sont contradictoires : le "
                  "premier efface ce que le second veut évaluer.",
                  file=sys.stderr)
            sys.exit(1)
        reset_training_artifacts(Path(CONFIG).parent)

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
