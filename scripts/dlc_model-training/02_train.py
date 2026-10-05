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


def envergure_animal(project_dir: Path, percentile: float = 90.0
                      ) -> tuple[float, float] | None:
    """(médiane, centile) de l'envergure de l'animal dans les annotations.

    L'envergure est la plus grande distance entre deux keypoints d'une
    même frame. Elle dit la taille minimale que doit avoir la fenêtre
    d'entraînement pour contenir l'animal entier.

    Pourquoi ça compte au-delà du cadrage : un réseau ne peut décider
    qu'une patte est la gauche qu'en la situant par rapport à
    l'orientation tête-queue. Si la fenêtre ne contient pas la tête, la
    question n'a pas de réponse dans l'image, et l'entraînement pousse le
    réseau vers la position moyenne des deux pattes. Les keypoints
    proches de la tête n'en souffrent pas ; ceux de l'autre extrémité,
    si.
    """
    import numpy as np
    import pandas as pd

    fichiers = sorted((project_dir / "labeled-data").glob(
        "*/CollectedData_*.h5"))
    if not fichiers:
        return None
    try:
        df = pd.concat([pd.read_hdf(f) for f in fichiers])
    except Exception:
        return None

    scorer = df.columns.get_level_values(0)[0]
    bps = list(dict.fromkeys(df.columns.get_level_values("bodyparts")))
    xs = np.column_stack([df[(scorer, b, "x")].to_numpy(float) for b in bps])
    ys = np.column_stack([df[(scorer, b, "y")].to_numpy(float) for b in bps])

    # Envergure = diagonale de la boîte englobante des keypoints visibles.
    with np.errstate(invalid="ignore"):
        largeur = np.nanmax(xs, axis=1) - np.nanmin(xs, axis=1)
        hauteur = np.nanmax(ys, axis=1) - np.nanmin(ys, axis=1)
    env = np.hypot(largeur, hauteur)
    env = env[np.isfinite(env)]
    if not env.size:
        return None
    return float(np.median(env)), float(np.percentile(env, percentile))


def crop_recommande(envergure_centile: float, marge: float = 1.1) -> int:
    """Taille de fenêtre couvrant l'envergure, arrondie à un multiple de 32.

    Le multiple de 32 n'est pas cosmétique : `auto_padding` du config DLC
    impose `pad_width_divisor: 32`, donc une valeur qui n'en est pas un
    serait de toute façon rembourrée.
    """
    import math
    return int(math.ceil(envergure_centile * marge / 32) * 32)


def verifier_crop(project_dir: Path, taille_voulue: int | None = None) -> None:
    """Compare la fenêtre d'entraînement à la taille réelle de l'animal.

    Signale toujours, applique seulement si `taille_voulue` est donnée :
    agrandir la fenêtre augmente la mémoire GPU avec le carré de la
    taille, et faire déborder un entraînement de 24 h sans prévenir
    serait pire que le problème qu'on corrige.
    """
    import yaml as _yaml

    cfg_path = trouver_pytorch_config(project_dir)
    if cfg_path is None:
        return
    cfg = _yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    crop = ((cfg.get("data") or {}).get("train") or {}).get("crop_sampling")
    if not isinstance(crop, dict):
        return
    actuel = int(min(crop.get("width", 0), crop.get("height", 0)) or 0)

    mesure = envergure_animal(project_dir)
    if mesure is None:
        return
    mediane, centile = mesure
    recommande = crop_recommande(centile)

    print(f"   Animal (annotations) : {mediane:.0f} px de médiane, "
          f"{centile:.0f} px au 90e centile")
    print(f"   Fenêtre d'entraînement : {actuel} px")

    if taille_voulue is None:
        if actuel and actuel < centile:
            facteur = (recommande / actuel) ** 2
            print(f"   ⚠ La fenêtre est plus petite que l'animal : les "
                  f"keypoints des extrémités")
            print(f"     (queue, pattes arrière) en sortent souvent, et "
                  f"avec eux la tête qui sert")
            print(f"     de référence d'orientation — d'où des gauche/droite "
                  f"indécidables.")
            print(f"     Recommandé : --crop-size {recommande}  "
                  f"(≈ ×{facteur:.1f} de mémoire GPU, baisse le batch_size "
                  f"si ça déborde)")
        else:
            print(f"   ✓ La fenêtre contient l'animal entier.")
        return

    crop["width"] = int(taille_voulue)
    crop["height"] = int(taille_voulue)
    cfg_path.write_text(
        _yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False),
        encoding="utf-8")
    print(f"   ✓ Fenêtre portée à {taille_voulue} px "
          f"(couvre {'le' if taille_voulue >= centile else 'une partie du'} "
          f"90e centile)")


def regler_batch_size(project_dir: Path, taille: int) -> None:
    """Écrit batch_size dans pytorch_config.yaml, où qu'il se trouve.

    Agrandir la fenêtre d'entraînement multiplie la mémoire par le carré
    du facteur ; réduire le lot dans la même proportion la ramène au
    niveau initial. Le compromis est du temps par epoch, pas de la
    qualité : un lot plus petit converge aussi bien, parfois mieux.

    La clé vit selon les versions à la racine ou sous `train_settings` /
    `runner`, d'où la recherche plutôt qu'un chemin en dur.
    """
    import yaml as _yaml

    cfg_path = trouver_pytorch_config(project_dir)
    if cfg_path is None:
        return
    cfg = _yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}

    def poser(noeud) -> bool:
        if isinstance(noeud, dict):
            if "batch_size" in noeud:
                ancien = noeud["batch_size"]
                noeud["batch_size"] = int(taille)
                print(f"   ✓ batch_size : {ancien} → {taille}")
                return True
            return any(poser(v) for v in noeud.values())
        return False

    if poser(cfg):
        cfg_path.write_text(
            _yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False),
            encoding="utf-8")
    else:
        print("   ⚠ batch_size introuvable dans pytorch_config.yaml")


def paires_symetriques(bodyparts: list[str]) -> list[list[int]]:
    """Indices des keypoints symétriques, d'après leurs noms `_left`/`_right`."""
    index = {bp: i for i, bp in enumerate(bodyparts)}
    paires = []
    for bp, i in index.items():
        if bp.endswith("_left"):
            jumeau = bp[: -len("_left")] + "_right"
            if jumeau in index:
                paires.append([i, index[jumeau]])
    return sorted(paires)


def trouver_pytorch_config(project_dir: Path) -> Path | None:
    """pytorch_config.yaml du shuffle le plus récemment créé."""
    candidats = list(project_dir.glob(
        "dlc-models-pytorch/iteration-*/*/train/pytorch_config.yaml"))
    if not candidats:
        return None
    return max(candidats, key=lambda p: p.stat().st_mtime)


def corriger_hflip(project_dir: Path, bodyparts: list[str]) -> None:
    """Empêche l'augmentation miroir d'enseigner la confusion gauche/droite.

    Le problème : retourner une image horizontalement transforme
    visuellement une patte gauche en patte droite. Si les paires
    symétriques ne sont pas déclarées, l'étiquette, elle, ne suit pas —
    le réseau reçoit donc la consigne explicite que « gauche » et
    « droite » désignent la même chose, et il l'apprend très bien.

    Le symptôme final est sans ambiguïté : les deux marqueurs d'une paire
    tombent sur le même pixel à l'inférence, et le côté qui perd l'argmax
    récolte une confiance proche de zéro. Aucune quantité de frames
    supplémentaires ne corrige ça, puisque c'est l'entraînement lui-même
    qui enseigne la confusion.

    Deux réparations, par ordre de préférence :
      1. déclarer les symétries, si le schéma de config les accepte —
         le miroir devient alors une augmentation valide et utile ;
      2. sinon désactiver le miroir. On perd de la diversité
         d'augmentation, ce qui est toujours préférable à une
         supervision contradictoire.
    """
    import yaml as _yaml

    cfg_path = trouver_pytorch_config(project_dir)
    if cfg_path is None:
        print("⚠  pytorch_config.yaml introuvable — vérification hflip "
              "sautée.")
        return

    cfg = _yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    train = (cfg.get("data") or {}).get("train")
    if not isinstance(train, dict) or "hflip" not in train:
        print("ℹ  Pas d'augmentation hflip dans ce config — rien à faire.")
        return

    hflip = train["hflip"]
    paires = paires_symetriques(bodyparts)

    if isinstance(hflip, dict):
        cle_sym = next((k for k in hflip if "symmetr" in k.lower()), None)
        if cle_sym and paires:
            if hflip.get(cle_sym):
                print(f"ℹ  hflip : symétries déjà déclarées "
                      f"({hflip[cle_sym]}).")
                return
            hflip[cle_sym] = paires
            action = (f"symétries déclarées sur {len(paires)} paire(s) "
                      f"→ {cle_sym}={paires}")
        else:
            train["hflip"] = False
            action = ("désactivé (le schéma n'expose pas de champ de "
                      "symétries)")
    elif hflip:
        train["hflip"] = False
        action = "désactivé (était actif, sans symétries possibles)"
    else:
        print("ℹ  hflip déjà désactivé — rien à faire.")
        return

    cfg_path.write_text(
        _yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False),
        encoding="utf-8")
    print(f"✓ Augmentation miroir corrigée : {action}")
    print(f"  ({cfg_path.relative_to(project_dir)})")


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

    # Un modèle entraîné représente 8 à 24 h de GPU. S'il existe, on ne le
    # supprime pas sans le dire explicitement : l'utilisateur lance souvent
    # --reset en pensant « nettoyer », pas « jeter le modèle ».
    snapshots = []
    for racine in ("dlc-models-pytorch", "dlc-models"):
        d = project_dir / racine
        if d.exists():
            snapshots += list(d.rglob("snapshot-*.pt"))
            snapshots += list(d.rglob("snapshot-*.index"))
    if snapshots:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from interactive import confirm  # noqa: E402

        print(f"⚠  {len(snapshots)} snapshot(s) entraîné(s) vont être "
              f"supprimés — le modèle sera à réentraîner de zéro "
              f"(8-24 h de GPU).")
        for s in sorted(snapshots)[-3:]:
            print(f"     · {s.name}")
        if not confirm("   Continuer ?", default="n"):
            print("Annulé — rien n'a été supprimé.")
            sys.exit(0)

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
        "--crop-size", type=int, default=None, metavar="N",
        help="Taille de la fenêtre d'entraînement (crop_sampling), en px. "
             "Doit contenir l'animal ENTIER : sinon les keypoints des "
             "extrémités en sortent, et la tête qui sert de référence "
             "d'orientation aussi, ce qui rend gauche/droite indécidable. "
             "Sans ce flag, le script mesure l'animal et affiche la valeur "
             "recommandée sans rien changer. Mémoire GPU ∝ N².",
    )
    parser.add_argument(
        "--batch-size", type=int, default=None, metavar="N",
        help="Taille de lot. À baisser quand on agrandit --crop-size : la "
             "mémoire GPU varie comme le carré de la fenêtre, et comme le "
             "lot. Passer de 448 à 768 px se compense en divisant le lot "
             "par 3. Coûte du temps par epoch, pas de la qualité.",
    )
    parser.add_argument(
        "--keep-hflip", action="store_true",
        help="Laisse l'augmentation miroir telle quelle. Par défaut le "
             "script déclare les paires symétriques (ou désactive le "
             "miroir), sans quoi le réseau apprend que gauche et droite "
             "désignent le même point.",
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

        # create_training_dataset vient de (re)générer pytorch_config.yaml :
        # c'est le seul moment où la correction tient, avant l'entraînement.
        if not args.keep_hflip:
            import yaml as _yaml
            bodyparts = (_yaml.safe_load(Path(CONFIG).read_text(
                encoding="utf-8")) or {}).get("bodyparts") or []
            corriger_hflip(Path(CONFIG).parent, list(bodyparts))
            print()
        verifier_crop(Path(CONFIG).parent, args.crop_size)
        if args.batch_size:
            regler_batch_size(Path(CONFIG).parent, args.batch_size)
        print()

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
