"""Crée le projet DLC et extrait les frames à labelliser (end-to-end).

Cette version fait TOUT en une commande : elle crée le projet DLC, patche
son `config.yaml` avec la liste des bodyparts + skeleton + numframes2pick
définis dans `_config.py`, puis lance l'extraction k-means.

Workflow simplifié :
    1. python scripts/dlc_model-training/00_init_training_config.py
       (wizard → écrit ton _config.py custom dans un dossier hors du repo)
    2. python scripts/dlc_model-training/01_setup_project.py \\
           --config-dir <dossier créé à l'étape 1>
    3. Mets à jour PROJECT_DIR dans ton _config.py avec le nom exact
       du dossier créé (DLC ajoute la date au nom du projet)
    4. dlc.label_frames(CONFIG) dans une session Python pour labelliser
       (ou GUI : python -c "import deeplabcut; deeplabcut.launch_dlc()")

Sans `--config-dir`, le script utilise le _config.py template du repo
(valeurs par défaut, à éditer manuellement avant de lancer).

Pré-requis : conda activate dlc, DeepLabCut 3.x + PyTorch, vidéo pilote mp4.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _load_config(config_dir: Path | None):
    """Insère `config_dir` (ou le dossier du script) en tête de sys.path
    pour que `from _config import ...` cible la bonne copie.
    """
    if config_dir is not None:
        cd = config_dir.resolve()
        if not (cd / "_config.py").exists():
            print(f"❌ _config.py introuvable dans {cd}\n"
                  f"   Lance d'abord : python scripts/dlc_model-training/"
                  f"00_init_training_config.py", file=sys.stderr)
            sys.exit(1)
        sys.path.insert(0, str(cd))
    else:
        sys.path.insert(0, str(Path(__file__).resolve().parent))


def patch_dlc_config(config_path: Path, bodyparts: list[str],
                      skeleton: list[list[str]], n_frames: int) -> None:
    """Écrit bodyparts + skeleton + numframes2pick dans le config.yaml DLC.

    Évite au user d'ouvrir manuellement le fichier. Utilise `ruamel.yaml`
    si dispo (préserve les commentaires DLC), sinon retombe sur PyYAML.
    """
    try:
        from ruamel.yaml import YAML
        yaml = YAML()
        yaml.preserve_quotes = True
        with open(config_path, encoding="utf-8") as f:
            cfg = yaml.load(f)
        cfg["bodyparts"] = bodyparts
        cfg["skeleton"] = [list(edge) for edge in skeleton]
        cfg["numframes2pick"] = n_frames
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(cfg, f)
    except ImportError:
        import yaml
        with open(config_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        cfg["bodyparts"] = bodyparts
        cfg["skeleton"] = [list(edge) for edge in skeleton]
        cfg["numframes2pick"] = n_frames
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--config-dir", type=Path, default=None,
        help="Dossier contenant ton _config.py custom (produit par "
             "00_init_training_config.py). Sans ce flag, utilise le "
             "template dans scripts/dlc_model-training/.",
    )
    parser.add_argument(
        "--skip-extract", action="store_true",
        help="Ne lance pas l'extraction k-means à la fin (utile si tu "
             "veux d'abord vérifier le config.yaml patché).",
    )
    args = parser.parse_args()

    _load_config(args.config_dir)
    import deeplabcut as dlc  # noqa: E402 — après sys.path setup
    from _config import (  # noqa: E402
        DEFAULT_BODYPARTS,
        DEFAULT_SKELETON,
        EXPERIMENTER,
        N_AUTO_FRAMES,
        PILOT_VIDEO,
        PROJECT_NAME,
        WORKDIR,
    )

    if not PILOT_VIDEO.exists():
        raise FileNotFoundError(
            f"Vidéo pilote introuvable : {PILOT_VIDEO}\n"
            "Vérifie la constante PILOT_VIDEO dans ton _config.py."
        )

    print(f"Création du projet '{PROJECT_NAME}' dans {WORKDIR}/")
    config_path = dlc.create_new_project(
        project=PROJECT_NAME,
        experimenter=EXPERIMENTER,
        videos=[str(PILOT_VIDEO)],
        working_directory=str(WORKDIR),
        copy_videos=True,
    )
    config_path = Path(config_path)
    print(f"✅ Projet créé : {config_path}")

    # ---- Auto-patch du config.yaml DLC ----
    print(f"\nPatch du config.yaml :")
    print(f"  · bodyparts       = {len(DEFAULT_BODYPARTS)} keypoints")
    print(f"  · skeleton        = {len(DEFAULT_SKELETON)} liaisons")
    print(f"  · numframes2pick  = {N_AUTO_FRAMES}")
    patch_dlc_config(config_path, DEFAULT_BODYPARTS, DEFAULT_SKELETON,
                      N_AUTO_FRAMES)
    print(f"  → OK\n")

    project_dir_name = config_path.parent.name
    print(f"⚠  Mets à jour PROJECT_DIR dans ton _config.py :")
    print(f"     PROJECT_DIR = WORKDIR / \"{project_dir_name}\"\n")

    # ---- Extraction k-means ----
    if args.skip_extract:
        print("--skip-extract activé, pas d'extraction k-means. "
              "Lance-la à la main plus tard :")
        print(f"     dlc.extract_frames(r\"{config_path}\", mode=\"automatic\", "
              f"algo=\"kmeans\", crop=False, userfeedback=False)")
    else:
        print(f"Extraction de {N_AUTO_FRAMES} frames k-means "
              f"(peut prendre 2-5 min)...")
        dlc.extract_frames(
            str(config_path),
            mode="automatic",
            algo="kmeans",
            crop=False,
            userfeedback=False,
        )
        print(f"✅ Frames extraites dans {config_path.parent / 'labeled-data'}")

    print()
    print("Étapes suivantes :")
    print("  1. Ouvre la GUI DLC :")
    print("     python -c \"import deeplabcut; deeplabcut.launch_dlc()\"")
    print("  2. Charge le config.yaml ci-dessus dans la GUI")
    print("  3. Labellise les frames (compter ~1 min par frame)")
    print("  4. Lance 02_train.py")


if __name__ == "__main__":
    main()
