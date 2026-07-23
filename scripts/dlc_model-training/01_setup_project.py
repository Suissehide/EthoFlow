"""Crée le projet DLC et extrait les frames à labelliser (end-to-end).

Cette version fait TOUT en une commande : elle crée le projet DLC, patche
son `config.yaml` avec la liste des bodyparts + skeleton + numframes2pick
définis dans `_config.py`, puis lance l'extraction k-means.

Workflow simplifié :
    1. python scripts/dlc_model-training/00_init_training_config.py
       (wizard → écrit ton _config.py custom)
    2. python scripts/dlc_model-training/01_setup_project.py \\
           --config-dir <dossier créé à l'étape 1>
    3. dlc.label_frames(CONFIG) dans une session Python pour labelliser
       (ou GUI : python -c "import deeplabcut; deeplabcut.launch_dlc()")

À l'étape 2, ce script :
    - crée le projet DLC dans WORKDIR
    - strip le suffixe date que DLC ajoute au nom du dossier (le résultat
      matche exactement <PROJECT_NAME>-<EXPERIMENTER>/, déterministe)
    - patche le config.yaml de DLC avec DEFAULT_BODYPARTS +
      DEFAULT_SKELETON + numframes2pick
    - extrait N_AUTO_FRAMES frames en k-means automatique

Rien à éditer manuellement dans _config.py à la fin.

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


def merge_dlc_project_into_config_dir(dlc_config_path: Path,
                                        config_dir: Path,
                                        project_name: str,
                                        experimenter: str) -> Path:
    """Merge le contenu du dossier daté créé par DLC dans le dossier de config.

    DLC vient de créer un dossier `<WORKDIR>/<project_name>-<exp>-<date>/`.
    On ne veut pas de ce dossier daté : le _config.py écrit par le wizard
    vit dans `<WORKDIR>/<project_name>/` (== config_dir), et on souhaite
    que le projet DLC vive dans le MÊME dossier — donc un seul dossier
    propre par projet.

    Ce helper :
      1. déplace chaque fichier/sous-dossier du dossier daté vers config_dir
      2. supprime le dossier daté (maintenant vide)
      3. met à jour toutes les occurrences de l'ancien chemin dans
         `config.yaml` (couvre `project_path` + les clés de `video_sets`)

    Refuse et lève RuntimeError si config_dir contient déjà un item
    homonyme (autre que _config.py) — signale un conflit avec un run
    précédent que l'user doit nettoyer.
    """
    import shutil

    dated_dir = dlc_config_path.parent
    if dated_dir == config_dir:
        # Rien à merger — déjà au bon endroit (ex : re-run après nettoyage)
        return dlc_config_path

    # Vérification stricte du nom attendu, pour ne pas merger un dossier
    # créé par autre chose que ce script.
    expected_prefix = f"{project_name}-{experimenter}-"
    if not dated_dir.name.startswith(expected_prefix):
        print(f"⚠  Dossier DLC inattendu ({dated_dir.name}), merge annulé.\n"
              f"   Le _config.py devra être ajusté à la main.", file=sys.stderr)
        return dlc_config_path

    # Move chaque item dans config_dir. shutil.move gère cross-device
    # (rename plante si dated_dir et config_dir sont sur des volumes
    # différents ; en pratique impossible ici mais safe).
    for item in dated_dir.iterdir():
        dst = config_dir / item.name
        if dst.exists():
            raise RuntimeError(
                f"Conflit : {dst} existe déjà dans le dossier de config.\n"
                f"Nettoie {config_dir} (garde uniquement _config.py) puis "
                f"supprime aussi {dated_dir} avant de relancer 01."
            )
        shutil.move(str(item), str(dst))
    dated_dir.rmdir()

    new_config = config_dir / "config.yaml"

    # Substitution texte globale sur config.yaml : remplace TOUTES les
    # occurrences de l'ancien chemin par le nouveau. Couvre `project_path`
    # ET les clés de `video_sets` (chemins absolus vers les vidéos), sans
    # avoir à connaître la structure exacte du config DLC.
    old_str = str(dated_dir)
    new_str = str(config_dir)
    text = new_config.read_text(encoding="utf-8")
    text = text.replace(old_str, new_str)
    # Sur Windows, DLC peut aussi stocker le chemin avec forward slashes
    # dans certains champs — on remplace aussi cette variante.
    old_fwd = old_str.replace("\\", "/")
    new_fwd = new_str.replace("\\", "/")
    if old_fwd != old_str:
        text = text.replace(old_fwd, new_fwd)
    new_config.write_text(text, encoding="utf-8")

    return new_config


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

    # ---- Merge du dossier daté que DLC vient de créer dans le dossier
    # de config existant (où vit _config.py écrit par le wizard) ----
    # DLC crée <WORKDIR>/<name>-<exp>-YYYY-MM-DD/. On déplace tout son
    # contenu dans <WORKDIR>/<name>/ (le dossier de config), et on
    # supprime le dossier daté vide. Résultat : un SEUL dossier par
    # projet, contenant à la fois _config.py et le projet DLC.
    if args.config_dir is not None:
        target_dir = args.config_dir.resolve()
    else:
        # Cas legacy sans --config-dir : on merge dans WORKDIR/PROJECT_NAME
        target_dir = WORKDIR / PROJECT_NAME
        target_dir.mkdir(parents=True, exist_ok=True)
    config_path = merge_dlc_project_into_config_dir(
        config_path, target_dir, PROJECT_NAME, EXPERIMENTER,
    )
    print(f"✅ Projet DLC prêt dans : {config_path.parent}\n")

    # ---- Auto-patch du config.yaml DLC ----
    print(f"Patch du config.yaml :")
    print(f"  · bodyparts       = {len(DEFAULT_BODYPARTS)} keypoints")
    print(f"  · skeleton        = {len(DEFAULT_SKELETON)} liaisons")
    print(f"  · numframes2pick  = {N_AUTO_FRAMES}")
    patch_dlc_config(config_path, DEFAULT_BODYPARTS, DEFAULT_SKELETON,
                      N_AUTO_FRAMES)
    print(f"  → OK\n")

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
