"""Ajoute des vidéos supplémentaires au projet DLC (phase cross-mouse).

But : enrichir le dataset d'entraînement avec d'autres souris pour que le
modèle apprenne à généraliser. Sur un seul pilote, DLC tend à mémoriser
des features parasites du décor (coins, reflets IR fixes…) parce que
l'arrière-plan est identique d'une frame à l'autre. En ajoutant des
souris différentes, les éléments parasites changent ou disparaissent et
le modèle est forcé d'apprendre la silhouette de la souris elle-même.

Recommandation Tony (VAME/LIN) : « prendre autant d'animaux différents
que possible ». L'objectif est la plus large variété de situations
possible. Sur un projet à ~40 souris, viser 6-10 animaux distincts dans
le training set final. Sur un dataset plus petit, ratisse plus large.

Deux façons de passer les vidéos :

  A) --videos <path1> <path2> ... en CLI (recommandé) : les chemins sont
     écrits automatiquement dans ADDITIONAL_VIDEOS de ton _config.py
     puis lus par le reste du script. Zéro édition manuelle.

     python scripts/dlc_model-training/04_add_videos.py \\
         --config-dir D:/EthoFlow/models/souris-bottomview \\
         --videos D:/data/souris02.mp4 D:/data/souris03.mp4

     Par défaut les vidéos sont AJOUTÉES à la liste existante (dédup).
     Passe --replace-videos pour repartir de zéro.

  B) Édite ADDITIONAL_VIDEOS à la main dans ton _config.py puis lance
     sans --videos.

Autres réglages :
  --new-video-frames N   : override le nombre de kmeans par vidéo
                           (défaut = NEW_VIDEO_FRAMES du _config.py).

Ce script :
  - (si --videos) écrit les chemins dans ADDITIONAL_VIDEOS du _config.py ;
  - met à jour `numframes2pick` dans le config.yaml du projet DLC pour
    que extract_frames sorte NEW_VIDEO_FRAMES par vidéo ;
  - appelle dlc.add_new_videos pour enregistrer les vidéos ;
  - appelle dlc.extract_frames en mode k-means automatique.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import yaml

# Insère le dossier du script en tête de sys.path pour trouver _load_config
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _load_config import add_config_dir_arg, load_config  # noqa: E402


def add_videos_to_config_py(config_py_path: Path,
                              video_paths: list[Path],
                              append: bool = True) -> tuple[list[str], list[str]]:
    """Écrit ADDITIONAL_VIDEOS dans le _config.py de l'user.

    Trouve le bloc ADDITIONAL_VIDEOS = [ ... ] via un scan de profondeur
    des crochets (robuste multi-lignes + single-line). Reformate le bloc
    complet, virant les commentaires d'exemple du template.

    Si `append=True` (défaut), les vidéos déjà présentes (non-commentées)
    sont préservées et dédupliquées avec les nouvelles.

    Retourne (paths_ajoutés, paths_déjà_présents_préservés).
    """
    if not config_py_path.exists():
        raise FileNotFoundError(f"_config.py introuvable : {config_py_path}")

    text = config_py_path.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)

    # Trouve le début de l'assignation ADDITIONAL_VIDEOS
    start = None
    for i, line in enumerate(lines):
        if re.match(r"^\s*ADDITIONAL_VIDEOS\b", line):
            start = i
            break
    if start is None:
        raise ValueError(f"ADDITIONAL_VIDEOS absent de {config_py_path}")

    # Trouve la fin (profondeur des crochets revient à 0)
    depth = 0
    seen_open = False
    end = None
    for i in range(start, len(lines)):
        for c in lines[i]:
            if c == "[":
                depth += 1
                seen_open = True
            elif c == "]":
                depth -= 1
        if seen_open and depth == 0:
            end = i
            break
    if end is None:
        raise ValueError(f"Bloc ADDITIONAL_VIDEOS mal formé dans {config_py_path}")

    # Extrait les Path(...) déjà présents dans le bloc, hors commentaires
    existing: list[str] = []
    if append:
        for k in range(start, end + 1):
            line_str = lines[k]
            for m in re.finditer(r'Path\(r?"([^"]+)"\)', line_str):
                # Skip si dans un commentaire (# avant le match sur la même ligne)
                if "#" in line_str[:m.start()]:
                    continue
                existing.append(m.group(1))

    # Dédup (case-insensitive sur Windows, comparaison via chemin normalisé)
    def _key(p: str) -> str:
        return str(Path(p)).lower().replace("/", "\\")

    seen: set[str] = set()
    all_paths: list[str] = []
    for p in existing + [str(v) for v in video_paths]:
        k = _key(p)
        if k not in seen:
            seen.add(k)
            all_paths.append(p)

    # Reconstruit le bloc en écrasant l'ancien
    block = ["ADDITIONAL_VIDEOS: list[Path] = [\n"]
    for p in all_paths:
        block.append(f'    Path(r"{p}"),\n')
    block.append("]\n")

    new_lines = lines[:start] + block + lines[end + 1:]
    config_py_path.write_text("".join(new_lines), encoding="utf-8")

    newly_added = [str(v) for v in video_paths if _key(str(v)) not in
                     {_key(e) for e in existing}]
    preserved = existing
    return newly_added, preserved


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
    parser.add_argument(
        "--videos", nargs="+", type=Path, default=None,
        help="Chemins des vidéos à ajouter (écrits automatiquement dans "
             "ADDITIONAL_VIDEOS du _config.py, plus rien à éditer à la main). "
             "Par défaut ajoute à la liste existante — utilise --replace-videos "
             "pour repartir de zéro. Requiert --config-dir.",
    )
    parser.add_argument(
        "--replace-videos", action="store_true",
        help="Remplace complètement ADDITIONAL_VIDEOS au lieu d'ajouter "
             "(seulement pertinent avec --videos).",
    )
    parser.add_argument(
        "--new-video-frames", type=int, default=None,
        help="Override NEW_VIDEO_FRAMES (nombre de kmeans par vidéo).",
    )
    args = parser.parse_args()
    # `load_config` renseigne args.config_dir (flag ou invite) : le
    # _config.py à éditer plus bas est toujours celui de l'utilisateur.
    config_dir = load_config(args)

    # Si --videos fourni, écris dans _config.py AVANT de l'importer
    if args.videos:
        cfg_py = config_dir / "_config.py"
        # Vérifie que les vidéos existent avant d'écrire
        missing = [v for v in args.videos if not v.exists()]
        if missing:
            print("❌ Vidéos introuvables (ne rien écrit dans _config.py) :",
                  file=sys.stderr)
            for v in missing:
                print(f"   - {v}", file=sys.stderr)
            sys.exit(1)
        try:
            added, preserved = add_videos_to_config_py(
                cfg_py, args.videos, append=not args.replace_videos,
            )
        except (FileNotFoundError, ValueError) as e:
            print(f"❌ Impossible d'écrire dans {cfg_py} : {e}", file=sys.stderr)
            sys.exit(1)
        mode = "remplacé" if args.replace_videos else "ajouté"
        print(f"✓ _config.py mis à jour : {len(added)} vidéo(s) {mode}, "
              f"{len(preserved)} préservée(s).\n")

    import deeplabcut as dlc  # noqa: E402
    from _config import (  # noqa: E402
        ADDITIONAL_VIDEOS, CONFIG, NEW_VIDEO_FRAMES, PROJECT_DIR,
    )

    # Override NEW_VIDEO_FRAMES si le user l'a passé en CLI
    if args.new_video_frames is not None:
        NEW_VIDEO_FRAMES = args.new_video_frames  # noqa: F811

    if not ADDITIONAL_VIDEOS:
        print(
            "⚠ ADDITIONAL_VIDEOS est vide dans _config.py.\n"
            "Passe les vidéos en CLI :\n"
            "    python scripts/dlc_model-training/04_add_videos.py \\\n"
            "        --config-dir <ton dossier> \\\n"
            "        --videos D:/data/souris02.mp4 D:/data/souris03.mp4\n"
            "\n"
            "Ou édite la liste ADDITIONAL_VIDEOS à la main dans ton _config.py\n"
            "puis relance sans --videos."
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
