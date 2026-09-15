r"""Extraction manuelle de frames — sur TOUTES les vidéos du projet.

Pourquoi ce script existe. L'appel documenté partout ailleurs :

    deeplabcut.extract_frames(CONFIG, mode="manual", crop=False,
                              userfeedback=False)

n'ouvre la GUI que sur **la première vidéo** du projet. Ce n'est pas un
réglage qu'on aurait raté : en mode `manual`, DLC ignore la liste et fait
`launch_napari(videos[0])` avant de rendre la main — `crop`, `algo` et
`userfeedback` ne servent qu'au mode automatique. Les vidéos ajoutées par
`04_add_videos.py` ne sont donc jamais proposées, alors que ce sont
précisément elles qui portent la diversité inter-individus qu'on cherche
à couvrir à la main (cf. Parcours B du README, B.3.2 et B.6).

Ce script lit la liste qui fait foi — `video_sets` du `config.yaml` DLC,
c'est-à-dire la pilote **plus** tout ce que 04 a enregistré — et lance la
GUI une vidéo à la fois, en te disant combien de frames chacune porte
déjà.

Usage :

    conda activate dlc

    # Menu des vidéos du projet, tu choisis lesquelles
    python scripts/dlc_model-training/extract_frames_manual.py ^
        --config-dir D:\EthoFlow\models\souris-bottomview

    # Toutes, sans rien demander
    python scripts/dlc_model-training/extract_frames_manual.py ^
        --config-dir D:\EthoFlow\models\souris-bottomview --all

    # Deux vidéos précises (numéro du menu ou nom, extension optionnelle)
    python scripts/dlc_model-training/extract_frames_manual.py ^
        --config-dir D:\EthoFlow\models\souris-bottomview ^
        --videos souris02 souris03

Comme partout dans le Parcours B, `--config-dir` est optionnel : sans
lui, le menu des modèles trouvés sous `D:\EthoFlow\models` s'affiche.

Dans la GUI (napari) : slider + flèches gauche/droite pour le
frame-par-frame, puis « Extract frame » sur chaque moment intéressant.
**Ferme la fenêtre pour passer à la vidéo suivante.**
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import yaml

# Insère le dossier du script en tête de sys.path pour trouver _load_config
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _load_config import add_config_dir_arg, load_config  # noqa: E402
from interactive import prompt  # noqa: E402

# Réponses qui veulent dire « toutes les vidéos ».
TOUT = {"", "all", "tout", "toutes", "*"}

# Séparateurs acceptés dans une sélection : « 1,3 », « 1 3 », « 1; 3 ».
SEPARATEURS = re.compile(r"[,;\s]+")


def read_video_sets(project_config_path: Path | str) -> list[Path]:
    """Liste les vidéos déclarées dans `video_sets` du config.yaml DLC.

    C'est la liste qui fait foi : `01_setup_project.py` y met la pilote,
    `04_add_videos.py` y ajoute les suivantes. L'ordre du fichier est
    conservé (la pilote reste en tête).

    Les clés sont rendues **telles quelles**. Pas de `.resolve()` : DLC y
    écrit des chemins absolus Windows, qu'un resolve préfixerait du cwd
    côté POSIX et dont il casserait la forme UNC côté Windows.
    """
    with open(project_config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    return [Path(k) for k in (cfg.get("video_sets") or {})]


def select_videos(videos: list[Path], reponse: str) -> list[Path]:
    """Filtre `videos` depuis une saisie utilisateur.

    Accepte les numéros du menu (1-indexés), les noms de vidéo avec ou
    sans extension (insensible à la casse), et un mélange des deux. Vide
    ou « all » prend tout. Le résultat est dédupliqué et rendu dans
    l'ordre du projet.

    Lève `ValueError` sur une saisie non résolvable — mieux vaut
    redemander que d'ouvrir napari sur la mauvaise vidéo.
    """
    saisie = (reponse or "").strip()
    if saisie.lower() in TOUT:
        return list(videos)

    par_nom: dict[str, Path] = {}
    for v in videos:
        par_nom.setdefault(v.name.lower(), v)
        par_nom.setdefault(v.stem.lower(), v)

    retenues: set[Path] = set()
    for token in SEPARATEURS.split(saisie):
        if not token:
            continue
        if token.isdigit():
            i = int(token)
            if not 1 <= i <= len(videos):
                raise ValueError(
                    f"numéro hors liste : {token} (attendu 1-{len(videos)})")
            retenues.add(videos[i - 1])
        elif token.lower() in par_nom:
            retenues.add(par_nom[token.lower()])
        else:
            raise ValueError(f"vidéo inconnue : {token!r}")

    if not retenues:
        raise ValueError("sélection vide")
    return [v for v in videos if v in retenues]


def count_extracted_frames(project_dir: Path | str, video: Path) -> int:
    """Nombre de PNG déjà extraits pour cette vidéo (kmeans + manuel)."""
    labeled = Path(project_dir) / "labeled-data" / video.stem
    return len(list(labeled.glob("*.png"))) if labeled.is_dir() else 0


def print_menu(videos: list[Path], project_dir: Path) -> None:
    """Affiche les vidéos du projet avec leur compte de frames."""
    print(f"\nVidéos du projet ({len(videos)}) :")
    for i, v in enumerate(videos, start=1):
        n = count_extracted_frames(project_dir, v)
        print(f"  {i}. {v.name:<40} {n:>4} frame(s) déjà extraite(s)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    add_config_dir_arg(parser)
    parser.add_argument(
        "--videos", nargs="+", default=None, metavar="VIDEO",
        help="Vidéos à ouvrir : numéro du menu ou nom (extension "
             "optionnelle). Par défaut, le menu te demande.",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Ouvre toutes les vidéos du projet, sans rien demander.",
    )
    args = parser.parse_args()
    load_config(args)

    from _config import CONFIG, PROJECT_DIR  # noqa: E402

    project_config = Path(CONFIG)
    if not project_config.exists():
        print(f"❌ config.yaml DLC introuvable : {project_config}\n"
              f"   Lance d'abord 01_setup_project.py.", file=sys.stderr)
        sys.exit(1)

    videos = read_video_sets(project_config)
    if not videos:
        print(f"❌ Aucune vidéo dans video_sets de {project_config}\n"
              f"   01_setup_project.py y met la pilote, 04_add_videos.py "
              f"les suivantes.", file=sys.stderr)
        sys.exit(1)

    # Sélection : flags d'abord, menu sinon.
    if args.all:
        choisies = videos
    elif args.videos:
        try:
            choisies = select_videos(videos, " ".join(args.videos))
        except ValueError as exc:
            print(f"❌ --videos : {exc}", file=sys.stderr)
            sys.exit(1)
    elif args.no_prompt:
        choisies = videos  # non-interactif : tout, faute de mieux
    else:
        print_menu(videos, PROJECT_DIR)
        while True:
            try:
                choisies = select_videos(
                    videos,
                    prompt("Vidéos à ouvrir (numéros, noms, ou all)",
                           default="all"),
                )
                break
            except ValueError as exc:
                print(f"  ⚠ {exc}")

    require_deeplabcut()
    launch = _napari_launcher()
    if launch is None:
        _fallback_wx(CONFIG)
        return

    # DLC écrit les frames en chemin relatif au projet — comme le fait
    # `extract_frames` lui-même avant d'ouvrir sa GUI.
    os.chdir(PROJECT_DIR)

    print(f"\n{len(choisies)} vidéo(s) à ouvrir. Ferme la fenêtre napari "
          f"pour passer à la suivante.\n")
    total = 0
    for i, video in enumerate(choisies, start=1):
        avant = count_extracted_frames(PROJECT_DIR, video)
        print(f"[{i}/{len(choisies)}] {video.name}  ({avant} frame(s) déjà là)")
        if not video.exists():
            print(f"   ⚠ skip : fichier introuvable — {video}\n")
            continue
        launch(str(video))
        apres = count_extracted_frames(PROJECT_DIR, video)
        ajoutees = apres - avant
        total += ajoutees
        print(f"   {'✅' if ajoutees else '—'} {ajoutees} frame(s) ajoutée(s)"
              f"  →  labeled-data/{video.stem}/\n")

    print(f"Total : {total} frame(s) extraite(s) à la main.\n\n"
          f"Étape suivante — labellise-les :\n"
          f"  python -c \"import deeplabcut; deeplabcut.launch_dlc()\"\n"
          f"  puis charge {Path(CONFIG)} → onglet « Label Frames ».")


def require_deeplabcut() -> None:
    """Sort avec un message clair si l'environnement `dlc` n'est pas actif."""
    try:
        import deeplabcut  # noqa: F401
    except ImportError:
        print("❌ deeplabcut introuvable dans cet environnement Python.\n"
              "   Lance `conda activate dlc` avant ce script.",
              file=sys.stderr)
        sys.exit(1)


def _napari_launcher():
    """Renvoie `launch_napari` si cette version de DLC l'expose."""
    try:
        from deeplabcut.gui.widgets import launch_napari
    except ImportError:
        return None
    return launch_napari


def _fallback_wx(config: str) -> None:
    """DLC 2.1/2.2 : la GUI wx d'extraction, qui a son bouton Load Video.

    Cette génération de GUI sait changer de vidéo sans relancer quoi que
    ce soit — il n'y a donc rien à boucler, on l'ouvre une fois.
    """
    print("ℹ DLC sans napari détecté — ouverture du toolbox wx.\n"
          "   Utilise son bouton « Load Video » pour changer de vidéo.\n")
    from deeplabcut.generate_training_dataset import frame_extraction_toolbox
    frame_extraction_toolbox.show(config)


if __name__ == "__main__":
    main()
