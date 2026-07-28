"""Initialise un nouveau projet EthoFlow à un chemin donné.

Un projet EthoFlow est un dossier auto-suffisant qui contient :

    <project>/
    ├── data/
    │   ├── raw/                 # data/raw/<session>/metadata.yaml
    │   ├── cropped/             # cropped videos (multi-animal seulement)
    │   ├── dlc-output/          # output DLC + h5 nettoyés (_clean.h5, _A*.h5)
    │   ├── vame/                # projets VAME par sous-dossier
    │   └── results/             # exports figures/csv finaux
    └── configs/
        └── pipeline_config.yaml # DLC project config + (multi-animal) arena coords

Ce script crée la structure de dossiers vide + une pipeline_config.yaml
template adaptée au type de projet.

Le `--kind` détermine si le projet contient plusieurs animaux par vidéo
(nécessite un split par arène) ou un seul animal par vidéo. C'est
indépendant de l'angle caméra (top vs bottom) : un projet bottom-view
avec 4 souris dans 4 arènes séparées est un `--kind multi`, tandis
qu'un projet top-view avec une seule souris dans une arène ouverte
est un `--kind single`.

Usage :
    # 1 animal par vidéo (pas d'arènes)
    python scripts/create_project.py \\
        --project-dir D:/ethoflow/projects/bottomview-MCC-2026-06 \\
        --kind single \\
        --dlc-config "E:/dlc-projects/souris-bottomview-labo-2026-06-05/config.yaml"

    # N animaux par vidéo (splitting d'arènes)
    python scripts/create_project.py \\
        --project-dir D:/ethoflow/projects/openfield-4mice \\
        --kind multi \\
        --dlc-config <chemin vers config.yaml>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import (  # noqa: E402
    add_project_dir_arg,
    configs_dir,
    cropped_dir,
    data_dir,
    dlc_output_dir,
    pipeline_config_path,
    raw_dir,
    results_dir,
    resolve_project,
    vame_dir,
)
from excel_templates import write_starter_excel  # noqa: E402


# Coords par défaut des 4 arènes (setup labo actuel : vidéo 1024×1080, grille
# 2×2). Utilisées quand `--kind multi` sans autre précision — à ajuster par
# session via `calibrate_arenes.py` si la caméra bouge ou si le layout change.
DEFAULT_MULTI_ARENA_COORDS = {
    "A1": [599, 40, 495, 465],
    "A2": [599, 506, 496, 503],
    "A3": [106, 501, 490, 505],
    "A4": [110, 39, 486, 460],
}


def build_pipeline_config(kind: str, dlc_config: str | None) -> dict:
    """Construit la config pipeline du projet selon son type."""
    config: dict = {}
    if dlc_config:
        config["dlc_project_config"] = dlc_config

    if kind == "multi":
        config["default_arenes_coords"] = DEFAULT_MULTI_ARENA_COORDS

    # single : pas d'arenes coords, c'est tout
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    add_project_dir_arg(parser, required=True)
    parser.add_argument(
        "--kind", choices=["single", "multi"], required=True,
        help=(
            "Nombre d'animaux par vidéo. "
            "'single' = 1 animal par vidéo (pas d'arena splitting), "
            "'multi' = N animaux par vidéo dans N arènes séparées "
            "(arena splitting activé + coords par défaut écrites dans "
            "pipeline_config.yaml). Indépendant de l'angle caméra."
        ),
    )
    parser.add_argument(
        "--dlc-config", type=str, default=None,
        help=(
            "OPTIONNEL. Chemin absolu vers le config.yaml d'un projet DLC "
            "déjà entraîné. Écrit dans pipeline_config.yaml comme "
            "`dlc_project_config`, lu uniquement par run_dlc_inference.py "
            "--mode custom. Le modèle DLC reste où il est (jamais copié) "
            "et peut servir à autant de projets EthoFlow que tu veux. "
            "Saute ce flag si tu ne sais pas encore quel modèle utiliser — "
            "édite pipeline_config.yaml plus tard, ou relance avec --force."
        ),
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Écrase pipeline_config.yaml même s'il existe déjà",
    )
    args = parser.parse_args()

    project = resolve_project(args)
    print(f"Initialisation du projet : {project}")
    print(f"  type : {args.kind}\n")

    # 1) Crée la structure de dossiers
    subdirs = [
        data_dir(project),
        raw_dir(project),
        cropped_dir(project),
        dlc_output_dir(project),
        vame_dir(project),
        results_dir(project),
        configs_dir(project),
    ]
    for d in subdirs:
        d.mkdir(parents=True, exist_ok=True)
        print(f"  ✓ {d.relative_to(project)}/")
    print()

    # 2) Crée la pipeline_config.yaml
    cfg_path = pipeline_config_path(project)
    if cfg_path.exists() and not args.force:
        print(f"  · pipeline_config.yaml existe déjà, skip (utilise --force pour écraser)")
    else:
        cfg = build_pipeline_config(args.kind, args.dlc_config)
        with open(cfg_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
        print(f"  ✓ {cfg_path.relative_to(project)} créé")
        if not args.dlc_config:
            print(
                f"    ⚠ dlc_project_config n'est pas renseigné — édite\n"
                f"      {cfg_path}\n"
                f"      pour pointer vers ton config.yaml DLC entraîné."
            )

    # 3) Crée le template Excel starter à la racine du projet
    excel_path = project / f"{project.name}_sessions.xlsx"
    if excel_path.exists() and not args.force:
        print(f"  · {excel_path.name} existe déjà, skip (utilise --force pour écraser)")
    else:
        try:
            write_starter_excel(excel_path, args.kind, project.name)
            print(f"  ✓ {excel_path.name} généré (à la racine du projet)")
        except RuntimeError as e:
            print(f"  ⚠ Template Excel non généré : {e}")

    print(f"\n✅ Projet initialisé.\n")

    print("Étapes suivantes :")
    print(f"  1. Remplis le template Excel :")
    print(f"       {excel_path}")
    print(f"     (feuille 'Instructions' à l'ouverture pour le mode d'emploi)")
    print()
    if args.kind == "single":
        print(f"  2. Sync des sessions depuis l'Excel :")
        print(f"       python scripts/sync_from_excel_single.py \\")
        print(f"           --project-dir {project} \\")
        print(f"           --excel {excel_path} \\")
        print(f"           --videos-dir <dossier des .mp4> \\")
        print(f"           --date YYYY-MM-DD")
        print()
        print(f"  3. Puis lancer l'inférence DLC :")
        print(f"       python scripts/run_dlc_inference.py --project-dir {project} "
              f"--all --mode custom")
    else:
        print(f"  2. Sync des sessions depuis l'Excel multi-animal :")
        print(f"       python scripts/sync_from_excel_multi.py \\")
        print(f"           --project-dir {project} \\")
        print(f"           --excel {excel_path} \\")
        print(f"           --videos-dir <dossier des .mp4>")
        print()
        print(f"  3. Puis le pipeline multi-animal complet "
              f"(crop + DLC + assign + clean) :")
        print(f"       python scripts/run_pipeline.py --project-dir {project} --all")


if __name__ == "__main__":
    main()
