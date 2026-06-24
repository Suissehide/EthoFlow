"""Initialise un nouveau projet EthoFlow à un chemin donné.

Un projet EthoFlow est un dossier auto-suffisant qui contient :

    <project>/
    ├── data/
    │   ├── raw/                 # data/raw/<session>/metadata.yaml
    │   ├── cropped/             # cropped videos (topview seulement)
    │   ├── dlc-output/          # output DLC + h5 nettoyés (_clean.h5, _A*.h5)
    │   ├── vame/                # projets VAME par sous-dossier
    │   └── results/             # exports figures/csv finaux
    └── configs/
        └── pipeline_config.yaml # DLC project config + (topview) arena coords

Ce script crée la structure de dossiers vide + une pipeline_config.yaml
template adaptée au type de projet.

Usage :
    # Projet bottom-view (1 souris = 1 vidéo, pas d'arènes)
    python scripts/create_project.py \\
        --project-dir D:/ethoflow/projects/bottomview-MCC-2026-06 \\
        --kind bottomview \\
        --dlc-config "E:/DLC/souris-bottomview-Leo-2026-06-05/config.yaml"

    # Projet topview (4 souris par vidéo, splitting d'arènes)
    python scripts/create_project.py \\
        --project-dir D:/ethoflow/projects/openfield-topview \\
        --kind topview \\
        --dlc-config <chemin vers config.yaml topview>
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


# Coords par défaut des 4 arènes (topview, vidéo 1024×1080, grille 2×2 du
# setup actuel du labo). À ajuster par session via calibrate_arenes.py.
DEFAULT_TOPVIEW_ARENA_COORDS = {
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

    if kind == "topview":
        config["default_arenes_coords"] = DEFAULT_TOPVIEW_ARENA_COORDS

    # bottomview : pas d'arenes coords, c'est tout
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    add_project_dir_arg(parser, required=True)
    parser.add_argument(
        "--kind", choices=["bottomview", "topview"], required=True,
        help=(
            "Type de projet : 'bottomview' = 1 souris par vidéo (pas "
            "d'arena splitting) ; 'topview' = 4 souris par vidéo (arena "
            "coords dans la config)."
        ),
    )
    parser.add_argument(
        "--dlc-config", type=str, default=None,
        help=(
            "Chemin absolu vers le config.yaml du projet DLC (sera écrit "
            "dans pipeline_config.yaml comme `dlc_project_config`). "
            "Peut être édité manuellement plus tard."
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

    print(f"\n✅ Projet initialisé.\n")

    next_step = (
        "Étape suivante :\n"
        "  - Sync des sessions depuis Excel :\n"
        "      python scripts/sync_from_excel_bottomview.py "
        f"--project-dir {project} \\\n"
        "          --excel <chemin> --videos-dir <chemin>\n"
        "  - Puis lancer l'inférence DLC :\n"
        f"      python scripts/run_dlc_inference.py --project-dir {project} "
        "--all --mode custom\n"
        if args.kind == "bottomview" else
        "Étape suivante :\n"
        "  - Sync des sessions depuis Excel topview :\n"
        "      python scripts/sync_from_excel.py "
        f"--project-dir {project} --excel <chemin>\n"
        "  - Puis le pipeline topview complet (crop + DLC + assign + clean) :\n"
        f"      python scripts/run_pipeline.py --project-dir {project} --all\n"
    )
    print(next_step)


if __name__ == "__main__":
    main()
