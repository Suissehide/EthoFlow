"""
Runner VAME — bootstrap d'un projet à partir de data/vame-input/.

Étapes (sous-commandes séparées, à lancer dans l'ordre) :

    setup     init projet VAME à partir des paires (vidéo croppée, .h5)
    align     alignement égocentrique des poses
    trainset  création du jeu d'entraînement
    train     entraînement du VAE
    evaluate  diagnostics du modèle entraîné
    segment   segmentation des poses en motifs comportementaux
    info      affiche le projet courant et le statut
    all       enchaîne setup → segment (long, plusieurs heures sur GPU)

Pré-requis :
    - Avoir crée l'env conda 'vame' (`conda env create -f environment-vame.yml`)
    - Avoir `data/vame-input/<session>/<session>_A*.h5` (sorties d'assign_arenas
      ou de run_dlc_inference --mode single-animal)
    - Avoir les vidéos correspondantes dans `data/cropped/<session>/<session>_A*.mp4`
      → si tu n'as pas encore croppé, lance `python scripts/crop_arenes.py --all`
      depuis l'env ethoflow (rapide, ~2 min/session avec ffmpeg)

Usage typique :
    conda activate vame
    python scripts/run_vame.py setup --project-name ethoflow-2026-05
    python scripts/run_vame.py align
    python scripts/run_vame.py trainset
    python scripts/run_vame.py train
    python scripts/run_vame.py evaluate
    python scripts/run_vame.py segment

Référence officielle (à garder ouverte pour ajuster les hyperparamètres) :
    https://github.com/LINCellularNeuroscience/VAME/blob/master/examples/demo.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VAME_INPUT_DIR = ROOT / "data" / "vame-input"
CROPPED_DIR = ROOT / "data" / "cropped"
VAME_PROJECTS_DIR = ROOT.parent / "vame-projects"          # vit à côté d'ethoflow/
CONFIG_POINTER = ROOT / ".vame_config_path"                # ID du projet courant


# ============================================================
# Helpers
# ============================================================

def find_pairs(vame_input_dir: Path, cropped_dir: Path) -> list[tuple[Path, Path]]:
    """
    Liste toutes les paires (video croppée, .h5) trouvées.

    Pour chaque session du vame_input_dir, on cherche les .h5 nommés
    <session>_A*.h5 et la vidéo croppée correspondante <session>_A*.mp4.
    """
    pairs: list[tuple[Path, Path]] = []
    if not vame_input_dir.exists():
        return pairs
    for session_dir in sorted(vame_input_dir.iterdir()):
        if not session_dir.is_dir() or session_dir.name.startswith("."):
            continue
        session_id = session_dir.name
        for h5_path in sorted(session_dir.glob(f"{session_id}_A*.h5")):
            arena = h5_path.stem.rsplit("_", 1)[-1]  # "A1"
            video_path = cropped_dir / session_id / f"{session_id}_{arena}.mp4"
            if not video_path.exists():
                print(f"  ⚠️  vidéo manquante : {video_path} — skip {h5_path.name}",
                      file=sys.stderr)
                continue
            pairs.append((video_path, h5_path))
    return pairs


def save_config_pointer(config_path: str) -> None:
    CONFIG_POINTER.write_text(str(config_path))


def load_config_pointer() -> str:
    if not CONFIG_POINTER.exists():
        raise FileNotFoundError(
            "Pas de projet VAME initialisé.\n"
            "   Lance d'abord : python scripts/run_vame.py setup"
        )
    return CONFIG_POINTER.read_text().strip()


def detect_pose_ref_index(h5_path: Path) -> list[int]:
    """Trouve les indices de nose et tail_base parmi les keypoints du .h5."""
    import pandas as pd
    df = pd.read_hdf(h5_path)
    bp = df.columns.get_level_values("bodyparts").unique().tolist()
    # On cherche des noms compatibles avec ce qu'on a pu rencontrer
    def find(candidates, default):
        for name in candidates:
            if name in bp:
                return bp.index(name)
        return default
    nose = find(["nose", "Nose", "snout"], 0)
    tail = find(["tail_base", "tailbase", "TailBase", "tail", "tail_root"],
                len(bp) - 1)
    print(f"  → keypoints détectés : {bp}")
    print(f"  → pose_ref_index = [{nose} ({bp[nose]}), {tail} ({bp[tail]})]")
    return [nose, tail]


# ============================================================
# Commandes
# ============================================================

def cmd_setup(args) -> None:
    try:
        import vame
    except ImportError:
        print("❌ VAME non installé. Active l'env conda 'vame'.", file=sys.stderr)
        sys.exit(1)

    input_dir = Path(args.input_dir) if args.input_dir else VAME_INPUT_DIR
    cropped_dir = Path(args.cropped_dir) if args.cropped_dir else CROPPED_DIR

    pairs = find_pairs(input_dir, cropped_dir)
    if not pairs:
        print("❌ Aucune paire (vidéo croppée, .h5) trouvée.\n"
              "   Vérifie que tu as :\n"
              f"   - des .h5 dans {input_dir}/<session>/\n"
              f"   - des vidéos croppées dans {cropped_dir}/<session>/\n"
              "   (Lance `python scripts/crop_arenes.py --all` si besoin)",
              file=sys.stderr)
        sys.exit(1)

    print(f"{len(pairs)} paires trouvées :")
    for v, h in pairs[:5]:
        print(f"  {v.name}  +  {h.name}")
    if len(pairs) > 5:
        print(f"  ... (+{len(pairs) - 5} autres)")

    VAME_PROJECTS_DIR.mkdir(parents=True, exist_ok=True)
    videos = [str(v) for v, _ in pairs]
    poses = [str(p) for _, p in pairs]

    print(f"\nCréation du projet VAME '{args.project_name}' dans {VAME_PROJECTS_DIR}...")
    config_path = vame.init_new_project(
        project=args.project_name,
        videos=videos,
        poses_estimations=poses,
        working_directory=str(VAME_PROJECTS_DIR),
        videotype=".mp4",
    )
    save_config_pointer(config_path)
    print(f"\n✅ Projet VAME créé.\n   config.yaml : {config_path}")
    print("\nLe `config.yaml` contient les hyperparamètres du modèle. Tu peux\n"
          "l'éditer avant l'entraînement (taille de fenêtre, learning rate, etc).\n"
          "\nÉtape suivante :  python scripts/run_vame.py align")


def cmd_align(args) -> None:
    import vame
    config = load_config_pointer()
    # Le pose_ref_index est calé sur le premier .h5 trouvé (tous les fichiers
    # partagent la même structure de keypoints en sortie SuperAnimal)
    pairs = find_pairs(VAME_INPUT_DIR, CROPPED_DIR)
    pose_ref_index = detect_pose_ref_index(pairs[0][1])
    print(f"\nAlignement égocentrique avec pose_ref_index = {pose_ref_index}...")
    vame.egocentric_alignment(config, pose_ref_index=pose_ref_index)
    print("\n✅ Alignement terminé. Étape suivante : "
          "python scripts/run_vame.py trainset")


def cmd_trainset(args) -> None:
    import vame
    vame.create_trainset(load_config_pointer())
    print("\n✅ Trainset créé. Étape suivante : python scripts/run_vame.py train")


def cmd_train(args) -> None:
    import vame
    config = load_config_pointer()
    print("Entraînement du VAE — peut prendre plusieurs heures sur GPU.")
    print("Les hyperparamètres sont dans le config.yaml du projet VAME.")
    vame.train_model(config)
    print("\n✅ Entraînement terminé.")


def cmd_evaluate(args) -> None:
    import vame
    vame.evaluate_model(load_config_pointer())
    print("\n✅ Évaluation terminée — vois les figures dans le dossier du projet.")


def cmd_segment(args) -> None:
    import vame
    config = load_config_pointer()
    print("Segmentation des poses en motifs comportementaux...")
    vame.pose_segmentation(config)
    print("\n✅ Segmentation terminée.")
    print("\nÉtapes optionnelles :")
    print("  - python -c 'import vame; vame.motif_videos(\"%s\", videoType=\".mp4\")'" % config)
    print("  - python -c 'import vame; vame.community(\"%s\", show_umap=True)'" % config)


def cmd_info(args) -> None:
    if not CONFIG_POINTER.exists():
        print("Pas de projet VAME initialisé.")
        return
    print(f"Projet VAME courant : {CONFIG_POINTER.read_text().strip()}")
    pairs = find_pairs(VAME_INPUT_DIR, CROPPED_DIR)
    print(f"\nPaires (vidéo, h5) qui seraient utilisées par un nouveau setup : "
          f"{len(pairs)}")
    sessions = {p[1].stem.rsplit('_', 1)[0] for p in pairs}
    print(f"Sessions distinctes : {len(sessions)}")


def cmd_all(args) -> None:
    cmd_setup(args)
    cmd_align(args)
    cmd_trainset(args)
    cmd_train(args)
    cmd_evaluate(args)
    cmd_segment(args)


# ============================================================
# CLI
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Runner VAME pour EthoFlow.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_setup = sub.add_parser("setup", help="Init projet VAME")
    p_setup.add_argument("--project-name", default="ethoflow-vame",
                         help="Nom du projet VAME (créé dans ../vame-projects/)")
    p_setup.add_argument("--input-dir", default=None,
                         help="Dossier des .h5 (défaut: data/vame-input/)")
    p_setup.add_argument("--cropped-dir", default=None,
                         help="Dossier des vidéos croppées (défaut: data/cropped/)")

    sub.add_parser("align",    help="Alignement égocentrique")
    sub.add_parser("trainset", help="Création du trainset")
    sub.add_parser("train",    help="Entraînement du VAE (long)")
    sub.add_parser("evaluate", help="Évaluation du modèle")
    sub.add_parser("segment",  help="Segmentation en motifs")
    sub.add_parser("info",     help="Projet courant + diag rapide")
    sub.add_parser("all",      help="Tout enchaîner (très long)")

    args = parser.parse_args()
    {
        "setup":    cmd_setup,
        "align":    cmd_align,
        "trainset": cmd_trainset,
        "train":    cmd_train,
        "evaluate": cmd_evaluate,
        "segment":  cmd_segment,
        "info":     cmd_info,
        "all":      cmd_all,
    }[args.cmd](args)


if __name__ == "__main__":
    main()
