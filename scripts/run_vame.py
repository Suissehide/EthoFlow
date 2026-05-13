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


def load_vame_config() -> dict:
    """vame-py 0.13 attend un config: dict (pas un chemin) dans la plupart des appels."""
    import vame
    path = load_config_pointer()
    return vame.read_config(path)


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

    # Auto-rekey : VAME (via movement) attend la clé HDF5 'df_with_missing'.
    # Les .h5 produits avant le fix avaient key='df' et VAME crashe dessus.
    # On corrige en place avant d'appeler init_new_project.
    if not args.no_auto_rekey:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        try:
            from rekey_h5 import is_already_correct, rekey
        except ImportError as e:
            print(f"⚠️  Impossible d'importer rekey_h5 ({e}), skip auto-rekey",
                  file=sys.stderr)
        else:
            to_fix = [h for _, h in pairs if not is_already_correct(h)]
            if to_fix:
                print(f"\n🔧 Auto-rekey : {len(to_fix)} fichier(s) à corriger "
                      f"(ancienne clé 'df' → 'df_with_missing')")
                for h in to_fix:
                    status = rekey(h)
                    if status == "rekeyed":
                        print(f"   ✓ {h.name}")
                    else:
                        print(f"   ⚠️  {h.name} : {status}", file=sys.stderr)

    VAME_PROJECTS_DIR.mkdir(parents=True, exist_ok=True)

    # Refus d'écraser un projet existant — sauf avec --force, qui supprime
    # tout le dossier (utile pour rattraper une tentative ratée qui a laissé
    # un dossier sans config.yaml)
    project_dir = VAME_PROJECTS_DIR / args.project_name
    if project_dir.exists():
        if not args.force:
            print(f"❌ Projet déjà présent : {project_dir}\n"
                  f"   Choisis un autre --project-name, ou relance avec --force "
                  f"pour écraser (le dossier sera supprimé).",
                  file=sys.stderr)
            sys.exit(1)
        import shutil
        print(f"⚠️  --force : suppression de {project_dir}")
        shutil.rmtree(project_dir)

    videos = [str(v) for v, _ in pairs]
    poses = [str(p) for _, p in pairs]

    # Symlink vs copy : sur Windows, les symlinks demandent des privilèges
    # admin ou le Developer Mode — sinon on prend une OSError 1314. Par défaut
    # on copie sur Windows et on symlinke ailleurs ; --copy-videos / --no-copy-videos
    # forcent un choix.
    import platform
    if args.copy_videos is None:
        copy_videos = (platform.system() == "Windows")
    else:
        copy_videos = args.copy_videos
    if copy_videos:
        print("ℹ️  Mode COPY (les vidéos sont copiées dans le projet, "
              "pas de symlinks).")

    print(f"\nCréation du projet VAME '{args.project_name}' dans {VAME_PROJECTS_DIR}...")
    # vame-py 0.13 : init_new_project retourne (config_path, config_dict)
    result = vame.init_new_project(
        project_name=args.project_name,
        poses_estimations=poses,
        source_software="DeepLabCut",
        working_directory=str(VAME_PROJECTS_DIR),
        videos=videos,
        video_type=".mp4",
        copy_videos=copy_videos,
    )
    # Compatibilité défensive : ancienne API renvoyait juste un chemin
    if isinstance(result, tuple):
        config_path = result[0]
    else:
        config_path = result
    save_config_pointer(config_path)
    print(f"\n✅ Projet VAME créé.\n   config.yaml : {config_path}")
    print("\nLe `config.yaml` contient les hyperparamètres du modèle. Tu peux\n"
          "l'éditer avant l'entraînement (taille de fenêtre, learning rate, etc).\n"
          "\nÉtape suivante :  python scripts/run_vame.py align")


def cmd_align(args) -> None:
    """
    vame-py >= 0.x : remplacé par `preprocessing` qui regroupe alignement
    égocentrique + filtrage outliers + lissage. Signature probable :
        vame.preprocessing(
            config,
            centered_reference_keypoint=<nom kp queue>,
            orientation_reference_keypoint=<nom kp nez>,
        )
    Si la signature diffère sur ta version, lance :
        python -c "import vame; help(vame.preprocessing)"
    et corrige les noms d'arguments dans la fonction ci-dessous.
    """
    import vame
    config = load_vame_config()

    # Les keypoints sont déjà dans le config.yaml du projet (copiés par
    # init_new_project), pas besoin de relire les .h5 originaux. Du coup
    # cmd_align ne dépend plus du dossier vame-input, ce qui simplifie
    # l'organisation quand on a plusieurs runs DLC distincts.
    bp = list(config.get("keypoints") or [])
    if not bp:
        print("❌ Pas de 'keypoints' dans le config.yaml du projet VAME.",
              file=sys.stderr)
        sys.exit(1)

    def find(candidates, default):
        for name in candidates:
            if name in bp:
                return name
        return default

    nose_kp = find(["nose", "Nose", "Snout", "snout"], bp[0])
    tail_kp = find(["tail_base", "tailbase", "Tailbase", "TailBase", "tail"],
                   bp[len(bp) // 2])

    print(f"  → keypoints détectés : {bp}")
    print(f"  → reference keypoints : center={tail_kp}, orientation={nose_kp}")

    steps = {
        "run_lowconf_cleaning":     not args.no_lowconf_cleaning,
        "run_egocentric_alignment": not args.no_alignment,
        "run_outlier_cleaning":     not args.no_outlier_cleaning,
        "run_savgol_filtering":     not args.no_savgol,
        "run_rescaling":            args.rescaling,
    }
    enabled = [k for k, v in steps.items() if v]
    skipped = [k for k, v in steps.items() if not v]
    print(f"  → étapes actives : {enabled}")
    if skipped:
        print(f"  → étapes désactivées : {skipped}")

    vame.preprocessing(
        config,
        centered_reference_keypoint=tail_kp,
        orientation_reference_keypoint=nose_kp,
        **steps,
    )
    print("\n✅ Preprocessing terminé. "
          "Étape suivante : python scripts/run_vame.py trainset")


def cmd_trainset(args) -> None:
    import vame
    vame.create_trainset(load_vame_config())
    print("\n✅ Trainset créé. Étape suivante : python scripts/run_vame.py train")


def cmd_train(args) -> None:
    import vame
    print("Entraînement du VAE — peut prendre plusieurs heures sur GPU.")
    print("Les hyperparamètres sont dans le config.yaml du projet VAME.")
    vame.train_model(load_vame_config())
    print("\n✅ Entraînement terminé.")


def cmd_evaluate(args) -> None:
    import vame
    vame.evaluate_model(load_vame_config())
    print("\n✅ Évaluation terminée — vois les figures dans le dossier du projet.")


def cmd_segment(args) -> None:
    """vame-py >= 0.x : `pose_segmentation` renommée en `segment_session`."""
    import vame
    print("Segmentation des poses en motifs comportementaux...")
    vame.segment_session(load_vame_config())
    print("\n✅ Segmentation terminée.")
    print("\nÉtapes optionnelles :")
    print(f'  - python -c "import vame; vame.motif_videos(vame.read_config(\\"{load_config_pointer()}\\"))"')
    print(f'  - python -c "import vame; vame.community(vame.read_config(\\"{load_config_pointer()}\\"))"')


def cmd_info(args) -> None:
    if CONFIG_POINTER.exists():
        config_path = CONFIG_POINTER.read_text().strip()
        print(f"Projet VAME courant : {config_path}")
        try:
            import vame
            cfg = vame.read_config(config_path)
            print(f"  → {len(cfg.get('session_names') or [])} session(s) "
                  f"importée(s) dans le projet")
            print(f"  → {len(cfg.get('keypoints') or [])} keypoint(s)")
        except Exception:
            pass
    else:
        print("Pas de projet VAME initialisé pour le moment.")

    # Scan optionnel d'un dossier vame-input (pour planifier un futur setup)
    input_dir = Path(args.input_dir) if args.input_dir else VAME_INPUT_DIR
    cropped_dir = Path(args.cropped_dir) if args.cropped_dir else CROPPED_DIR
    if input_dir.exists():
        pairs = find_pairs(input_dir, cropped_dir)
        sessions = {p[1].stem.rsplit("_", 1)[0] for p in pairs}
        print(f"\nDans {input_dir} :")
        print(f"  → {len(pairs)} paire(s) (vidéo + h5) disponibles")
        print(f"  → {len(sessions)} session(s) distincte(s)")


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
    p_setup.add_argument("--force", action="store_true",
                         help="Supprimer un projet du même nom s'il existe déjà")
    p_setup.add_argument("--no-auto-rekey", action="store_true",
                         help="Ne pas re-clé-er automatiquement les .h5 à la "
                              "clé 'df_with_missing' (par défaut auto-corrigé)")
    copy_grp = p_setup.add_mutually_exclusive_group()
    copy_grp.add_argument("--copy-videos", dest="copy_videos",
                          action="store_const", const=True, default=None,
                          help="Copier les vidéos dans le projet (au lieu de symlink). "
                               "Défaut auto : copy sur Windows, symlink ailleurs.")
    copy_grp.add_argument("--no-copy-videos", dest="copy_videos",
                          action="store_const", const=False,
                          help="Forcer le symlink (échouera sur Windows sans Developer Mode)")

    p_align = sub.add_parser("align", help="Preprocessing VAME (alignement + nettoyage)")
    p_align.add_argument("--no-lowconf-cleaning", action="store_true",
                         help="Skip le nettoyage low-confidence (seuil 0.99 par défaut, "
                              "trop strict pour SuperAnimal — déjà fait par notre pipeline)")
    p_align.add_argument("--no-alignment", action="store_true",
                         help="Skip l'alignement égocentrique")
    p_align.add_argument("--no-outlier-cleaning", action="store_true",
                         help="Skip le nettoyage des outliers IQR")
    p_align.add_argument("--no-savgol", action="store_true",
                         help="Skip le filtre Savitzky-Golay (qui crashe sur NaN — "
                              "à activer si trop de trous)")
    p_align.add_argument("--rescaling", action="store_true",
                         help="Activer le rescaling (désactivé par défaut)")
    sub.add_parser("trainset", help="Création du trainset")
    sub.add_parser("train",    help="Entraînement du VAE (long)")
    sub.add_parser("evaluate", help="Évaluation du modèle")
    sub.add_parser("segment",  help="Segmentation en motifs")

    p_info = sub.add_parser("info", help="Projet courant + diag rapide")
    p_info.add_argument("--input-dir", default=None,
                        help="Scanne ce dossier pour montrer combien de paires "
                             "seraient utilisées par un futur setup")
    p_info.add_argument("--cropped-dir", default=None)
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
