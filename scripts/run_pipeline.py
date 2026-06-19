"""
Orchestrateur du pipeline complet.

Enchaîne par défaut : DLC (multi-animal) → assign_arenas → VAME.
Gère le passage entre les environnements conda automatiquement
en utilisant `conda run` (à condition que conda soit dans le PATH).

Le crop des vidéos en 4 single-animal n'est pas dans le chemin par défaut :
il sert uniquement aux étapes de labellisation / fine-tuning de modèle DLC,
pas à l'inférence de prod. Activable via `--crop-first`.

Usage:
    python scripts/run_pipeline.py <session_id>
    python scripts/run_pipeline.py --all                    # sessions non traitées
    python scripts/run_pipeline.py <session_id> --skip-vame
    python scripts/run_pipeline.py <session_id> --crop-first   # bonus: cropper avant
"""
import argparse
import subprocess
import sys
from pathlib import Path

# Import des chemins projet-aware
sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import (  # noqa: E402
    add_project_dir_arg,
    dlc_output_dir,
    raw_dir,
    resolve_project,
)

SCRIPTS_DIR = Path(__file__).resolve().parent


def list_unprocessed(project: Path):
    """Sessions présentes dans raw/ mais sans dossier dans dlc-output/."""
    rd = raw_dir(project)
    dd = dlc_output_dir(project)
    if not rd.exists():
        return []
    raw_sessions = {d.name for d in rd.iterdir() if d.is_dir()}
    done_sessions = (
        {d.name for d in dd.iterdir() if d.is_dir()}
        if dd.exists() else set()
    )
    return sorted(raw_sessions - done_sessions)


def run_in_env(env_name: str, script: str, session_id: str,
               project: Path, propagate_project: bool) -> int:
    """Lance un script dans un env conda spécifique via `conda run`.

    `propagate_project` : si True, on transmet `--project-dir` au sous-script
    (cas où on a un projet explicite). Sinon (fallback legacy), on laisse
    le sous-script retomber sur la racine du repo lui-même.
    """
    cmd = [
        "conda", "run", "-n", env_name,
        "python", str(SCRIPTS_DIR / script),
    ]
    if propagate_project:
        cmd += ["--project-dir", str(project)]
    cmd += [session_id]
    print(f"\n>>> [{env_name}] {script} {session_id}")
    return subprocess.call(cmd)


def main():
    parser = argparse.ArgumentParser(description="EthoFlow — orchestrateur du pipeline")
    add_project_dir_arg(parser)
    parser.add_argument("session_id", nargs="?", help="ID de session à traiter")
    parser.add_argument("--all", action="store_true",
                        help="Traiter toutes les sessions non traitées")
    parser.add_argument("--crop-first", action="store_true",
                        help="(Optionnel) cropper en 4 vidéos avant DLC — pour labellisation")
    parser.add_argument("--skip-dlc", action="store_true")
    parser.add_argument("--skip-assign", action="store_true",
                        help="Ne pas splitter la sortie DLC par arène")
    parser.add_argument("--skip-vame", action="store_true")
    parser.add_argument("--env-pipeline", default="ethoflow",
                        help="Nom de l'env conda pour les scripts utilitaires")
    parser.add_argument("--env-dlc", default="dlc",
                        help="Nom de l'env conda DeepLabCut")
    parser.add_argument("--env-vame", default="vame",
                        help="Nom de l'env conda VAME")
    args = parser.parse_args()

    project = resolve_project(args)
    # Seulement propager --project-dir aux sous-scripts si l'utilisateur l'a
    # explicitement fourni — sinon on laisse chacun retomber sur son repo.
    propagate_project = args.project_dir is not None

    if args.all:
        sessions = list_unprocessed(project)
        if not sessions:
            print("Aucune session à traiter.")
            return
        print(f"Sessions à traiter : {sessions}")
    elif args.session_id:
        sessions = [args.session_id]
    else:
        parser.print_help()
        sys.exit(1)

    for session_id in sessions:
        print(f"\n{'='*60}\nSession : {session_id}\n{'='*60}")

        if args.crop_first:
            rc = run_in_env(args.env_pipeline, "crop_arenes.py", session_id,
                            project, propagate_project)
            if rc != 0:
                print(f"❌ Crop a échoué pour {session_id}", file=sys.stderr)
                continue

        if not args.skip_dlc:
            rc = run_in_env(args.env_dlc, "run_dlc_inference.py", session_id,
                            project, propagate_project)
            if rc != 0:
                print(f"❌ DLC a échoué pour {session_id}", file=sys.stderr)
                continue

        if not args.skip_assign:
            rc = run_in_env(args.env_pipeline, "assign_arenas.py", session_id,
                            project, propagate_project)
            if rc != 0:
                print(f"❌ assign_arenas a échoué pour {session_id}", file=sys.stderr)
                continue

        if not args.skip_vame:
            rc = run_in_env(args.env_vame, "run_vame.py", session_id,
                            project, propagate_project)
            if rc != 0:
                print(f"❌ VAME a échoué pour {session_id}", file=sys.stderr)
                continue

        print(f"\n✅ Session {session_id} terminée")


if __name__ == "__main__":
    main()
