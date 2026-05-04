"""
Orchestrateur du pipeline complet.

Enchaîne crop → DLC → VAME pour une ou plusieurs sessions.
Gère le passage entre les environnements conda automatiquement
en utilisant `conda run` (à condition que conda soit dans le PATH).

Usage:
    python scripts/run_pipeline.py <session_id>
    python scripts/run_pipeline.py --all                    # toutes les sessions non traitées
    python scripts/run_pipeline.py <session_id> --skip-vame # sauter VAME

Note: ce script peut être appelé depuis n'importe quel env Python
(typiquement 'pipeline-souris'), il invoque les bons envs en sous-processus.
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
DLC_OUTPUT_DIR = ROOT / "data" / "dlc-output"
SCRIPTS_DIR = ROOT / "scripts"


def list_unprocessed():
    """Sessions présentes dans raw/ mais sans dossier dans dlc-output/."""
    if not RAW_DIR.exists():
        return []
    raw_sessions = {d.name for d in RAW_DIR.iterdir() if d.is_dir()}
    done_sessions = (
        {d.name for d in DLC_OUTPUT_DIR.iterdir() if d.is_dir()}
        if DLC_OUTPUT_DIR.exists() else set()
    )
    return sorted(raw_sessions - done_sessions)


def run_in_env(env_name: str, script: str, session_id: str) -> int:
    """Lance un script dans un env conda spécifique via `conda run`."""
    cmd = [
        "conda", "run", "-n", env_name,
        "python", str(SCRIPTS_DIR / script), session_id,
    ]
    print(f"\n>>> [{env_name}] {script} {session_id}")
    return subprocess.call(cmd)


def main():
    parser = argparse.ArgumentParser(description="Pipeline souris orchestrateur")
    parser.add_argument("session_id", nargs="?", help="ID de session à traiter")
    parser.add_argument("--all", action="store_true",
                        help="Traiter toutes les sessions non traitées")
    parser.add_argument("--skip-crop", action="store_true")
    parser.add_argument("--skip-dlc", action="store_true")
    parser.add_argument("--skip-vame", action="store_true")
    parser.add_argument("--env-pipeline", default="pipeline-souris",
                        help="Nom de l'env conda pour les scripts utilitaires")
    parser.add_argument("--env-dlc", default="dlc",
                        help="Nom de l'env conda DeepLabCut")
    parser.add_argument("--env-vame", default="vame",
                        help="Nom de l'env conda VAME")
    args = parser.parse_args()

    if args.all:
        sessions = list_unprocessed()
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

        if not args.skip_crop:
            rc = run_in_env(args.env_pipeline, "crop_arenes.py", session_id)
            if rc != 0:
                print(f"❌ Crop a échoué pour {session_id}", file=sys.stderr)
                continue

        if not args.skip_dlc:
            rc = run_in_env(args.env_dlc, "run_dlc_inference.py", session_id)
            if rc != 0:
                print(f"❌ DLC a échoué pour {session_id}", file=sys.stderr)
                continue

        if not args.skip_vame:
            rc = run_in_env(args.env_vame, "run_vame.py", session_id)
            if rc != 0:
                print(f"❌ VAME a échoué pour {session_id}", file=sys.stderr)
                continue

        print(f"\n✅ Session {session_id} terminée")


if __name__ == "__main__":
    main()
