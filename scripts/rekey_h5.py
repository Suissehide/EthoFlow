"""
Re-écrit chaque .h5 single-animal avec la clé `df_with_missing` (convention DLC),
qui est ce que VAME / movement attend.

Utile pour rattraper les fichiers qui ont été sauvés avec key='df' avant le fix.
Lecture + ré-écriture, pas de calcul, c'est instantané.

Usage:
    python scripts/rekey_h5.py                          # toutes les .h5 dans data/dlc-output/
    python scripts/rekey_h5.py --root data/dlc-output-alt      # autre dossier
    python scripts/rekey_h5.py --root path/to/dir       # n'importe quel dossier
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import tables  # noqa: F401  — assure que pytables est bien là

# Import des chemins projet-aware
sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import (  # noqa: E402
    add_project_dir_arg,
    dlc_output_dir,
    resolve_project,
)

TARGET_KEY = "df_with_missing"


def is_already_correct(h5_path: Path) -> bool:
    """Vrai si le .h5 a déjà la clé df_with_missing au root."""
    try:
        with pd.HDFStore(str(h5_path), mode="r") as store:
            keys = [k.lstrip("/") for k in store.keys()]
            return TARGET_KEY in keys
    except Exception:
        return False


def rekey(h5_path: Path, force: bool = False) -> str:
    """Renvoie un statut : 'rekeyed' / 'already_ok' / 'failed: ...'."""
    if not force and is_already_correct(h5_path):
        return "already_ok"
    try:
        df = pd.read_hdf(h5_path)
        df.to_hdf(h5_path, key=TARGET_KEY, mode="w", format="table")
        return "rekeyed"
    except Exception as e:
        return f"failed: {e}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-écrit les .h5 avec key='df_with_missing'.")
    add_project_dir_arg(parser)
    parser.add_argument("--root", type=Path, default=None,
                        help="Racine à scanner (défaut: <project>/data/dlc-output/)")
    parser.add_argument("--force", action="store_true",
                        help="Re-écrire même les fichiers déjà OK")
    args = parser.parse_args()

    project = resolve_project(args)
    root = args.root if args.root is not None else dlc_output_dir(project)

    if not root.exists():
        print(f"❌ Racine introuvable : {root}", file=sys.stderr)
        sys.exit(1)

    h5_files = sorted(root.rglob("*.h5"))
    if not h5_files:
        print(f"Aucun .h5 dans {root}")
        sys.exit(0)

    print(f"Scan : {len(h5_files)} fichiers dans {root}\n")
    counts = {"rekeyed": 0, "already_ok": 0, "failed": 0}
    for h5 in h5_files:
        status = rekey(h5, force=args.force)
        if status == "rekeyed":
            counts["rekeyed"] += 1
            print(f"  ✓ {h5.relative_to(root)}")
        elif status == "already_ok":
            counts["already_ok"] += 1
            print(f"  · {h5.relative_to(root)}  (déjà OK)")
        else:
            counts["failed"] += 1
            print(f"  ❌ {h5.relative_to(root)} : {status}",
                  file=sys.stderr)

    print(f"\n{counts['rekeyed']} re-keyés, "
          f"{counts['already_ok']} déjà OK, "
          f"{counts['failed']} échec(s).")
    if counts["failed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
