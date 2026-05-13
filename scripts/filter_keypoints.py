"""
Filtre les .h5 single-animal pour ne garder qu'une sous-liste de keypoints.

Indispensable quand certains keypoints (typiquement les segments distaux de queue
sur souris vue du dessus) sont si peu fiables qu'ils font planter VAME lors du
training (SVD sur matrice de Gram mal conditionnée).

Le script reproduit la même arborescence que --input-dir dans --output-dir.

Usage:
    python scripts/filter_keypoints.py \\
        --input-dir data/vame-input/pilote-2026-05 \\
        --output-dir data/vame-input/pilote-2026-05-clean \\
        --keep nose left_ear right_ear neck mid_back mouse_center \\
               tail_base left_hip right_hip

Options:
    --keep <list>           Keypoints à conserver (espace ou virgule séparés)
    --drop <list>           Keypoints à virer (alternative à --keep)
    --min-validity <float>  Auto-virer les keypoints avec une validité moyenne
                            inférieure à ce seuil sur l'ensemble des fichiers
                            (défaut : 0.7). Combinable avec --keep ou --drop.

Le format de sortie utilise key='df_with_missing' (convention DLC, attendu par VAME).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tables  # noqa: F401

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_KEEP = [
    "nose", "left_ear", "right_ear",
    "neck", "mid_back", "mouse_center",
    "tail_base", "left_hip", "right_hip",
]


def get_keypoint_names(h5_path: Path) -> list[str]:
    df = pd.read_hdf(h5_path)
    if "bodyparts" not in df.columns.names:
        return []
    return list(df.columns.get_level_values("bodyparts").unique())


def compute_validities(h5_files: list[Path]) -> dict[str, float]:
    """% moyen de frames où x n'est pas NaN, par keypoint, sur tous les fichiers."""
    totals: dict[str, list[float]] = {}
    for h5 in h5_files:
        df = pd.read_hdf(h5)
        bp = df.columns.get_level_values("bodyparts").unique()
        n_frames = len(df)
        for kp in bp:
            x_col = next(
                (c for c in df.columns if c[-2] == kp and c[-1] == "x"),
                None,
            )
            if x_col is None:
                continue
            v = float(df[x_col].notna().sum()) / max(n_frames, 1)
            totals.setdefault(str(kp), []).append(v)
    return {kp: sum(vs) / len(vs) for kp, vs in totals.items()}


def filter_h5(src: Path, dst: Path, keep: set[str]) -> tuple[int, int]:
    """Filtre les colonnes du .h5 pour ne garder que les keypoints dans `keep`.
    Retourne (n_kp_avant, n_kp_après)."""
    df = pd.read_hdf(src)
    all_bp = set(df.columns.get_level_values("bodyparts").unique())
    cols_to_keep = [c for c in df.columns if c[-2] in keep]
    filtered = df[cols_to_keep]

    dst.parent.mkdir(parents=True, exist_ok=True)
    filtered.to_hdf(dst, key="df_with_missing", mode="w", format="table")
    return len(all_bp), len(keep & all_bp)


def parse_list(arg: str | None) -> list[str]:
    if not arg:
        return []
    # accepte espaces, virgules, ou plusieurs args (déjà géré par nargs='+')
    items = []
    for chunk in arg if isinstance(arg, list) else [arg]:
        for x in chunk.replace(",", " ").split():
            if x:
                items.append(x)
    return items


def main() -> None:
    parser = argparse.ArgumentParser(description="Filtre les keypoints des .h5 single-animal.")
    parser.add_argument("--input-dir", type=Path, required=True,
                        help="Racine contenant <session>/<session>_A*.h5")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Racine de sortie (sera créée)")
    parser.add_argument("--keep", nargs="+", default=None,
                        help="Liste explicite de keypoints à conserver")
    parser.add_argument("--drop", nargs="+", default=None,
                        help="Liste de keypoints à virer (alternative à --keep)")
    parser.add_argument("--min-validity", type=float, default=None,
                        help="Auto-vire les keypoints en dessous de ce seuil de validité moyenne")
    parser.add_argument("--dry-run", action="store_true",
                        help="Affiche le plan sans rien écrire")
    args = parser.parse_args()

    h5_files = sorted(args.input_dir.rglob("*.h5"))
    if not h5_files:
        print(f"❌ Aucun .h5 dans {args.input_dir}", file=sys.stderr)
        sys.exit(1)

    # Détermine la liste à conserver
    all_kp = set()
    for h5 in h5_files[:5]:  # échantillon pour découvrir
        all_kp.update(get_keypoint_names(h5))

    keep: set[str] = set(all_kp)
    if args.keep:
        keep = set(args.keep)
    elif args.drop:
        keep = all_kp - set(args.drop)
    else:
        keep = set(DEFAULT_KEEP)
        print(f"ℹ️  Pas de --keep / --drop : valeurs par défaut "
              f"({len(keep)} keypoints) : {sorted(keep)}")

    if args.min_validity is not None:
        print(f"\nCalcul de validité moyenne sur {len(h5_files)} fichiers...")
        validities = compute_validities(h5_files)
        below = {kp for kp, v in validities.items() if v < args.min_validity}
        if below:
            print(f"   → {len(below)} keypoint(s) < {args.min_validity*100:.0f}% : "
                  f"{sorted(below)}")
            print(f"     ils seront virés en plus")
            keep -= below

    # Resync : ne garde dans `keep` que des keypoints qui existent vraiment
    keep &= all_kp
    if not keep:
        print(f"❌ Aucun keypoint à conserver après filtrage.", file=sys.stderr)
        sys.exit(1)

    print(f"\nKeypoints conservés ({len(keep)}) : {sorted(keep)}")
    print(f"Keypoints virés    ({len(all_kp - keep)}) : "
          f"{sorted(all_kp - keep)}\n")

    if args.dry_run:
        print(f"[dry-run] {len(h5_files)} fichier(s) seraient filtrés vers "
              f"{args.output_dir}")
        return

    n_done = n_fail = 0
    for src in h5_files:
        rel = src.relative_to(args.input_dir)
        dst = args.output_dir / rel
        try:
            n_before, n_after = filter_h5(src, dst, keep)
            print(f"  ✓ {rel}  ({n_before} → {n_after} kp)")
            n_done += 1
        except Exception as e:
            print(f"  ❌ {rel} : {e}", file=sys.stderr)
            n_fail += 1

    print(f"\n✅ Terminé : {n_done} OK, {n_fail} échec(s) — {args.output_dir}")


if __name__ == "__main__":
    main()
