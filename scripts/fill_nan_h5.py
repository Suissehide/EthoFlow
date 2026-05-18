"""
Remplit agressivement les NaN dans les .h5 single-animal avant VAME.

Stratégie par keypoint :
  1. Interpolation linéaire sur les trous (sans limite, on remplit tout
     entre deux valeurs valides)
  2. Forward-fill pour propager la dernière valeur valide sur la fin
  3. Backward-fill pour propager la première valeur valide au début
  4. Médiane du keypoint sur l'ensemble de la session pour les keypoints
     qui n'ont AUCUNE valeur valide (sinon fallback à 0)

La likelihood est mise à 1.0 sur les points qu'on remplit, pour que VAME
ne re-supprime pas les valeurs au seuil `pose_confidence` (0.99 par défaut).

Usage:
    python scripts/fill_nan_h5.py --root data/vame-input/single-2026-05-clean
    python scripts/fill_nan_h5.py --root <dir> --output-dir <dir-filled>
    python scripts/fill_nan_h5.py --root <dir> --dry-run

Ce script écrit en place dans --root par défaut, ou dans --output-dir
si tu veux préserver l'original.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tables  # noqa: F401  — pytables requis pour pd.read_hdf

ROOT = Path(__file__).resolve().parent.parent


def fill_h5(src: Path, dst: Path) -> dict:
    """Fill NaN dans un .h5 single-animal. Retourne des stats."""
    df = pd.read_hdf(src)
    n_total = df.size
    n_nan_before = int(df.isna().sum().sum())

    # Groupe les colonnes par bodypart : { bp -> {'x': col, 'y': col, 'likelihood': col} }
    by_bp: dict[str, dict[str, tuple]] = {}
    for col in df.columns:
        if not isinstance(col, tuple) or len(col) < 2:
            continue
        bp, coord = col[-2], col[-1]
        by_bp.setdefault(bp, {})[coord] = col

    n_filled_pts = 0
    for bp, coords in by_bp.items():
        if not all(c in coords for c in ("x", "y", "likelihood")):
            continue
        xcol, ycol, lcol = coords["x"], coords["y"], coords["likelihood"]

        # Mask des points à remplir (où x ou y est NaN)
        was_nan = df[xcol].isna() | df[ycol].isna()
        n_to_fill = int(was_nan.sum())
        if n_to_fill == 0:
            continue

        # Interpolation linéaire sans limite + ffill + bfill
        df[xcol] = df[xcol].interpolate(method="linear", limit_area="inside")
        df[ycol] = df[ycol].interpolate(method="linear", limit_area="inside")
        df[xcol] = df[xcol].ffill().bfill()
        df[ycol] = df[ycol].ffill().bfill()

        # Si TOUTE la colonne est NaN après ces étapes (rare), fillna(0)
        if df[xcol].isna().all():
            df[xcol] = 0.0
        if df[ycol].isna().all():
            df[ycol] = 0.0
        # Si quelques NaN résiduels (cas pathologique), médiane
        if df[xcol].isna().any():
            df[xcol] = df[xcol].fillna(df[xcol].median())
        if df[ycol].isna().any():
            df[ycol] = df[ycol].fillna(df[ycol].median())

        # Likelihood : on met 1.0 partout (et pas seulement sur les points
        # remplis). Justification :
        #  - les points remplis doivent passer n'importe quel seuil de confiance
        #  - les points originaux ont déjà été nettoyés par notre pipeline
        #    (assign_arenas filtre déjà à likelihood > 0.6), donc on a confiance
        #    en leurs valeurs ; pas de raison que VAME les rejette à 0.99.
        # Conséquence : VAME's lowconf_cleaning devient un no-op, ce qui
        # permet de garder ses défauts (alignment, IQR, etc.) sans casser.
        df[lcol] = 1.0

        n_filled_pts += n_to_fill

    n_nan_after = int(df.isna().sum().sum())

    # Sauve en .h5 avec la clé attendue par VAME
    dst.parent.mkdir(parents=True, exist_ok=True)
    df.to_hdf(dst, key="df_with_missing", mode="w", format="table")

    return {
        "n_frames": len(df),
        "n_nan_before": n_nan_before,
        "n_nan_after": n_nan_after,
        "n_filled_pts": n_filled_pts,
        "pct_before": 100 * n_nan_before / n_total if n_total else 0,
        "pct_after": 100 * n_nan_after / n_total if n_total else 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Fill NaN in single-animal .h5 files.")
    parser.add_argument("--root", type=Path, required=True,
                        help="Dossier source des .h5 (récursivement scanné)")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Dossier de sortie (défaut: réécriture en place)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Affiche les stats sans écrire")
    args = parser.parse_args()

    if not args.root.exists():
        print(f"❌ Dossier introuvable : {args.root}", file=sys.stderr)
        sys.exit(1)

    h5_files = sorted(args.root.rglob("*.h5"))
    if not h5_files:
        print(f"❌ Aucun .h5 dans {args.root}", file=sys.stderr)
        sys.exit(1)

    print(f"Traitement de {len(h5_files)} fichiers")
    print(f"Source : {args.root}")
    if args.output_dir:
        print(f"Sortie : {args.output_dir}")
    else:
        print(f"Sortie : en place (écrasement)")
    print()

    total_filled = 0
    for src in h5_files:
        rel = src.relative_to(args.root)
        if args.output_dir:
            dst = args.output_dir / rel
        else:
            dst = src

        if args.dry_run:
            # Juste lire pour calculer les stats
            df = pd.read_hdf(src)
            n_nan = int(df.isna().sum().sum())
            pct = 100 * n_nan / df.size if df.size else 0
            print(f"  [dry] {rel}  ({pct:.1f}% NaN)")
            continue

        try:
            stats = fill_h5(src, dst)
        except Exception as e:
            print(f"  ❌ {rel} : {e}", file=sys.stderr)
            continue

        total_filled += stats["n_filled_pts"]
        print(f"  ✓ {str(rel):60s}  "
              f"NaN: {stats['pct_before']:.1f}% → {stats['pct_after']:.1f}%  "
              f"({stats['n_filled_pts']} pts remplis)")

    if not args.dry_run:
        print(f"\n✅ {total_filled} points remplis au total.")
        print(f"   Prochaine étape : recrée le projet VAME (setup --force) "
              f"avec ces .h5.")


if __name__ == "__main__":
    main()
