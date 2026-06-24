"""
Inspection qualité des fichiers .h5 prêts pour VAME.

Lit les fichiers `data/dlc-output/<session_id>/<session_id>_A*.h5` et imprime
un bilan détaillé par arène : couverture (frames valides vs trous), distribution
des trous, validité par keypoint, et un verdict pour VAME.

Marche aussi bien sur les sorties du chemin A (multi-animal + assign_arenas)
que du chemin B (single-animal cropped) puisque les deux produisent le même
format de fichier.

Usage:
    python scripts/inspect_session.py <session_id>
    python scripts/inspect_session.py <s1> <s2>            # plusieurs
    python scripts/inspect_session.py --all                # toutes
    python scripts/inspect_session.py <session_id> --input-dir data/dlc-output-alt
    python scripts/inspect_session.py <session_id> --fps 25
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# Import des chemins projet-aware
sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import (  # noqa: E402
    add_project_dir_arg,
    dlc_output_dir,
    raw_dir,
    resolve_project,
)


def get_bodypart_columns(df: pd.DataFrame) -> dict[str, dict[str, tuple]]:
    """Regroupe les colonnes par bodypart : { bp -> {x, y, likelihood} -> col tuple }."""
    by_bp: dict[str, dict[str, tuple]] = {}
    for col in df.columns:
        if not isinstance(col, tuple) or len(col) < 2:
            continue
        bp, coord = col[-2], col[-1]
        by_bp.setdefault(bp, {})[coord] = col
    # On ne garde que ceux qui ont les 3
    return {
        bp: c for bp, c in by_bp.items()
        if all(k in c for k in ("x", "y", "likelihood"))
    }


def analyze_h5(h5_path: Path) -> dict:
    df = pd.read_hdf(h5_path)
    n_frames = len(df)
    by_bp = get_bodypart_columns(df)
    n_kp = len(by_bp)

    # Compte par frame : nb de keypoints non-NaN
    valid_per_frame = np.zeros(n_frames, dtype=int)
    per_kp_pct: dict[str, float] = {}
    for bp, cols in by_bp.items():
        x = df[cols["x"]]
        valid = x.notna().to_numpy()
        valid_per_frame += valid.astype(int)
        per_kp_pct[bp] = float(valid.sum()) / max(n_frames, 1)

    frames_all_valid = int((valid_per_frame == n_kp).sum())
    frames_some_valid = int((valid_per_frame > 0).sum())
    frames_zero_valid = int((valid_per_frame == 0).sum())

    # Distribution des trous : runs consecutifs où aucun keypoint n'est valide
    is_empty = (valid_per_frame == 0).astype(int)
    if is_empty.any():
        # Repere les transitions
        d = np.diff(np.concatenate(([0], is_empty, [0])))
        starts = np.where(d == 1)[0]
        ends = np.where(d == -1)[0]
        gap_lengths = ends - starts
    else:
        gap_lengths = np.array([], dtype=int)

    return {
        "n_frames": n_frames,
        "n_kp": n_kp,
        "frames_all_valid": frames_all_valid,
        "frames_some_valid": frames_some_valid,
        "frames_zero_valid": frames_zero_valid,
        "gap_lengths": gap_lengths,
        "per_kp_pct": per_kp_pct,
    }


def verdict(coverage_pct: float) -> str:
    if coverage_pct >= 90:
        return "✅ excellent — VAME va tourner sans souci"
    if coverage_pct >= 80:
        return "✓  bon — VAME OK"
    if coverage_pct >= 70:
        return "⚠️  marginal — VAME peut donner des résultats bruyants"
    return "❌ insuffisant — envisage de baisser --likelihood-threshold ou fine-tune DLC"


def print_report(arena_label: str, stats: dict, fps: float) -> None:
    n = stats["n_frames"]
    coverage = 100 * stats["frames_some_valid"] / n if n else 0
    print(f"\n  ── {arena_label} ──────────────────────────────────")
    print(f"    Frames totales            : {n}")
    print(f"    Frames toutes-kp valides  : {stats['frames_all_valid']:>6d} "
          f"({100 * stats['frames_all_valid'] / n:.1f}%)")
    print(f"    Frames avec ≥1 kp valide  : {stats['frames_some_valid']:>6d} "
          f"({coverage:.1f}%)")
    print(f"    Frames totalement vides   : {stats['frames_zero_valid']:>6d} "
          f"({100 * stats['frames_zero_valid'] / n:.1f}%)")

    gaps = stats["gap_lengths"]
    if len(gaps):
        print(f"    Trous vides (frames consécutives sans détection) :")
        print(f"      nombre  : {len(gaps)}")
        print(f"      max     : {gaps.max()} frames ({gaps.max() / fps:.1f}s @ {fps:.0f}fps)")
        print(f"      p95     : {int(np.percentile(gaps, 95))} frames "
              f"({np.percentile(gaps, 95) / fps:.1f}s)")
        print(f"      médiane : {int(np.median(gaps))} frames")
    else:
        print(f"    Aucun trou — couverture continue")

    # Keypoints : top 3 et bottom 3
    sorted_kp = sorted(stats["per_kp_pct"].items(), key=lambda x: -x[1])
    print(f"    Validité par keypoint (sur {stats['n_kp']} kp) :")
    print(f"      top 3    : " + ", ".join(f"{bp}={100*p:.0f}%"
                                            for bp, p in sorted_kp[:3]))
    print(f"      bottom 3 : " + ", ".join(f"{bp}={100*p:.0f}%"
                                            for bp, p in sorted_kp[-3:]))
    print(f"    Verdict : {verdict(coverage)}")


def get_session_fps(project: Path, session_id: str, default: float = 25.0) -> float:
    meta_path = raw_dir(project) / session_id / "metadata.yaml"
    if not meta_path.exists():
        return default
    with open(meta_path) as f:
        meta = yaml.safe_load(f) or {}
    return float(meta.get("camera", {}).get("fps", default))


def inspect_session(project: Path, session_id: str, input_dir: Path, fps: float | None) -> None:
    session_dir = input_dir / session_id
    if not session_dir.exists():
        raise FileNotFoundError(f"Dossier introuvable : {session_dir}")

    h5_files = sorted(session_dir.glob(f"{session_id}_A*.h5"))
    if not h5_files:
        raise FileNotFoundError(f"Aucun .h5 single-animal dans {session_dir}")

    if fps is None:
        fps = get_session_fps(project, session_id)

    print(f"\n══ {session_id} ({input_dir.name}, fps={fps:.0f}) ══")

    overall_coverages = []
    for h5_path in h5_files:
        arena = h5_path.stem.rsplit("_", 1)[-1]  # "..._A1" → "A1"
        stats = analyze_h5(h5_path)
        print_report(f"Arène {arena}  [{h5_path.name}]", stats, fps)
        cov = stats["frames_some_valid"] / max(stats["n_frames"], 1) * 100
        overall_coverages.append(cov)

    if overall_coverages:
        avg = sum(overall_coverages) / len(overall_coverages)
        print(f"\n  Couverture moyenne session : {avg:.1f}%")


def list_sessions(input_dir: Path) -> list[str]:
    if not input_dir.exists():
        return []
    return sorted(
        d.name for d in input_dir.iterdir()
        if d.is_dir() and not d.name.startswith(".")
        and any(d.glob(f"{d.name}_A*.h5"))
    )


def main():
    parser = argparse.ArgumentParser(description="Inspection qualité des sorties prêtes pour VAME.")
    add_project_dir_arg(parser)
    parser.add_argument("session_ids", nargs="*",
                        help="Une ou plusieurs sessions à inspecter")
    parser.add_argument("--all", action="store_true",
                        help="Inspecter toutes les sessions présentes dans --input-dir")
    parser.add_argument("--input-dir", type=Path, default=None,
                        help="Racine des .h5 single-animal (défaut: <project>/data/dlc-output/)")
    parser.add_argument("--fps", type=float, default=None,
                        help="FPS (défaut : lu depuis metadata.yaml, sinon 25)")
    args = parser.parse_args()

    project = resolve_project(args)
    input_dir = args.input_dir if args.input_dir is not None else dlc_output_dir(project)

    if args.all:
        sessions = list_sessions(input_dir)
        if not sessions:
            print(f"Aucune session dans {input_dir}", file=sys.stderr)
            sys.exit(1)
    elif args.session_ids:
        sessions = list(args.session_ids)
    else:
        parser.print_help()
        sys.exit(1)

    for s in sessions:
        try:
            inspect_session(project, s, input_dir, args.fps)
        except FileNotFoundError as e:
            print(f"❌ {s} : {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
