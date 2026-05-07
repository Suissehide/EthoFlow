"""
Assigne chaque frame DLC multi-animal à une arène, et split en .h5 single-animal.

Logique (assignation **par-frame**, robuste au tracker bancal) :
1. Lit le `.h5` multi-animal produit par SuperAnimal (ou un modèle DLC ma-)
2. Pour chaque individu × chaque frame, calcule sa position (centroïde) et
   sa qualité (likelihood moyenne sur les keypoints valides)
3. Pour chaque arène, construit un track single-animal en piochant à chaque
   frame la position du meilleur individu qui s'y trouve (en termes de
   likelihood). Si aucun individu n'est dans l'arène à une frame donnée,
   cette frame reste NaN.
4. **Nettoie** chaque track : low-likelihood → NaN, puis interpolation
   linéaire des trous courts (≤ interp-limit frames)
5. Écrit un fichier `<session>_<arene>.h5` single-animal par arène, dans
   `data/dlc-output/<session>/`

Pourquoi par-frame plutôt que par track entier : le tracker multi-animal
de SuperAnimal mélange souvent les identités des souris au cours d'une
vidéo. Avec une assignation par centroïde global, plusieurs tracks peuvent
revendiquer la même arène (et une autre arène se retrouve sans rien). Vu
que les arènes sont physiquement séparées (les souris ne peuvent pas
changer de boîte), la position à chaque frame est un signal parfaitement
fiable pour l'assignation.

Le nettoyage évite les trous dans les coordonnées (typique quand le détecteur
de SuperAnimal perd une souris immobile pendant quelques frames) — VAME
n'aime pas les NaN. Les trous longs (vraies absences) restent en NaN, c'est
ensuite à VAME de les ignorer.

Pré-requis :
- Avoir un metadata.yaml avec `arenes[*].coords = [x, y, w, h]` renseignées
- Avoir une sortie DLC multi-animal dans data/dlc-output/<session>/

Usage:
    python scripts/assign_arenas.py <session_id>
    python scripts/assign_arenas.py <session_id> --likelihood-threshold 0.6
    python scripts/assign_arenas.py <session_id> --interp-limit 50    # 2s @25fps
    python scripts/assign_arenas.py <session_id> --no-clean           # pas d'interpolation
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
DLC_OUTPUT_DIR = ROOT / "data" / "dlc-output"
CONFIG_PATH = ROOT / "configs" / "pipeline_config.yaml"


def load_default_coords() -> dict:
    if not CONFIG_PATH.exists():
        return {}
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f) or {}
    return config.get("default_arenes_coords", {}) or {}


def find_multianimal_h5(session_dlc_dir: Path) -> Path:
    """Cherche le fichier .h5 multi-animal d'origine (avant split)."""
    candidates = sorted(session_dlc_dir.glob("*.h5"))
    if not candidates:
        raise FileNotFoundError(f"Aucun .h5 dans {session_dlc_dir}")

    # On ignore les fichiers déjà splittés (suffixe _A1, _A2, etc.)
    multianimal = [c for c in candidates if not c.stem.endswith(("_A1", "_A2", "_A3", "_A4"))]
    if not multianimal:
        raise FileNotFoundError(
            f"Pas de .h5 multi-animal dans {session_dlc_dir} "
            f"(ne reste que des sorties déjà splittées)"
        )
    return multianimal[0]


def per_frame_centroids(sub: pd.DataFrame, threshold: float):
    """
    Pour un individu donné, calcule à chaque frame :
    - cx, cy : centroïde des keypoints valides (likelihood > threshold)
    - mean_lk : likelihood moyenne des keypoints valides (sert de qualité)

    Renvoie 3 ndarray de shape (n_frames,) — NaN aux frames sans détection.
    """
    xs = sub.xs("x", level="coords", axis=1).to_numpy()
    ys = sub.xs("y", level="coords", axis=1).to_numpy()
    likes = sub.xs("likelihood", level="coords", axis=1).to_numpy()

    mask = likes > threshold
    xs_masked = np.where(mask, xs, np.nan)
    ys_masked = np.where(mask, ys, np.nan)
    likes_masked = np.where(mask, likes, np.nan)

    with np.errstate(invalid="ignore", all="ignore"):
        cx = np.nanmedian(xs_masked, axis=1)
        cy = np.nanmedian(ys_masked, axis=1)
        mean_lk = np.nanmean(likes_masked, axis=1)
    return cx, cy, mean_lk


def build_per_arena_tracks(
    df: pd.DataFrame,
    arenes: list[dict],
    threshold: float,
) -> tuple[dict[str, pd.DataFrame], dict[str, dict[str, int]]]:
    """
    Construit un track single-animal par arène, en parcourant tous les
    individus et en attribuant chaque frame à son arène contenante.
    Si plusieurs individus revendiquent la même arène à la même frame,
    on garde celui dont la likelihood moyenne est la plus haute.

    Renvoie (per_arena_dfs, source_counts) où source_counts[arena_id][ind]
    indique combien de frames ont été contribuées par chaque individu.
    """
    individuals = list(df.columns.get_level_values("individuals").unique())
    n_frames = len(df)

    # Schéma de colonnes single-animal : on prend celui d'un individu pris
    # au hasard, en supprimant le niveau 'individuals'
    sample_sub = df.xs(individuals[0], level="individuals", axis=1)
    out_columns = sample_sub.columns
    n_cols = len(out_columns)

    valid_arenes = [ar for ar in arenes if ar.get("coords")]

    # Buffers numpy par arène : data + qualité courante (-inf au départ)
    arena_data: dict[str, np.ndarray] = {
        ar["id"]: np.full((n_frames, n_cols), np.nan) for ar in valid_arenes
    }
    arena_quality: dict[str, np.ndarray] = {
        ar["id"]: np.full(n_frames, -np.inf) for ar in valid_arenes
    }
    source_counts: dict[str, dict[str, int]] = {
        ar["id"]: defaultdict(int) for ar in valid_arenes
    }

    for ind in individuals:
        sub = df.xs(ind, level="individuals", axis=1)
        sub_arr = sub.to_numpy()  # (n_frames, n_cols)
        cx, cy, mean_lk = per_frame_centroids(sub, threshold)

        for ar in valid_arenes:
            x, y, w, h = ar["coords"]
            in_arena = (
                (cx >= x) & (cx < x + w) &
                (cy >= y) & (cy < y + h)
            )
            current_q = arena_quality[ar["id"]]
            # On améliore l'attribution si in_arena ET likelihood meilleure
            better = in_arena & (mean_lk > current_q)
            if better.any():
                arena_data[ar["id"]][better] = sub_arr[better]
                arena_quality[ar["id"]][better] = mean_lk[better]
                source_counts[ar["id"]][ind] += int(better.sum())

    # Conversion numpy → DataFrame (avec mêmes index/colonnes que la source)
    result = {
        ar_id: pd.DataFrame(data, index=df.index, columns=out_columns)
        for ar_id, data in arena_data.items()
    }
    return result, source_counts


def clean_individual(
    df: pd.DataFrame,
    likelihood_threshold: float,
    interp_limit: int,
) -> tuple[pd.DataFrame, dict]:
    """
    Pré-traite un track single-animal :
    1. Met x,y à NaN là où likelihood < threshold (détections peu fiables)
    2. Interpole linéairement les trous ≤ interp_limit frames
    3. Laisse les trous plus longs en NaN (vraies absences à laisser à VAME)

    Renvoie (df_clean, stats) où stats résume les frames touchées.
    Toutes les valeurs n_* sont en (frame × keypoint) ; total_slots = n_frames * n_keypoints.
    """
    df = df.copy()
    n_frames = len(df)

    # Grouper les colonnes par bodypart : on récupère le triplet (x, y, likelihood)
    by_bp: dict[str, dict[str, tuple]] = defaultdict(dict)
    for col in df.columns:
        # col est un tuple multi-index ; le bodypart est avant-dernier, coords dernier
        if not isinstance(col, tuple) or len(col) < 2:
            continue
        bp, coord = col[-2], col[-1]
        by_bp[bp][coord] = col

    valid_bps = [bp for bp, c in by_bp.items() if all(k in c for k in ("x", "y", "likelihood"))]
    n_keypoints = len(valid_bps)
    total_slots = n_frames * n_keypoints  # max possible (frame × keypoint) tuples

    n_low_likelihood = 0
    n_filled = 0
    n_remaining_nan = 0

    for bp in valid_bps:
        coords = by_bp[bp]
        x_col, y_col, l_col = coords["x"], coords["y"], coords["likelihood"]

        # Étape 1 : low-likelihood → NaN
        low = df[l_col] < likelihood_threshold
        n_low_likelihood += int(low.sum())
        df.loc[low, x_col] = np.nan
        df.loc[low, y_col] = np.nan

        nan_before = int(df[x_col].isna().sum())
        # Étape 2 : interpolation des trous courts
        df[x_col] = df[x_col].interpolate(
            method="linear", limit=interp_limit, limit_area="inside"
        )
        df[y_col] = df[y_col].interpolate(
            method="linear", limit=interp_limit, limit_area="inside"
        )
        nan_after = int(df[x_col].isna().sum())
        n_filled += (nan_before - nan_after)
        n_remaining_nan += nan_after

    stats = {
        "n_frames": n_frames,
        "n_keypoints": n_keypoints,
        "total_slots": total_slots,
        "n_low_likelihood": n_low_likelihood,
        "n_interpolated": n_filled,
        "n_remaining_nan": n_remaining_nan,
    }
    return df, stats


def assign_arenas(
    session_id: str,
    threshold: float = 0.6,
    interp_limit: int = 25,
    do_clean: bool = True,
) -> None:
    metadata_path = RAW_DIR / session_id / "metadata.yaml"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata absent : {metadata_path}")

    with open(metadata_path) as f:
        metadata = yaml.safe_load(f)
    arenes = metadata.get("arenes", [])

    # Fallback : compléter les coords manquantes avec default_arenes_coords
    default_coords = load_default_coords()
    for ar in arenes:
        if not ar.get("coords"):
            fallback = default_coords.get(ar.get("id"))
            if fallback:
                ar["coords"] = fallback

    if not any(ar.get("coords") for ar in arenes):
        raise ValueError(
            "Aucune arène n'a de coords (ni metadata.yaml, ni default_arenes_coords). "
            "Lance `python scripts/calibrate_arenes.py` pour les définir."
        )

    session_dlc_dir = DLC_OUTPUT_DIR / session_id
    h5_path = find_multianimal_h5(session_dlc_dir)
    print(f"Source DLC : {h5_path.name}")

    df = pd.read_hdf(h5_path)
    if "individuals" not in df.columns.names:
        raise ValueError(
            ".h5 ne contient pas le niveau 'individuals' — ce n'est pas une sortie "
            "multi-animal. Utilise une inférence multi-animal (SuperAnimal ou DLC ma-)."
        )

    individuals = list(df.columns.get_level_values("individuals").unique())
    print(f"Individus détectés : {len(individuals)} ({individuals})")
    print("Assignation par-frame (un track peut contribuer à plusieurs arènes).\n")

    valid_arenes = [ar for ar in arenes if ar.get("coords")]
    arena_dfs, source_counts = build_per_arena_tracks(df, valid_arenes, threshold)

    n_assigned = 0
    n_frames_total = len(df)
    for ar in valid_arenes:
        ar_id = ar["id"]
        sub = arena_dfs[ar_id]
        # Nombre de frames qui ont au moins une donnée non-NaN
        any_data = ~sub.isna().all(axis=1)
        n_filled = int(any_data.sum())

        if n_filled == 0:
            print(f"  ⚠️  {ar_id} : 0 frames assignées — aucune souris détectée "
                  f"dans cette zone")
            continue

        # Nettoyage avant écriture : low-lk → NaN, interpolation des trous courts
        if do_clean:
            sub_clean, stats = clean_individual(sub, threshold, interp_limit)
        else:
            sub_clean = sub
            stats = None

        mouse_id = ar.get("mouse_id")
        mouse_label = f"M{mouse_id:02d}" if isinstance(mouse_id, int) else "—"
        out_path = session_dlc_dir / f"{session_id}_{ar_id}.h5"
        sub_clean.to_hdf(out_path, key="df", mode="w")
        n_assigned += 1

        # Détail des sources : quels individus ont contribué à cette arène
        sc = source_counts[ar_id]
        sources_sorted = sorted(sc.items(), key=lambda x: -x[1])
        sources_str = ", ".join(f"{k}:{v}" for k, v in sources_sorted)

        pct_filled = 100 * n_filled / n_frames_total
        msg = (f"  ✓ {ar_id} ({mouse_label})  "
               f"{n_filled}/{n_frames_total} frames couvertes ({pct_filled:.1f}%)  "
               f"→ {out_path.name}\n"
               f"         sources : {sources_str}")
        if stats is not None:
            total = max(stats["total_slots"], 1)
            pct_low = 100 * stats["n_low_likelihood"] / total
            pct_interp = 100 * stats["n_interpolated"] / total
            pct_useful = 100 - 100 * stats["n_remaining_nan"] / total
            msg += (f"\n         clean ({stats['n_keypoints']} kp × {stats['n_frames']} frames): "
                    f"{pct_low:.1f}% low-lk → NaN, "
                    f"{pct_interp:.1f}% interpolés, "
                    f"{pct_useful:.1f}% utilisables au final")
        print(msg)

    expected = {ar["id"] for ar in valid_arenes if ar.get("mouse_id") is not None}
    found = {ar["id"] for ar in valid_arenes
             if not arena_dfs[ar["id"]].isna().all(axis=1).all()}
    missing = expected - found
    if missing:
        print(f"\n⚠️  Arènes attendues non couvertes : {sorted(missing)} "
              f"(souris non détectée — vérifier la calibration et la vidéo annotée)")

    print(f"\n✅ {n_assigned}/{len(valid_arenes)} arène(s) assignée(s)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("session_id")
    parser.add_argument("--likelihood-threshold", type=float, default=0.6,
                        help="Seuil de confiance — détections en dessous → NaN (défaut 0.6)")
    parser.add_argument("--interp-limit", type=int, default=25,
                        help="Taille max d'un trou interpolable, en frames (défaut 25 = 1s @ 25fps)")
    parser.add_argument("--no-clean", action="store_true",
                        help="Désactive complètement le nettoyage (low-lk → NaN, interpolation)")
    args = parser.parse_args()
    try:
        assign_arenas(
            args.session_id,
            threshold=args.likelihood_threshold,
            interp_limit=args.interp_limit,
            do_clean=not args.no_clean,
        )
    except (FileNotFoundError, ValueError) as e:
        print(f"❌ {e}", file=sys.stderr)
        sys.exit(1)
