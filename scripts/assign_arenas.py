"""
Assigne chaque track DLC multi-animal à une arène, et split en .h5 single-animal.

Logique :
1. Lit le `.h5` multi-animal produit par SuperAnimal (ou un modèle DLC ma-)
2. Pour chaque individu détecté, calcule un centroïde médian (toutes frames,
   tous keypoints, filtrés par likelihood)
3. Trouve l'arène dont le rectangle contient ce centroïde
4. **Nettoie** chaque track : low-likelihood → NaN, puis interpolation
   linéaire des trous courts (≤ interp-limit frames)
5. Écrit un fichier `<session>_<arene>.h5` single-animal par arène, dans
   `data/dlc-output/<session>/`

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


def compute_centroid(df_individual: pd.DataFrame, threshold: float) -> tuple[float, float, int]:
    """
    Centroïde médian d'un individu — utilise les détections > threshold.
    Retourne (cx, cy, n_valid) où n_valid = nombre total de (frame, keypoint)
    valides ayant servi au calcul.

    Robuste aux colonnes multi-niveau (scorer × bodyparts × coords) :
    on convertit en numpy et on fait la médiane sur tout d'un coup.
    """
    coords_levels = df_individual.columns.names
    if "coords" not in coords_levels:
        raise ValueError(
            f"Colonnes attendues avec un niveau 'coords' (x/y/likelihood), "
            f"trouvé : {coords_levels}"
        )

    xs = df_individual.xs("x", level="coords", axis=1).to_numpy()
    ys = df_individual.xs("y", level="coords", axis=1).to_numpy()
    likes = df_individual.xs("likelihood", level="coords", axis=1).to_numpy()

    mask = likes > threshold
    n_valid = int(np.sum(mask & ~np.isnan(xs) & ~np.isnan(ys)))
    if n_valid == 0:
        raise ValueError(
            f"0 détection > {threshold} (track sans données fiables)"
        )

    xs_masked = np.where(mask, xs, np.nan)
    ys_masked = np.where(mask, ys, np.nan)

    cx = float(np.nanmedian(xs_masked))
    cy = float(np.nanmedian(ys_masked))

    if np.isnan(cx) or np.isnan(cy):
        raise ValueError(
            f"Pas assez de détections > {threshold} pour calculer un centroïde"
        )
    return cx, cy, n_valid


def find_arena(cx: float, cy: float, arenes: list[dict]) -> dict | None:
    for ar in arenes:
        coords = ar.get("coords")
        if not coords:
            continue
        x, y, w, h = coords
        if x <= cx < x + w and y <= cy < y + h:
            return ar
    return None


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

    n_low_likelihood = 0
    n_filled = 0
    n_remaining_nan = 0

    for bp, coords in by_bp.items():
        if not all(c in coords for c in ("x", "y", "likelihood")):
            continue
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

    # Calcul des centroïdes pour TOUS les individus, puis tri par fiabilité
    # (n_valid décroissant) : ça met les 4 vraies souris en tête, les
    # détections fantômes (ghosts à très peu de frames valides) à la fin.
    candidates = []
    for ind in individuals:
        sub = df.xs(ind, level="individuals", axis=1)
        try:
            cx, cy, n_valid = compute_centroid(sub, threshold)
        except ValueError as e:
            print(f"  ⚠️  {ind} : {e}, skip")
            continue
        candidates.append((n_valid, ind, sub, cx, cy))

    candidates.sort(key=lambda c: -c[0])

    used_arenas: set[str] = set()
    for n_valid, ind, sub, cx, cy in candidates:
        arena = find_arena(cx, cy, arenes)
        if arena is None:
            print(f"  ⚠️  {ind} ({n_valid} pts valides) : centroïde "
                  f"({cx:.0f}, {cy:.0f}) hors de toute arène, skip")
            continue

        if arena["id"] in used_arenas:
            print(f"  ⚠️  {ind} ({n_valid} pts valides) : arène {arena['id']} "
                  f"déjà attribuée, skip (doublon de détection)")
            continue
        used_arenas.add(arena["id"])

        # Nettoyage avant écriture : interpolation des trous courts
        if do_clean:
            sub_clean, stats = clean_individual(sub, threshold, interp_limit)
            clean_pct_kept = (
                100 * (1 - stats["n_remaining_nan"] / max(stats["n_frames"], 1))
            )
        else:
            sub_clean = sub
            stats = None

        mouse_id = arena.get("mouse_id")
        out_path = session_dlc_dir / f"{session_id}_{arena['id']}.h5"
        sub_clean.to_hdf(out_path, key="df", mode="w")
        mouse_label = f"M{mouse_id:02d}" if isinstance(mouse_id, int) else "—"
        msg = (f"  ✓ {ind:>12s} → {arena['id']} ({mouse_label})  "
               f"centroïde ({cx:.0f}, {cy:.0f})  → {out_path.name}")
        if stats is not None:
            msg += (f"\n         clean: {stats['n_low_likelihood']:>5d} pts low-lk → NaN, "
                    f"{stats['n_interpolated']:>5d} interpolés, "
                    f"{stats['n_remaining_nan']:>5d} NaN restants ({clean_pct_kept:.1f}% utiles)")
        print(msg)

    expected = {ar["id"] for ar in arenes if ar.get("mouse_id") is not None}
    missing = expected - used_arenas
    if missing:
        print(f"\n⚠️  Arènes attendues non couvertes : {sorted(missing)} "
              f"(souris non détectée ?)")

    print(f"\n✅ {len(used_arenas)} arène(s) assignée(s)")


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
