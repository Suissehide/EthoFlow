"""
Assigne chaque track DLC multi-animal à une arène, et split en .h5 single-animal.

Logique :
1. Lit le `.h5` multi-animal produit par SuperAnimal (ou un modèle DLC ma-)
2. Pour chaque individu détecté, calcule un centroïde médian (toutes frames,
   tous keypoints, filtrés par likelihood)
3. Trouve l'arène dont le rectangle contient ce centroïde
4. Écrit un fichier `<session>_<arene>.h5` single-animal par arène, dans
   `data/dlc-output/<session>/`

Pré-requis :
- Avoir un metadata.yaml avec `arenes[*].coords = [x, y, w, h]` renseignées
- Avoir une sortie DLC multi-animal dans data/dlc-output/<session>/

Usage:
    python scripts/assign_arenas.py <session_id>
    python scripts/assign_arenas.py <session_id> --likelihood-threshold 0.6
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
DLC_OUTPUT_DIR = ROOT / "data" / "dlc-output"


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


def compute_centroid(df_individual: pd.DataFrame, threshold: float) -> tuple[float, float]:
    """Centroïde médian d'un individu — utilise les détections > threshold."""
    coords_levels = df_individual.columns.names
    if "coords" not in coords_levels:
        raise ValueError(
            f"Colonnes attendues avec un niveau 'coords' (x/y/likelihood), "
            f"trouvé : {coords_levels}"
        )

    xs = df_individual.xs("x", level="coords", axis=1)
    ys = df_individual.xs("y", level="coords", axis=1)
    likes = df_individual.xs("likelihood", level="coords", axis=1)

    mask = likes > threshold
    cx = xs.where(mask).stack().median()
    cy = ys.where(mask).stack().median()

    if pd.isna(cx) or pd.isna(cy):
        raise ValueError(
            f"Pas assez de détections > {threshold} pour calculer un centroïde"
        )
    return float(cx), float(cy)


def find_arena(cx: float, cy: float, arenes: list[dict]) -> dict | None:
    for ar in arenes:
        coords = ar.get("coords")
        if not coords:
            continue
        x, y, w, h = coords
        if x <= cx < x + w and y <= cy < y + h:
            return ar
    return None


def assign_arenas(session_id: str, threshold: float = 0.6) -> None:
    metadata_path = RAW_DIR / session_id / "metadata.yaml"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata absent : {metadata_path}")

    with open(metadata_path) as f:
        metadata = yaml.safe_load(f)
    arenes = metadata.get("arenes", [])

    if not any(ar.get("coords") for ar in arenes):
        raise ValueError(
            "Aucune arène n'a de coords dans le metadata.yaml. "
            "Définis-les avant de lancer ce script."
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
    print(f"Individus détectés : {individuals}")

    used_arenas: set[str] = set()
    for ind in individuals:
        sub = df.xs(ind, level="individuals", axis=1)
        try:
            cx, cy = compute_centroid(sub, threshold)
        except ValueError as e:
            print(f"  ⚠️  {ind} : {e}, skip")
            continue

        arena = find_arena(cx, cy, arenes)
        if arena is None:
            print(f"  ⚠️  {ind} : centroïde ({cx:.0f}, {cy:.0f}) hors de toute arène, skip")
            continue

        if arena["id"] in used_arenas:
            print(f"  ⚠️  {ind} : arène {arena['id']} déjà attribuée, skip "
                  f"(probable doublon de détection)")
            continue
        used_arenas.add(arena["id"])

        mouse_id = arena.get("mouse_id")
        out_path = session_dlc_dir / f"{session_id}_{arena['id']}.h5"
        sub.to_hdf(out_path, key="df", mode="w")
        mouse_label = f"M{mouse_id:02d}" if isinstance(mouse_id, int) else "—"
        print(f"  ✓ {ind:>12s} → {arena['id']} ({mouse_label})  "
              f"centroïde ({cx:.0f}, {cy:.0f})  → {out_path.name}")

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
                        help="Seuil de confiance pour le calcul du centroïde (défaut 0.6)")
    args = parser.parse_args()
    try:
        assign_arenas(args.session_id, threshold=args.likelihood_threshold)
    except (FileNotFoundError, ValueError) as e:
        print(f"❌ {e}", file=sys.stderr)
        sys.exit(1)
