"""
Post-traitement des sorties single-animal cropped :
aplatit le multi-index (individuals=1), nettoie (low-lk + interp),
et copie vers data/vame-input/<session>/<session>_<arene>.h5.

À utiliser si l'inférence DLC s'est bien passée (fichiers présents dans
`data/dlc-output/<session>/cropped-raw/`) mais que le post-traitement n'a
pas tourné (script interrompu, ancienne version du script, appel manuel
à DLC, etc.).

Le `run_dlc_inference.py --mode single-animal` fait normalement ce
post-traitement à la fin de l'inférence ; ce script ne sert qu'à le
rattraper après coup.

Usage:
    python scripts/post_process_cropped.py <session_id>
    python scripts/post_process_cropped.py <s1> <s2> ...
    python scripts/post_process_cropped.py --all
    python scripts/post_process_cropped.py <session> --output-dir data/vame-input-single
    python scripts/post_process_cropped.py <session> --likelihood-threshold 0.3 --interp-limit 100
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DLC_OUTPUT_DIR = ROOT / "data" / "dlc-output"
VAME_INPUT_DIR = ROOT / "data" / "vame-input"

# Import de clean_individual depuis assign_arenas
sys.path.insert(0, str(Path(__file__).resolve().parent))
from assign_arenas import clean_individual


def cropped_raw_dir(session_id: str) -> Path:
    return DLC_OUTPUT_DIR / session_id / "cropped-raw"


def list_sessions_with_cropped_raw() -> list[str]:
    if not DLC_OUTPUT_DIR.exists():
        return []
    return sorted(
        d.name for d in DLC_OUTPUT_DIR.iterdir()
        if d.is_dir() and (d / "cropped-raw").exists()
        and any((d / "cropped-raw").glob("*.h5"))
    )


def extract_arena_id(stem: str) -> str | None:
    """Trouve _A1 / _A2 / _A3 / _A4 dans le nom de fichier."""
    m = re.search(r"_A(\d)", stem)
    return f"A{m.group(1)}" if m else None


def post_process(
    session_id: str,
    likelihood_threshold: float = 0.6,
    interp_limit: int = 25,
    output_dir: Path | None = None,
) -> None:
    raw_dir = cropped_raw_dir(session_id)
    if not raw_dir.exists():
        raise FileNotFoundError(
            f"Pas de dossier cropped-raw pour cette session : {raw_dir}\n"
            f"   Lance d'abord (env dlc) :\n"
            f"   python scripts/run_dlc_inference.py {session_id} --mode single-animal"
        )

    # On ignore les .h5 'filtered' produits par d'éventuelles passes de filterpredictions
    h5_files = sorted(f for f in raw_dir.glob("*.h5") if "filtered" not in f.stem)
    if not h5_files:
        raise FileNotFoundError(f"Aucun .h5 dans {raw_dir}")

    base_out = Path(output_dir) if output_dir else VAME_INPUT_DIR
    out_dir = base_out / session_id
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Source : {raw_dir}")
    print(f"Sortie : {out_dir}")
    print(f"Fichiers trouvés ({len(h5_files)}) :")
    for f in h5_files:
        print(f"   - {f.name}")
    print()

    n_ok = n_skip = 0
    for h5_path in h5_files:
        arena_id = extract_arena_id(h5_path.stem)
        if arena_id is None:
            print(f"  ⚠️  {h5_path.name} : pas de _A<n> dans le nom, skip")
            n_skip += 1
            continue

        df = pd.read_hdf(h5_path)
        # Aplatissement : single-animal vrai, on drop le niveau 'individuals'
        if "individuals" in df.columns.names:
            df = df.droplevel("individuals", axis=1)

        # Nettoyage (réutilise la fonction d'assign_arenas)
        df_clean, stats = clean_individual(df, likelihood_threshold, interp_limit)

        out_path = out_dir / f"{session_id}_{arena_id}.h5"
        df_clean.to_hdf(out_path, key="df", mode="w")

        total = stats.get("total_slots") or 1
        pct_useful = 100 - 100 * stats["n_remaining_nan"] / total
        pct_low = 100 * stats["n_low_likelihood"] / total
        pct_interp = 100 * stats["n_interpolated"] / total
        print(f"  ✓ {arena_id} → {out_path.name}")
        print(f"     {pct_low:.1f}% low-lk → NaN, "
              f"{pct_interp:.1f}% interpolés, "
              f"{pct_useful:.1f}% utilisables")
        n_ok += 1

    print(f"\n✅ Post-traitement terminé : {n_ok} fichier(s), {n_skip} ignoré(s)")
    print(f"   Sortie : {out_dir}")
    print(f"\n   Pour l'inspection qualité :")
    print(f"   python scripts/inspect_session.py {session_id}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("session_ids", nargs="*",
                        help="Un ou plusieurs session_id à post-traiter")
    parser.add_argument("--all", action="store_true",
                        help="Toutes les sessions ayant un cropped-raw/")
    parser.add_argument("--likelihood-threshold", type=float, default=0.6,
                        help="Seuil de likelihood pour le nettoyage (défaut 0.6)")
    parser.add_argument("--interp-limit", type=int, default=25,
                        help="Taille max d'un trou interpolable, en frames "
                             "(défaut 25 = 1s @ 25fps)")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Dossier de sortie alternatif (défaut: data/vame-input/)")
    args = parser.parse_args()

    if args.all:
        sessions = list_sessions_with_cropped_raw()
    elif args.session_ids:
        sessions = list(args.session_ids)
    else:
        parser.print_help()
        sys.exit(1)

    if not sessions:
        print("Aucune session à traiter.")
        sys.exit(0)

    if len(sessions) > 1:
        print(f"{len(sessions)} session(s) : {sessions}\n")

    n_ok = n_fail = 0
    for i, s in enumerate(sessions, 1):
        if len(sessions) > 1:
            print(f"\n{'='*60}\n[{i}/{len(sessions)}] {s}\n{'='*60}")
        try:
            post_process(
                s,
                likelihood_threshold=args.likelihood_threshold,
                interp_limit=args.interp_limit,
                output_dir=args.output_dir,
            )
            n_ok += 1
        except FileNotFoundError as e:
            print(f"❌ {s} : {e}", file=sys.stderr)
            n_fail += 1
            continue

    if len(sessions) > 1:
        print(f"\nBatch terminé : {n_ok} OK, {n_fail} échec(s) sur {len(sessions)}")


if __name__ == "__main__":
    main()
