"""
Inspecte les fichiers .nc préparés par VAME pour détecter les NaN/Inf
qui peuvent faire planter `train_model` (SVD ill-conditioned).

À lancer dans l'env vame :

    python scripts/inspect_vame_project.py
    python scripts/inspect_vame_project.py /chemin/vers/vame-projects/<projet>

Affiche par session :
- taille (frames, keypoints, espace)
- nombre de NaN par variable
- nombre d'Inf par variable
- min/max/std (pour repérer les valeurs aberrantes)
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONFIG_POINTER = ROOT / ".vame_config_path"


def project_root_from_pointer() -> Path | None:
    if not CONFIG_POINTER.exists():
        return None
    config_path = Path(CONFIG_POINTER.read_text().strip())
    return config_path.parent


def inspect_project(project_path: Path) -> None:
    import numpy as np
    import xarray as xr

    raw_dir = project_path / "data" / "raw"
    proc_dir = project_path / "data" / "processed"

    for label, d in [("raw", raw_dir), ("processed", proc_dir)]:
        print(f"\n{'═' * 60}")
        print(f"  Dossier {label}: {d}")
        print(f"{'═' * 60}")
        if not d.exists():
            print("  (n'existe pas)")
            continue

        nc_files = sorted(d.glob("*.nc"))
        if not nc_files:
            print("  (aucun .nc)")
            continue

        for nc in nc_files:
            print(f"\n  ── {nc.name} ──")
            try:
                ds = xr.open_dataset(nc)
            except Exception as e:
                print(f"    ⚠️  ouverture échouée : {e}")
                continue

            print(f"    dims  : {dict(ds.dims)}")
            for var_name, var in ds.data_vars.items():
                arr = var.values
                n_nan = int(np.isnan(arr).sum())
                n_inf = int(np.isinf(arr).sum())
                finite = arr[np.isfinite(arr)]
                total = arr.size
                if finite.size:
                    vmin, vmax, vstd = finite.min(), finite.max(), finite.std()
                    summary = f"min={vmin:.2f} max={vmax:.2f} std={vstd:.2f}"
                else:
                    summary = "tout NaN/Inf !"
                flag = "❌" if (n_nan or n_inf) else "✓ "
                pct_nan = 100 * n_nan / total if total else 0
                print(f"    {flag} {var_name:>20s} : "
                      f"NaN={n_nan} ({pct_nan:.2f}%)  Inf={n_inf}  {summary}")
            ds.close()


def main() -> None:
    if len(sys.argv) > 1:
        project = Path(sys.argv[1])
    else:
        project = project_root_from_pointer()
        if project is None:
            print("❌ Pas de projet VAME enregistré. Passe le chemin en argument.",
                  file=sys.stderr)
            sys.exit(1)

    if not project.exists():
        print(f"❌ Projet introuvable : {project}", file=sys.stderr)
        sys.exit(1)

    print(f"Projet VAME : {project}")
    inspect_project(project)


if __name__ == "__main__":
    main()
