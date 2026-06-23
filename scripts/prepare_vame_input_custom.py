"""Prépare les .h5 DLC custom-mode pour VAME (bottom-view, 12 keypoints).

Étapes par session (lit `<project>/data/dlc-output/<session>/...h5` →
écrit `<project>/data/vame-input/<session>/<session>.h5`) :

    1. dlc.filterpredictions  (median window, défaut 5 frames)
       Smooth temporel : tue les jitters d'une-deux frames sans toucher
       aux mouvements réels. Produit un `*_filtered.h5` à côté du brut.

    2. Likelihood < seuil → NaN  (défaut 0.3)
       Sur bottom-view IR, le seuil par défaut DLC (0.6) écrase 90 % des
       prédictions paws. À 0.3 on garde le corps confiant ET les pattes
       intermittentes. Tout point sous le seuil devient NaN.

    3. Interpolation linéaire des trous ≤ interp_limit frames (défaut 25)
       Reuses `clean_individual` de assign_arenas.py. Les trous longs
       (>25 frames ≈ 1s à 30 fps) restent NaN — vraies occlusions à
       laisser à VAME (ou à fill_nan_h5.py si besoin).

    4. Écriture en h5 single-animal (key="df_with_missing", format="table")
       dans `<project>/data/vame-input/<session>/<session>.h5`. C'est
       l'entrée canonique attendue par run_vame.py setup.

Pré-requis :
    - run_dlc_inference.py --mode custom a produit les .h5 dans dlc-output/
    - conda activate dlc  (besoin de deeplabcut pour filterpredictions)

Usage :
    python scripts/prepare_vame_input_custom.py \\
        --project-dir D:/ethoflow/projects/bottomview-MCC-2026-06

    # Skip le filtre temporel (si déjà fait, ou si tu veux la donnée brute) :
    python scripts/prepare_vame_input_custom.py \\
        --project-dir <...> --no-filter

    # Tweak les seuils (paws plus exigeantes, gap plus court) :
    python scripts/prepare_vame_input_custom.py \\
        --project-dir <...> \\
        --likelihood-threshold 0.4 --interp-limit 15

    # Cible une session précise (debug) :
    python scripts/prepare_vame_input_custom.py \\
        --project-dir <...> BV-970 BV-971

Étape suivante : si tu veux remplir agressivement les NaN résiduels (pour
VAME qui n'aime pas les trous), enchaîne avec :
    python scripts/fill_nan_h5.py --root <project>/data/vame-input
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from assign_arenas import clean_individual  # noqa: E402
from paths import (  # noqa: E402
    add_project_dir_arg,
    dlc_output_dir,
    pipeline_config_path,
    raw_dir,
    resolve_project,
    vame_input_dir,
)


def load_dlc_project_config(project: Path) -> str:
    """Lit dlc_project_config depuis la pipeline_config.yaml du projet."""
    import yaml
    cfg_path = pipeline_config_path(project)
    if not cfg_path.exists():
        raise FileNotFoundError(
            f"pipeline_config.yaml absent : {cfg_path}\n"
            f"Le filtre temporel a besoin du config DLC pour fonctionner."
        )
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f) or {}
    dlc_cfg = cfg.get("dlc_project_config")
    if not dlc_cfg:
        raise ValueError(f"dlc_project_config manquant dans {cfg_path}")
    return dlc_cfg


def find_raw_h5(session_out: Path) -> Path | None:
    """Trouve le .h5 brut produit par DLC analyze_videos pour cette session.

    Convention : c'est celui sans suffixe '_filtered' dans le nom.
    """
    candidates = [
        p for p in session_out.glob("*.h5")
        if "_filtered" not in p.stem
    ]
    if not candidates:
        return None
    # Si plusieurs (re-runs), prend le plus récent
    return max(candidates, key=lambda p: p.stat().st_mtime)


def filter_with_dlc(
    raw_h5: Path,
    source_video: Path,
    dlc_config: str,
    session_out: Path,
    windowlength: int,
) -> Path:
    """Lance dlc.filterpredictions et retourne le .h5 filtré produit."""
    import deeplabcut

    deeplabcut.filterpredictions(
        config=dlc_config,
        video=[str(source_video)],
        filtertype="median",
        windowlength=windowlength,
        save_as_csv=False,
        destfolder=str(session_out),
    )

    # DLC nomme la sortie en suffixant _filtered (et conserve _filtered.h5)
    filtered_candidates = list(session_out.glob("*_filtered.h5"))
    if not filtered_candidates:
        raise RuntimeError(
            f"dlc.filterpredictions n'a pas produit de *_filtered.h5 dans "
            f"{session_out}."
        )
    return max(filtered_candidates, key=lambda p: p.stat().st_mtime)


def process_session(
    project: Path,
    session_id: str,
    dlc_config: str | None,
    likelihood_threshold: float,
    interp_limit: int,
    window_length: int,
    skip_filter: bool,
) -> dict:
    """Pipeline complet pour une session. Renvoie un dict stats."""
    session_out = dlc_output_dir(project) / session_id
    if not session_out.exists():
        raise FileNotFoundError(f"Pas de dossier dlc-output pour {session_id}")

    raw_h5 = find_raw_h5(session_out)
    if raw_h5 is None:
        raise FileNotFoundError(f"Pas de .h5 brut dans {session_out}")

    # 1) Filtre temporel
    if skip_filter:
        h5_to_clean = raw_h5
    else:
        # Lit la metadata pour avoir le chemin de la vidéo source (DLC en a besoin)
        import yaml
        meta_path = raw_dir(project) / session_id / "metadata.yaml"
        with open(meta_path) as f:
            meta = yaml.safe_load(f)
        source_video = Path(meta["source_video"])

        if dlc_config is None:
            raise ValueError("dlc_config requis quand on filtre (charge depuis pipeline_config.yaml)")

        h5_to_clean = filter_with_dlc(
            raw_h5, source_video, dlc_config, session_out, window_length,
        )

    # 2) Mask + interpolation via clean_individual (réutilisation de assign_arenas)
    df = pd.read_hdf(h5_to_clean)
    df_clean, stats = clean_individual(df, likelihood_threshold, interp_limit)

    # 3) Écriture finale dans vame-input/<session>/<session>.h5
    vame_out_dir = vame_input_dir(project) / session_id
    vame_out_dir.mkdir(parents=True, exist_ok=True)
    out_path = vame_out_dir / f"{session_id}.h5"
    df_clean.to_hdf(out_path, key="df_with_missing", mode="w", format="table")

    stats["raw_h5"] = raw_h5.name
    stats["filtered_h5"] = h5_to_clean.name if not skip_filter else "(skipped)"
    stats["out_path"] = str(out_path)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    add_project_dir_arg(parser, required=True)
    parser.add_argument(
        "session_ids", nargs="*",
        help="Sessions à traiter (défaut: toutes celles avec un .h5 dans dlc-output/)",
    )
    parser.add_argument(
        "--likelihood-threshold", type=float, default=0.3,
        help="Seuil de likelihood en dessous duquel (x,y) deviennent NaN (défaut: 0.3)",
    )
    parser.add_argument(
        "--interp-limit", type=int, default=25,
        help="Taille max d'un trou interpolable en frames (défaut: 25 ≈ 1s à 30fps)",
    )
    parser.add_argument(
        "--window-length", type=int, default=5,
        help="Window du filtre médian DLC (défaut: 5 frames)",
    )
    parser.add_argument(
        "--no-filter", action="store_true",
        help="Skip dlc.filterpredictions (utilise le .h5 brut). "
             "Utile en debug ou pour des h5 déjà filtrés.",
    )
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="Ignore les sessions qui ont déjà un fichier dans vame-input/",
    )
    args = parser.parse_args()

    project = resolve_project(args)
    out_root = vame_input_dir(project)
    print(f"Projet     : {project}")
    print(f"Sortie     : {out_root}")
    print(f"Threshold  : {args.likelihood_threshold}  (paws bottom-view : 0.3 OK)")
    print(f"Interp lim : {args.interp_limit} frames")
    print(f"Filtre DLC : {'OFF (--no-filter)' if args.no_filter else f'median win={args.window_length}'}")
    print()

    # Liste des sessions à traiter
    dlc_out = dlc_output_dir(project)
    if args.session_ids:
        sessions = list(args.session_ids)
    else:
        sessions = sorted(
            d.name for d in dlc_out.iterdir()
            if d.is_dir() and not d.name.startswith(".")
            and any(p.suffix == ".h5" for p in d.iterdir() if p.is_file())
        )

    if args.skip_existing:
        before = len(sessions)
        sessions = [
            s for s in sessions
            if not (out_root / s / f"{s}.h5").exists()
        ]
        print(f"  skip-existing : {before} → {len(sessions)} sessions à traiter\n")

    if not sessions:
        print("Aucune session à traiter.")
        sys.exit(0)

    # Charge le dlc_config une seule fois
    dlc_config = None if args.no_filter else load_dlc_project_config(project)

    n_ok = n_fail = 0
    for i, session_id in enumerate(sessions, 1):
        print(f"[{i}/{len(sessions)}] {session_id}")
        try:
            stats = process_session(
                project, session_id, dlc_config,
                args.likelihood_threshold, args.interp_limit,
                args.window_length, args.no_filter,
            )
            slots = stats.get("total_slots") or 1
            pct_useful = 100 - 100 * stats["n_remaining_nan"] / slots
            print(f"  ✓ {stats['n_frames']} frames | "
                  f"{stats['n_keypoints']} kpts | "
                  f"{pct_useful:.1f}% utilisables après clean | "
                  f"→ {Path(stats['out_path']).name}\n")
            n_ok += 1
        except (FileNotFoundError, ValueError, RuntimeError) as e:
            print(f"  ❌ {e}\n", file=sys.stderr)
            n_fail += 1
            continue

    print(f"✅ {n_ok} sessions OK, {n_fail} échec(s) sur {len(sessions)}")
    if n_ok > 0:
        print(
            "\nÉtape suivante (optionnelle si tu veux boucher les NaN résiduels) :\n"
            f"  python scripts/fill_nan_h5.py --root {out_root}\n"
            "\nPuis setup VAME :\n"
            f"  python scripts/run_vame.py --project-dir {project} setup"
        )


if __name__ == "__main__":
    main()
