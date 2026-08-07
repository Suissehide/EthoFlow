"""Prépare les .h5 DLC custom-mode pour VAME (bottom-view, 12 keypoints).

Étapes par session (lit `<project>/data/dlc-output/<session>/...h5` →
écrit `<project>/data/dlc-output/<session>/<session>_clean.h5`) :

    1. dlc.filterpredictions  (median window, défaut 5 frames)
       Smooth temporel : tue les jitters d'une-deux frames sans toucher
       aux mouvements réels. Produit un `*_filtered.h5` à côté du brut.

    2. Likelihood < seuil → NaN  (défaut 0.70)
       La « likelihood » est la confiance que DLC attribue à chaque point,
       entre 0 et 1. Sous le seuil, la position est jugée non fiable :
       elle devient NaN, puis est reconstruite à l'étape 3. Seuil haut =
       sévère (beaucoup de points reconstruits) ; seuil bas = permissif.
       Demandé à l'invite si `--likelihood-threshold` n'est pas passé.

    3. Interpolation linéaire des trous ≤ interp_limit frames (défaut 25)
       Reuses `clean_individual` de assign_arenas.py. Les trous longs
       (>25 frames ≈ 1s à 30 fps) restent NaN — vraies occlusions à
       laisser à VAME (ou à fill_nan_h5.py si besoin).

    4. Écriture en h5 single-animal (key="df_with_missing", format="table")
       dans `<project>/data/dlc-output/<session>/<session>_clean.h5`. C'est
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
    python scripts/fill_nan_h5.py --root <project>/data/dlc-output
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pose_cleaning import clean_dataframe, plot_trajectory_qc  # noqa: E402
from interactive import prompt  # noqa: E402
from paths import (  # noqa: E402
    add_project_dir_arg,
    cleaned_h5_path,
    dlc_output_dir,
    pipeline_config_path,
    raw_dir,
    resolve_project,
)

DEFAULT_LIKELIHOOD = 0.70
DEFAULT_MAX_SPEED = 5.0


def prompt_float(question: str, default: float, explain: str,
                  no_prompt: bool) -> float:
    """Demande un seuil numérique en expliquant d'abord ce qu'il fait."""
    if no_prompt:
        return default
    print()
    print(explain)
    while True:
        raw = prompt(question, default=str(default))
        try:
            return float(raw.replace(",", "."))
        except ValueError:
            print("  ⚠ entre un nombre (ex : 0.70)")


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
    fps: float = 30.0,
    px_per_cm: float | None = None,
    max_speed_ms: float = 5.0,
    detect_sticky: bool = True,
    qc_plot: bool = True,
    qc_bodypart: str = "tail_base",
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

    # 2) Nettoyage multi-méthodes (cf. pose_cleaning.py) :
    #    cutoff likelihood → vitesse aberrante → points collants →
    #    interpolation des frames marquées.
    df = pd.read_hdf(h5_to_clean)
    df_clean, stats = clean_dataframe(
        df,
        fps=fps,
        px_per_cm=px_per_cm,
        likelihood_threshold=likelihood_threshold,
        max_speed_ms=max_speed_ms,
        interp_limit=interp_limit,
        detect_sticky=detect_sticky,
    )

    # 3) Écriture finale : <project>/data/dlc-output/<session>/<session>_clean.h5
    out_path = cleaned_h5_path(project, session_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_hdf(out_path, key="df_with_missing", mode="w", format="table")

    # 4) Graphe de contrôle trajectoire (le critère d'acceptation de Tony :
    #    aucun saut anormal visible sur la trajectoire complète)
    if qc_plot:
        qc_dir = dlc_output_dir(project) / "_qc_trajectories"
        qc_dir.mkdir(parents=True, exist_ok=True)
        ok = plot_trajectory_qc(
            df, df_clean, qc_bodypart,
            qc_dir / f"{session_id}_{qc_bodypart}.png",
            session_id=session_id, stats=stats,
        )
        stats["qc_plot"] = str(qc_dir / f"{session_id}_{qc_bodypart}.png") if ok else None

    stats["raw_h5"] = raw_h5.name
    stats["filtered_h5"] = h5_to_clean.name if not skip_filter else "(skipped)"
    stats["out_path"] = str(out_path)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    add_project_dir_arg(parser)
    parser.add_argument(
        "session_ids", nargs="*",
        help="Sessions à traiter (défaut: toutes celles avec un .h5 dans dlc-output/)",
    )
    parser.add_argument(
        "--likelihood-threshold", type=float, default=None,
        help="Seuil de confiance DLC en dessous duquel (x,y) deviennent NaN "
             "(défaut: 0.70, recommandation Tony/LIN). Demandé à l'invite si "
             "absent. Le cutoff seul n'est qu'un proxy — il est complété par "
             "la détection de vitesse aberrante et de points collants.",
    )
    parser.add_argument(
        "--px-per-cm", type=float, default=None,
        help="Échelle de la caméra en pixels par centimètre. ACTIVE la "
             "détection de vitesse aberrante (le meilleur filtre selon "
             "Tony). Obtiens-la avec calibrate_scale.py. Sans cette valeur "
             "la détection de vitesse est désactivée.",
    )
    parser.add_argument(
        "--max-speed", type=float, default=None,
        help="Vitesse max plausible d'un keypoint en m/s (défaut: 5.0). "
             "Demandé à l'invite si absent. Au-delà, la frame est considérée "
             "comme un saut de tracking et interpolée depuis ses voisines.",
    )
    parser.add_argument(
        "--fps", type=float, default=None,
        help="Framerate des vidéos (défaut: lu depuis la metadata, sinon 30).",
    )
    parser.add_argument(
        "--no-sticky-detection", action="store_true",
        help="Désactive la détection des points collants (coordonnées où "
             "un keypoint bruité atterrit anormalement souvent : reflet IR "
             "fixe, coin d'arène...).",
    )
    parser.add_argument(
        "--no-qc-plot", action="store_true",
        help="Ne génère pas les graphes de contrôle trajectoire.",
    )
    parser.add_argument(
        "--qc-bodypart", default="tail_base",
        help="Keypoint tracé dans le graphe de contrôle (défaut: tail_base, "
             "le plus stable pour juger la trajectoire globale).",
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
        help="Ignore les sessions qui ont déjà un fichier <session>_clean.h5 dans dlc-output/",
    )
    args = parser.parse_args()

    project = resolve_project(args)
    no_prompt = getattr(args, "no_prompt", False)
    out_root = dlc_output_dir(project)
    print(f"Projet     : {project}")
    print(f"Sortie     : {out_root}/<session>/<session>_clean.h5")
    print(f"Interp lim : {args.interp_limit} frames")
    print(f"Filtre DLC : {'OFF (--no-filter)' if args.no_filter else f'median win={args.window_length}'}")

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
            if not cleaned_h5_path(project, s).exists()
        ]
        print(f"  skip-existing : {before} → {len(sessions)} sessions à traiter\n")

    if not sessions:
        print("Aucune session à traiter.")
        sys.exit(0)

    # Charge le dlc_config une seule fois
    dlc_config = None if args.no_filter else load_dlc_project_config(project)

    # ---- fps et échelle px/cm ----
    # fps : CLI > pipeline_config.yaml > 30 par défaut
    fps = args.fps
    px_per_cm = args.px_per_cm
    if fps is None or px_per_cm is None:
        import yaml as _yaml
        cfg_path = pipeline_config_path(project)
        cfg = {}
        if cfg_path.exists():
            with open(cfg_path) as f:
                cfg = _yaml.safe_load(f) or {}
        if fps is None:
            fps = float(cfg.get("fps", 30.0))
        if px_per_cm is None and cfg.get("px_per_cm"):
            px_per_cm = float(cfg["px_per_cm"])

    # ---- Seuils : arguments s'ils sont passés, sinon demandés ----
    likelihood_threshold = args.likelihood_threshold
    if likelihood_threshold is None:
        likelihood_threshold = prompt_float(
            "Seuil de likelihood", DEFAULT_LIKELIHOOD,
            "Seuil de likelihood — la confiance que DLC attribue à chaque\n"
            "point, entre 0 (aucune) et 1 (certaine). En dessous du seuil,\n"
            "la position est jugée non fiable, effacée, puis reconstruite\n"
            "par interpolation depuis les frames voisines.\n"
            "  · plus haut (0.9) = plus sévère, plus de points reconstruits\n"
            "  · plus bas  (0.3) = plus permissif, on garde des points douteux\n"
            f"  · {DEFAULT_LIKELIHOOD} = recommandation de l'équipe VAME/LIN",
            no_prompt,
        )

    max_speed = args.max_speed
    if max_speed is None:
        if px_per_cm:
            max_speed = prompt_float(
                "Vitesse max plausible (m/s)", DEFAULT_MAX_SPEED,
                "Vitesse max — un keypoint qui se déplace plus vite que ça\n"
                "entre deux frames est un saut de tracking, pas un mouvement\n"
                "réel. Attrape les erreurs que la likelihood rate (un point\n"
                "peut être faux ET confiant).\n"
                f"  · {DEFAULT_MAX_SPEED} m/s = recommandation VAME/LIN pour une souris\n"
                "  · 4 m/s = plus strict s'il reste des sauts sur le graphe QC",
                no_prompt,
            )
        else:
            max_speed = DEFAULT_MAX_SPEED

    print()
    print(f"Nettoyage      : cutoff {likelihood_threshold}, "
          f"interp ≤ {args.interp_limit} frames, {fps:.0f} fps")
    if px_per_cm:
        print(f"Échelle        : {px_per_cm:.2f} px/cm → détection de "
              f"vitesse > {max_speed} m/s ACTIVE")
    else:
        print("Échelle        : non renseignée → détection de vitesse "
              "DÉSACTIVÉE")
        print("                 (lance calibrate_scale.py, ou passe "
              "--px-per-cm)")
    print()

    n_ok = n_fail = 0
    for i, session_id in enumerate(sessions, 1):
        print(f"[{i}/{len(sessions)}] {session_id}")
        try:
            stats = process_session(
                project, session_id, dlc_config,
                likelihood_threshold, args.interp_limit,
                args.window_length, args.no_filter,
                fps=fps, px_per_cm=px_per_cm, max_speed_ms=max_speed,
                detect_sticky=not args.no_sticky_detection,
                qc_plot=not args.no_qc_plot, qc_bodypart=args.qc_bodypart,
            )
            slots = stats["n_frames"] * max(stats["n_keypoints"], 1)
            pct_useful = 100 - 100 * stats["n_remaining_nan"] / slots
            detail = [f"cutoff {stats['n_low_likelihood']}"]
            if stats["velocity_enabled"]:
                detail.append(f"vitesse {stats['n_velocity_outliers']}")
            if stats["n_sticky"]:
                detail.append(f"collants {stats['n_sticky']}")
            print(f"  ✓ {stats['n_frames']} frames | "
                  f"{stats['n_keypoints']} kpts | "
                  f"marquées : {', '.join(detail)} | "
                  f"réparées {stats['n_repaired']} | "
                  f"{pct_useful:.1f}% utilisables | "
                  f"→ {Path(stats['out_path']).name}")
            if stats.get("sticky_points"):
                for bp, pts in list(stats["sticky_points"].items())[:3]:
                    coords = ", ".join(f"({x:.0f},{y:.0f})" for x, y in pts[:2])
                    print(f"      · point collant sur {bp} : {coords}")
            print()
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
