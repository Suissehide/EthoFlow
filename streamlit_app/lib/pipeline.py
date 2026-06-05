"""Wrappers de lancement des scripts du pipeline via `conda run`.

Chaque script tourne dans son env conda (`dlc`, `vame`, `ethoflow`). L'app
Streamlit elle-même tourne dans `ethoflow`. Tous les wrappers renvoient un
`subprocess.CompletedProcess` que l'appelant peut afficher via `format_log_output`.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import streamlit as st

from lib.config import CONDA_ENVS, SCRIPTS_DIR


# ============================================================
# Lanceur générique
# ============================================================

def run_in_env(
    env: str,
    script: str,
    args: list[str] | None = None,
    capture: bool = True,
) -> subprocess.CompletedProcess:
    """Lance `conda run -n <env_réel> python <SCRIPTS_DIR>/<script> <args>`.

    `env` est la clé logique (`dlc`, `vame`, `ethoflow`) ; le mapping vers le
    nom d'env conda réel passe par `CONDA_ENVS`.
    """
    real_env = CONDA_ENVS.get(env, env)
    cmd: list[str] = [
        "conda", "run", "-n", real_env, "--no-capture-output",
        "python", str(SCRIPTS_DIR / script),
    ]
    if args:
        cmd.extend(str(a) for a in args)
    return subprocess.run(
        cmd,
        capture_output=capture,
        text=True,
        shell=False,
    )


# ============================================================
# Wrappers spécifiques
# ============================================================

def run_dlc_inference(
    session_ids: list[str],
    mode: str = "superanimal",
    video_adapt: bool = False,
    video_adapt_batch_size: int = 2,
    skip_existing: bool = False,
    all_sessions: bool = False,
) -> subprocess.CompletedProcess:
    """Lance `scripts/run_dlc_inference.py` dans l'env dlc."""
    args: list[str] = []
    if all_sessions:
        args.append("--all")
    else:
        args.extend(session_ids)
    args.extend(["--mode", mode])
    if video_adapt:
        args.append("--video-adapt")
        args.extend(["--video-adapt-batch-size", str(video_adapt_batch_size)])
    if skip_existing:
        args.append("--skip-existing")
    return run_in_env("dlc", "run_dlc_inference.py", args)


def run_filter_keypoints(
    input_dir: str | Path,
    output_dir: str | Path,
    keep: list[str] | None = None,
    drop: list[str] | None = None,
    min_validity: float | None = None,
    dry_run: bool = False,
) -> subprocess.CompletedProcess:
    """Lance `scripts/filter_keypoints.py` dans l'env ethoflow."""
    args = ["--input-dir", str(input_dir), "--output-dir", str(output_dir)]
    if keep:
        args.append("--keep")
        args.extend(keep)
    if drop:
        args.append("--drop")
        args.extend(drop)
    if min_validity is not None:
        args.extend(["--min-validity", str(min_validity)])
    if dry_run:
        args.append("--dry-run")
    return run_in_env("ethoflow", "filter_keypoints.py", args)


def run_fill_nan(
    root: str | Path,
    output_dir: str | Path | None = None,
    dry_run: bool = False,
) -> subprocess.CompletedProcess:
    """Lance `scripts/fill_nan_h5.py` dans l'env ethoflow."""
    args = ["--root", str(root)]
    if output_dir:
        args.extend(["--output-dir", str(output_dir)])
    if dry_run:
        args.append("--dry-run")
    return run_in_env("ethoflow", "fill_nan_h5.py", args)


def run_inspect_session(
    session_ids: list[str] | None = None,
    input_dir: str | Path | None = None,
    fps: float | None = None,
    all_sessions: bool = False,
) -> subprocess.CompletedProcess:
    """Lance `scripts/inspect_session.py` dans l'env ethoflow."""
    args: list[str] = []
    if all_sessions or not session_ids:
        args.append("--all")
    else:
        args.extend(session_ids)
    if input_dir:
        args.extend(["--input-dir", str(input_dir)])
    if fps is not None:
        args.extend(["--fps", str(fps)])
    return run_in_env("ethoflow", "inspect_session.py", args)


def run_trim_empty(
    validity_csv: str | Path,
    h5_input: str | Path,
    h5_output: str | Path,
    video_input: str | Path,
    video_output: str | Path,
    dry_run: bool = False,
) -> subprocess.CompletedProcess:
    """Lance `scripts/trim_empty_arena.py` dans l'env ethoflow."""
    args = [
        "--validity-csv", str(validity_csv),
        "--h5-input", str(h5_input),
        "--h5-output", str(h5_output),
        "--video-input", str(video_input),
        "--video-output", str(video_output),
    ]
    if dry_run:
        args.append("--dry-run")
    return run_in_env("ethoflow", "trim_empty_arena.py", args)


def run_vame_stage(
    stage: str,
    extra_args: list[str] | None = None,
) -> subprocess.CompletedProcess:
    """Lance une sous-commande de `scripts/run_vame.py` dans l'env vame.

    `stage` ∈ {setup, align, trainset, train, evaluate, segment,
               motif-videos, community, info, use, all}.
    """
    args: list[str] = [stage]
    if extra_args:
        args.extend(str(a) for a in extra_args)
    return run_in_env("vame", "run_vame.py", args)


def run_analyze_vame(
    project: str | Path | None = None,
    algo: str = "hmm",
    n_clusters: int | None = None,
    labels_path: str | Path | None = None,
    validity_source: str | Path | None = None,
    mask_empty: bool = False,
    min_edge_frames: int | None = None,
) -> subprocess.CompletedProcess:
    """Lance `scripts/analyze_vame.py` dans l'env ethoflow."""
    args: list[str] = ["--algo", algo]
    if project:
        args.extend(["--project", str(project)])
    if n_clusters is not None:
        args.extend(["--n-clusters", str(n_clusters)])
    if labels_path:
        args.extend(["--labels", str(labels_path)])
    if validity_source:
        args.extend(["--validity-source", str(validity_source)])
    if min_edge_frames is not None:
        args.extend(["--min-edge-frames", str(min_edge_frames)])
    if mask_empty:
        args.append("--mask-empty")
    return run_in_env("ethoflow", "analyze_vame.py", args)


# ============================================================
# Affichage des logs
# ============================================================

def format_log_output(result: subprocess.CompletedProcess) -> None:
    """Affiche stdout/stderr d'un CompletedProcess dans Streamlit."""
    if result.stdout:
        st.code(result.stdout, language="text")
    else:
        st.code("(stdout silencieux)", language="text")
    if result.returncode != 0:
        st.error(f"Code de retour : {result.returncode}")
        if result.stderr:
            st.error(result.stderr)
    elif result.stderr:
        # stderr non vide mais returncode 0 : on garde en warning (logs informatifs)
        with st.expander("stderr (returncode 0)", expanded=False):
            st.code(result.stderr, language="text")
