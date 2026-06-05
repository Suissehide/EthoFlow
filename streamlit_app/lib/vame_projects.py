"""Découverte des projets VAME et lecture de leurs sorties.

Un projet VAME est un dossier contenant un `config.yaml` à sa racine et,
après segmentation, des sous-dossiers `results/<session>/VAME/<algo>-<n>/`.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from lib.config import CONFIG_POINTER


# ============================================================
# Projets
# ============================================================

def discover_projects(root: Path) -> list[Path]:
    """Liste les sous-dossiers de `root` qui contiennent un `config.yaml`."""
    if not root.exists():
        return []
    return sorted(
        d for d in root.iterdir()
        if d.is_dir() and (d / "config.yaml").exists()
    )


def get_current_project_path() -> Path | None:
    """Renvoie le dossier du projet VAME pointé par `<repo>/.vame_config_path`."""
    if not CONFIG_POINTER.exists():
        return None
    cfg = CONFIG_POINTER.read_text().strip()
    if not cfg:
        return None
    return Path(cfg).parent


# ============================================================
# Sessions / algos d'un projet
# ============================================================

def list_sessions_in_project(project: Path) -> list[str]:
    """Liste les sous-dossiers de `<project>/results/`."""
    results_dir = project / "results"
    if not results_dir.exists():
        return []
    return sorted(
        d.name for d in results_dir.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    )


_ALGO_RE = re.compile(r"^(hmm|kmeans)-(\d+)$")


def list_algos(project: Path) -> list[str]:
    """Retourne les algos disponibles, e.g. ['hmm-15', 'kmeans-15'].

    Cherche dans `<project>/results/<n'importe quelle session>/VAME/<algo>-<n>/`.
    """
    results_dir = project / "results"
    if not results_dir.exists():
        return []
    found: set[str] = set()
    for session_dir in results_dir.iterdir():
        if not session_dir.is_dir():
            continue
        vame_dir = session_dir / "VAME"
        if not vame_dir.exists():
            continue
        for algo_dir in vame_dir.iterdir():
            if algo_dir.is_dir() and _ALGO_RE.match(algo_dir.name):
                found.add(algo_dir.name)
    return sorted(found)


def parse_algo_n(algo: str) -> tuple[str, int]:
    """'hmm-15' → ('hmm', 15). Lève ValueError si format invalide."""
    m = _ALGO_RE.match(algo)
    if not m:
        raise ValueError(f"Format d'algo invalide : {algo!r} (attendu 'hmm-N' ou 'kmeans-N')")
    return m.group(1), int(m.group(2))


# ============================================================
# Motif videos
# ============================================================

def motif_video_path(project: Path, session: str, algo: str, motif_id: int) -> Path:
    """Chemin canonique de la cluster_video d'un motif pour une session donnée."""
    return (
        project / "results" / session / "VAME" / algo
        / "cluster_videos" / f"{session}-motif_{motif_id}.mp4"
    )


def find_any_motif_video(project: Path, algo: str, motif_id: int) -> Path | None:
    """Renvoie la première cluster_video trouvée pour ce motif, en scannant toutes les sessions.

    Le contenu d'une cluster_video est identique d'une session à l'autre, n'importe
    laquelle fait l'affaire pour la labellisation.
    """
    for session in list_sessions_in_project(project):
        candidate = motif_video_path(project, session, algo, motif_id)
        if candidate.exists():
            return candidate
        # Fallback : pattern moins strict (selon la version VAME, le nom peut varier)
        cluster_dir = project / "results" / session / "VAME" / algo / "cluster_videos"
        if cluster_dir.exists():
            matches = list(cluster_dir.glob(f"*motif_{motif_id}.mp4"))
            if matches:
                return matches[0]
            matches = list(cluster_dir.glob(f"*motif_{motif_id}_*.mp4"))
            if matches:
                return matches[0]
    return None


# ============================================================
# Stats d'usage
# ============================================================

def motif_usage_df(project: Path, algo: str) -> pd.DataFrame:
    """Agrège les motif_usage_<session>.npy en DataFrame long.

    Colonnes : session, motif, count, frequency.
    """
    rows: list[dict] = []
    results_dir = project / "results"
    if not results_dir.exists():
        return pd.DataFrame(columns=["session", "motif", "count", "frequency"])

    for session in list_sessions_in_project(project):
        algo_dir = results_dir / session / "VAME" / algo
        if not algo_dir.exists():
            continue
        # VAME 0.13+ utilise généralement motif_usage_<session>.npy
        npy_candidates = list(algo_dir.glob("motif_usage*.npy"))
        if not npy_candidates:
            continue
        # On prend le premier (typiquement un seul)
        try:
            arr = np.load(npy_candidates[0])
        except Exception:
            continue
        total = float(arr.sum()) if arr.sum() > 0 else 1.0
        for motif_id, count in enumerate(arr):
            rows.append({
                "session": session,
                "motif": int(motif_id),
                "count": int(count),
                "frequency": float(count) / total,
            })
    return pd.DataFrame(rows)


def validity_per_session_df(project: Path) -> pd.DataFrame | None:
    """Lit `<project>/analysis/validity_per_session.csv` si existe."""
    csv = project / "analysis" / "validity_per_session.csv"
    if not csv.exists():
        return None
    try:
        return pd.read_csv(csv)
    except Exception:
        return None
