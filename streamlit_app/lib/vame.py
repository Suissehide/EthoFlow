"""Lecture des sorties VAME du projet courant.

Layout plat : `<projet>/data/vame/` EST le projet VAME. Il y en a un par
projet EthoFlow — rien à découvrir, rien à sélectionner.

    data/vame/config.yaml
    data/vame/motif_labels.csv
    data/vame/model/
    data/vame/results/<session>/<model>/<algo>-<n>/
    data/vame/analysis/
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from lib.project import SCRIPTS_DIR

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
from paths import vame_dir  # noqa: E402

_ALGO_RE = re.compile(r"^(hmm|kmeans)-(\d+)$")


def vame_project(project: Path) -> Path:
    return vame_dir(Path(project))


def is_initialised(project: Path) -> bool:
    return (vame_project(project) / "config.yaml").is_file()


def read_config(project: Path) -> dict:
    cfg = vame_project(project) / "config.yaml"
    if not cfg.exists():
        return {}
    try:
        return yaml.safe_load(cfg.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}


def n_clusters(project: Path) -> int | None:
    cfg = read_config(project)
    n = cfg.get("n_clusters", cfg.get("n_cluster"))
    return int(n) if n else None


def analysis_dir(project: Path) -> Path:
    return vame_project(project) / "analysis"


def list_sessions(project: Path) -> list[str]:
    results = vame_project(project) / "results"
    if not results.is_dir():
        return []
    return sorted(d.name for d in results.iterdir()
                  if d.is_dir() and not d.name.startswith("."))


def list_algos(project: Path) -> list[str]:
    """Cherche `results/<session>/<model>/<algo>-<n>/`."""
    results = vame_project(project) / "results"
    if not results.is_dir():
        return []
    trouves: set[str] = set()
    for session_dir in results.iterdir():
        if not session_dir.is_dir():
            continue
        for model_dir in session_dir.iterdir():
            if not model_dir.is_dir():
                continue
            for algo_dir in model_dir.iterdir():
                if algo_dir.is_dir() and _ALGO_RE.match(algo_dir.name):
                    trouves.add(algo_dir.name)
    return sorted(trouves)


def parse_algo_n(algo: str) -> tuple[str, int]:
    m = _ALGO_RE.match(algo)
    if not m:
        raise ValueError(f"Format d'algo invalide : {algo!r} (attendu 'hmm-N')")
    return m.group(1), int(m.group(2))


def motif_usage_df(project: Path, algo: str) -> pd.DataFrame:
    """Agrège les `motif_usage_*.npy` en table longue."""
    colonnes = ["session", "motif", "count", "frequency"]
    results = vame_project(project) / "results"
    if not results.is_dir():
        return pd.DataFrame(columns=colonnes)
    lignes: list[dict] = []
    for npy in sorted(results.glob(f"*/*/{algo}/motif_usage_*.npy")):
        session = npy.parents[2].name
        try:
            arr = np.asarray(np.load(npy), dtype=float).ravel()
        except Exception:
            continue
        total = float(arr.sum()) or 1.0
        for motif_id, count in enumerate(arr):
            lignes.append({"session": session, "motif": int(motif_id),
                           "count": float(count), "frequency": float(count) / total})
    return pd.DataFrame(lignes, columns=colonnes)


def session_has_labels(project: Path, session: str) -> bool:
    results = vame_project(project) / "results" / session
    return results.is_dir() and any(results.glob("*/*/*_label_*.npy"))


def stage_status(project: Path) -> dict[str, bool]:
    """Ce qui est déjà fait, pour le stepper de la page VAME."""
    vp = vame_project(project)
    return {
        "setup": is_initialised(project),
        "align": any(vp.glob("data/processed/*_processed.nc")),
        "trainset": (vp / "data" / "train" / "train_seq.npy").exists(),
        "train": any((vp / "model" / "best_model").glob("*.pkl")),
        "segment": any(vp.glob("results/*/*/*/*_label_*.npy")),
    }
