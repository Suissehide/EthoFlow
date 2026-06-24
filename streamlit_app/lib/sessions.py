"""Lecture des sessions EthoFlow et de leurs metadata.

Centralise la logique d'inventaire qui était dans l'ancien `app.py` monofichier,
en l'enrichissant des statuts post-cleanup (h5 _clean dans dlc-output) et de
la disponibilité de validity_per_session.csv pour les projets VAME connus.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from lib.config import (
    cleaned_h5_path,
    dlc_output_dir,
    raw_dir,
    vame_dir,
    vame_projects_root,
)


def load_metadata(session_id: str) -> dict | None:
    """Renvoie le dict du `metadata.yaml` d'une session, ou None s'il n'existe pas."""
    path = raw_dir() / session_id / "metadata.yaml"
    if not path.exists():
        return None
    with open(path) as f:
        return yaml.safe_load(f)


def _cleanup_done(session_id: str) -> bool:
    """Vrai si le h5 nettoyé existe pour cette session.

    Convention : `<project>/data/dlc-output/<session>/<session>_clean.h5`
    (produit par `prepare_vame_input_custom.py`).
    """
    return cleaned_h5_path(session_id).exists()


def _validity_available(session_id: str) -> bool:
    """Vrai si au moins un projet VAME a un validity_per_session.csv contenant cette session."""
    root = vame_projects_root()
    if not root.exists():
        return False
    for project_dir in root.iterdir():
        if not project_dir.is_dir():
            continue
        csv = project_dir / "analysis" / "validity_per_session.csv"
        if not csv.exists():
            continue
        try:
            df = pd.read_csv(csv)
        except Exception:
            continue
        for col in ("session_full", "session", "session_id"):
            if col in df.columns and df[col].astype(str).str.contains(session_id).any():
                return True
    return False


def list_sessions() -> pd.DataFrame:
    """Liste les sessions, leur metadata et leur statut d'avancement enrichi."""
    rd = raw_dir()
    if not rd.exists():
        return pd.DataFrame()

    rows: list[dict] = []
    for session_dir in sorted(rd.iterdir()):
        if not session_dir.is_dir() or session_dir.name.startswith("."):
            continue
        session_id = session_dir.name
        meta = load_metadata(session_id) or {}
        source_video = meta.get("source_video")
        video_ok = bool(source_video and Path(source_video).exists())

        n_animals = sum(
            1 for a in meta.get("arenes", []) if a.get("mouse_id") is not None
        )

        dlc_session = dlc_output_dir() / session_id
        split_done = (
            dlc_session.exists()
            and any(dlc_session.glob(f"{session_id}_A*.h5"))
        )

        rows.append({
            "session_id": session_id,
            "timepoint": meta.get("timepoint", "—"),
            "date": meta.get("date", "—"),
            "animaux": n_animals,
            "vidéo": "OK" if video_ok else "manque",
            "DLC": "OK" if (dlc_output_dir() / session_id).exists() else "—",
            "split": "OK" if split_done else "—",
            "VAME": "OK" if (vame_dir() / session_id).exists() else "—",
            "cleanup": "OK" if _cleanup_done(session_id) else "—",
            "validity": "OK" if _validity_available(session_id) else "—",
        })
    return pd.DataFrame(rows)


def arenes_dataframe(meta: dict) -> pd.DataFrame:
    """Tableau d'arènes pour affichage : Arène / MouseID / Condition / etc."""
    rows = []
    for ar in meta.get("arenes", []):
        rows.append({
            "Arène": ar.get("id"),
            "MouseID": ar.get("mouse_id"),
            "Condition": ar.get("condition"),
            "Stress": "+" if ar.get("stress") else "",
            "ANGII": "+" if ar.get("angii") else "",
            "Coords": ar.get("coords") or "(à définir)",
            "MouseTrialCode": ar.get("mouse_trial_code") or "",
        })
    return pd.DataFrame(rows)
