"""Lecture des sessions EthoFlow et de leurs metadata.

Centralise la logique d'inventaire qui était dans l'ancien `app.py` monofichier,
en l'enrichissant des statuts post-cleanup (vame-input clean) et de la disponibilité
de validity_per_session.csv pour les projets VAME connus.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from lib.config import (
    dlc_output_dir,
    raw_dir,
    vame_input_dir,
    vame_output_dir,
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
    """Vrai si on trouve un dossier vame-input avec un suffixe -clean pour cette session."""
    vi = vame_input_dir()
    if not vi.exists():
        return False
    candidates = list(vi.glob(f"*clean*/{session_id}"))
    candidates += list(vi.glob(f"{session_id}-clean"))
    return any(c.exists() for c in candidates)


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

        vi = vame_input_dir()
        vame_input_session = vi / session_id
        split_done = (
            vame_input_session.exists()
            and any(vame_input_session.glob(f"{session_id}_A*.h5"))
        )

        rows.append({
            "session_id": session_id,
            "timepoint": meta.get("timepoint", "—"),
            "date": meta.get("date", "—"),
            "animaux": n_animals,
            "vidéo": "OK" if video_ok else "manque",
            "DLC": "OK" if (dlc_output_dir() / session_id).exists() else "—",
            "split": "OK" if split_done else "—",
            "VAME": "OK" if (vame_output_dir() / session_id).exists() else "—",
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
