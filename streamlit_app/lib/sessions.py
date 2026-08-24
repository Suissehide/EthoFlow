"""Inventaire des sessions et de leur avancement — sans Streamlit.

Aucune colonne de metadata n'est connue à l'avance : `sync_from_excel.py`
recopie *toutes* les colonnes de l'Excel, y compris celles que
l'utilisateur invente. Une app qui n'afficherait que des colonnes connues
casserait cette promesse.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import yaml

from lib.project import SCRIPTS_DIR
from lib.vame import session_has_labels

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
from paths import cleaned_h5_path, dlc_output_dir, raw_dir  # noqa: E402

# Clés de metadata qui ne sont pas des facteurs expérimentaux.
_NON_FACTEURS = {"source_video", "arenes", "camera", "video"}


def session_ids(project: Path) -> list[str]:
    rd = raw_dir(Path(project))
    if not rd.is_dir():
        return []
    return sorted(d.name for d in rd.iterdir()
                  if d.is_dir() and not d.name.startswith("."))


def load_metadata(project: Path, session_id: str) -> dict | None:
    p = raw_dir(Path(project)) / session_id / "metadata.yaml"
    if not p.is_file():
        return None
    try:
        return yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except Exception:
        return None


def metadata_fields(meta: dict | None) -> dict:
    """Clés scalaires de la metadata, dans l'ordre du fichier.

    Exclut les chemins et les structures imbriquées : ce qui reste est
    exactement ce qui peut servir d'axe de comparaison dans analyze_vame.
    """
    if not meta:
        return {}
    return {
        k: v for k, v in meta.items()
        if k not in _NON_FACTEURS and not isinstance(v, (dict, list))
    }


def _dlc_ok(project: Path, session_id: str) -> bool:
    """Un dossier vide ne compte pas : il faut un .h5 produit."""
    d = dlc_output_dir(Path(project)) / session_id
    return d.is_dir() and any(d.glob("*.h5"))


def _split_ok(project: Path, session_id: str) -> bool:
    d = dlc_output_dir(Path(project)) / session_id
    return d.is_dir() and any(d.glob(f"{session_id}_A*.h5"))


def list_sessions(project: Path) -> pd.DataFrame:
    project = Path(project)
    colonnes = ["session_id", "vidéo", "DLC", "split", "nettoyage", "VAME"]
    lignes: list[dict] = []
    for session_id in session_ids(project):
        meta = load_metadata(project, session_id) or {}
        source = meta.get("source_video")
        lignes.append({
            "session_id": session_id,
            "vidéo": "OK" if source and Path(source).exists() else "manque",
            "DLC": "OK" if _dlc_ok(project, session_id) else "—",
            "split": "OK" if _split_ok(project, session_id) else "—",
            "nettoyage": "OK" if cleaned_h5_path(project, session_id).exists() else "—",
            "VAME": "OK" if session_has_labels(project, session_id) else "—",
        })
    return pd.DataFrame(lignes, columns=colonnes)


def arenes_dataframe(meta: dict | None) -> pd.DataFrame:
    """Toutes les clés présentes dans les arènes, sans en présupposer aucune."""
    arenes = (meta or {}).get("arenes") or []
    if not arenes:
        return pd.DataFrame()
    colonnes: list[str] = []
    for arene in arenes:
        for cle in arene:
            if cle not in colonnes:
                colonnes.append(cle)
    lignes = []
    for arene in arenes:
        ligne = {}
        for cle in colonnes:
            valeur = arene.get(cle)
            if cle == "coords":
                ligne[cle] = str(valeur) if valeur else "(à définir)"
            else:
                ligne[cle] = "" if valeur is None else valeur
        lignes.append(ligne)
    return pd.DataFrame(lignes, columns=colonnes)
