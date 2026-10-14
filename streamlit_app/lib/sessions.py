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

# Clé de metadata qui n'est pas un facteur expérimental : chemin de vidéo, pas une donnée mesurée.
# Les structures imbriquées (dicts, listes) sont filtrées par isinstance(), pas ici.
_NON_FACTEURS = {"source_video"}


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


def qc_trajectories_dir(project: Path) -> Path:
    """Dossier des graphes de trajectoire produits par `prepare_vame_input_custom.py`.

    Un `.png` par (session, keypoint) — voir `parse_qc_trajectory_filename`.
    """
    return dlc_output_dir(Path(project)) / "_qc_trajectories"


def parse_qc_trajectory_filename(filename: str, known_session_ids: list[str]) -> tuple[str, str] | None:
    """`<session_id>_<keypoint>.png` -> `(session_id, keypoint)`, ou `None`.

    Un split naïf sur `_` est faux dans les deux sens : un `session_id` peut
    lui-même contenir des `_` (arène séparée, ex. `BV-970_A1`), et un
    keypoint aussi (`paw_front_left`). La seule frontière fiable est de
    reconnaître le préfixe parmi les `session_id` déjà connus du projet
    (`session_ids(project)`) — jamais une position de caractère devinée.
    Si plusieurs `session_id` connus préfixent le nom (ex. `BV-970` et
    `BV-970_A1` coexistent), le plus long l'emporte : c'est la
    correspondance la plus spécifique.

    `None` si aucun `session_id` connu ne préfixe le nom (fichier orphelin,
    projet dont les sessions ont changé depuis) — jamais une supposition.
    """
    stem = Path(filename).stem
    candidats = [
        sid for sid in known_session_ids
        if stem == sid or stem.startswith(sid + "_")
    ]
    if not candidats:
        return None
    session_id = max(candidats, key=len)
    keypoint = stem[len(session_id) + 1:]
    if not keypoint:
        return None
    return session_id, keypoint


def list_qc_trajectories(project: Path) -> dict[str, dict[str, Path]]:
    """`{keypoint: {session_id: chemin_png}}` pour tous les `.png` reconnus.

    Ne liste que les fichiers dont le `session_id` est reconnu parmi
    `session_ids(project)` — un fichier orphelin (projet renommé, session
    supprimée depuis) est silencieusement ignoré plutôt que mal découpé.
    """
    project = Path(project)
    d = qc_trajectories_dir(project)
    if not d.is_dir():
        return {}
    connus = session_ids(project)
    resultat: dict[str, dict[str, Path]] = {}
    for f in sorted(d.glob("*.png")):
        parsed = parse_qc_trajectory_filename(f.name, connus)
        if parsed is None:
            continue
        session_id, keypoint = parsed
        resultat.setdefault(keypoint, {})[session_id] = f
    return resultat


def arenes_dataframe(meta: dict | None) -> pd.DataFrame:
    """Toutes les clés présentes dans les arènes, sans en présupposer aucune.

    Contrat « prêt à afficher » (ruling R19.1) : chaque colonne est
    uniformément composée de chaînes. `mouse_id` est légitimement `null`
    pour une arène vide (voir `scripts/crop_arenes.py`) — sans cette
    normalisation, une colonne mélangeant un int (arène occupée) et `""`
    (arène vide) fait échouer silencieusement la sérialisation Arrow d'un
    `st.dataframe` en aval (Streamlit rattrape l'exception et reformate la
    colonne à sa façon, sans le dire). Normaliser ici, une fois, protège
    toute page qui affiche ce tableau plutôt que de compter sur chaque
    appelant pour s'en souvenir. Absent -> chaîne vide `""` (sauf `coords`,
    qui garde son `"(à définir)"` dédié).
    """
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
                ligne[cle] = "" if valeur is None else str(valeur)
        lignes.append(ligne)
    return pd.DataFrame(lignes, columns=colonnes)
