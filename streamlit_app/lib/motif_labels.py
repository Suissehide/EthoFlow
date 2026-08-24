"""Accès unique à `<projet>/data/vame/motif_labels.csv`.

Format défini par `run_vame.py` : séparateur `;`, encodage `utf-8-sig`
(Excel sur Windows ouvre correctement les accents), 8 colonnes.

L'app ne fabrique pas ce fichier à partir de rien — c'est
`run_vame motif-videos` ou `run_vame motif-labels` qui le génère avec
`usage_pct` et `video` pré-remplis. L'app met à jour des lignes
existantes, et reprend un ancien YAML si l'utilisateur le demande.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import yaml

from lib.project import SCRIPTS_DIR
from lib.vame import vame_project

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
import run_vame as _run_vame  # noqa: E402

SEP = ";"
ENCODING = "utf-8-sig"

# Source de vérité côté scripts — une copie divergerait au premier changement.
COLUMNS: list[str] = list(_run_vame.MOTIF_LABELS_COLUMNS)


def categories() -> list[str]:
    """Liste fermée écrite dans la colonne `category`, groupée par les analyses."""
    return list(_run_vame.ETHOGRAM_CATEGORIES)


def path(project: Path) -> Path:
    return vame_project(project) / "motif_labels.csv"


def exists(project: Path) -> bool:
    return path(project).is_file()


def load(project: Path) -> pd.DataFrame | None:
    p = path(project)
    if not p.is_file():
        return None
    # dtype=str + keep_default_na : une cellule vide reste "" et non NaN,
    # sinon on réécrirait "nan" dans le fichier de l'utilisateur.
    return pd.read_csv(p, sep=SEP, dtype=str, keep_default_na=False,
                       encoding=ENCODING)


def save(project: Path, df: pd.DataFrame) -> None:
    df.to_csv(path(project), sep=SEP, index=False, encoding=ENCODING)


def set_fields(project: Path, motif_id: int, **champs: str) -> None:
    """Met à jour une ligne. Les colonnes inconnues du CSV sont ignorées."""
    df = load(project)
    if df is None:
        return
    masque = df["motif_id"].astype(str) == str(motif_id)
    for colonne, valeur in champs.items():
        if colonne in df.columns:
            df.loc[masque, colonne] = "" if valeur is None else str(valeur)
    save(project, df)


def video_path(project: Path, row) -> Path | None:
    """Chemin absolu du clip, ou None s'il n'existe pas.

    La colonne `video` contient un chemin relatif au projet VAME, déjà
    résolu par `run_vame.find_motif_video()` — inutile de le deviner.
    """
    rel = str(row.get("video") or "").strip()
    if not rel:
        return None
    candidat = vame_project(project) / rel
    return candidat if candidat.is_file() else None


# ------------------------------------------------------- reprise ancien format

def legacy_yaml_files(project: Path) -> list[Path]:
    """Anciens `analysis/motif_labels_<algo>.yaml` écrits par l'app v1."""
    analysis = vame_project(project) / "analysis"
    if not analysis.is_dir():
        return []
    return sorted(analysis.glob("motif_labels_*.yaml"))


def migrate_from_yaml(project: Path, yaml_path: Path) -> int:
    """Recopie les labels d'un ancien YAML dans le CSV. Retourne le nombre repris.

    N'écrase que la colonne `label`, et seulement pour les motifs présents
    dans le YAML.
    """
    df = load(project)
    if df is None:
        return 0
    brut = yaml.safe_load(Path(yaml_path).read_text(encoding="utf-8")) or {}
    repris = 0
    for cle, valeur in brut.items():
        texte = str(valeur or "").strip()
        if not texte:
            continue
        masque = df["motif_id"].astype(str) == str(cle)
        if masque.any():
            df.loc[masque, "label"] = texte
            repris += 1
    save(project, df)
    return repris
