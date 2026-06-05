"""Lecture/écriture du YAML de labels de motifs VAME.

Le fichier vit dans `<projet>/analysis/motif_labels_<algo>.yaml` et est consommé
par `analyze_vame.py --labels` à l'identique.
"""
from __future__ import annotations

from pathlib import Path

import yaml


def labels_path(project: Path, algo: str) -> Path:
    """Chemin canonique du YAML de labels pour ce projet × algo."""
    return project / "analysis" / f"motif_labels_{algo}.yaml"


def load_labels(project: Path, algo: str) -> dict[int, str]:
    """Charge {motif_id: label}. Retourne {} si fichier absent.

    Tolère les clés en str (parse les ints) — utile quand on a édité à la main.
    """
    path = labels_path(project, algo)
    if not path.exists():
        return {}
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    out: dict[int, str] = {}
    for k, v in raw.items():
        try:
            out[int(k)] = str(v) if v is not None else ""
        except (TypeError, ValueError):
            # Clé non-int : on ignore plutôt que crasher (le YAML peut contenir
            # un commentaire mal parsé ou une entrée libre).
            continue
    return out


def save_labels(project: Path, algo: str, mapping: dict[int, str]) -> None:
    """Sauvegarde {motif_id: label} en YAML trié par motif_id, allow_unicode."""
    path = labels_path(project, algo)
    path.parent.mkdir(parents=True, exist_ok=True)
    # On normalise : on vire les entrées vides et on trie
    cleaned = {
        int(k): str(v)
        for k, v in sorted(mapping.items(), key=lambda kv: int(kv[0]))
        if v is not None and str(v).strip() != ""
    }
    with open(path, "w") as f:
        yaml.safe_dump(
            cleaned,
            f,
            allow_unicode=True,
            sort_keys=False,
            default_flow_style=False,
        )


def motif_display(motif_id: int, labels: dict[int, str]) -> str:
    """Format d'affichage : '3: grooming face' si labellé, sinon 'motif_3'."""
    label = labels.get(int(motif_id), "")
    if label:
        return f"{motif_id}: {label}"
    return f"motif_{motif_id}"
