"""Graphe de contrôle trajectoire : le keypoint demandé doit être respecté,
et son absence doit se voir.

Régression : `--qc-bodypart paw_front_left` (nom inexistant dans le modèle,
qui nomme ce keypoint `front_paw_left`) ne produisait aucun fichier ET
aucun message. Vu de l'utilisateur, le flag « ne faisait rien » — seuls
les anciens graphes `tail_base` restaient dans le dossier.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pose_cleaning import plot_trajectory_qc


def _df(bodyparts: list[str], n: int = 50) -> pd.DataFrame:
    cols, data = [], {}
    for bp in bodyparts:
        for coord in ("x", "y", "likelihood"):
            col = ("scorer", bp, coord)
            cols.append(col)
            data[col] = np.linspace(0, 10, n) if coord != "likelihood" \
                else np.ones(n)
    return pd.DataFrame(data, columns=pd.MultiIndex.from_tuples(
        cols, names=["scorer", "bodyparts", "coords"]))


def test_trace_le_keypoint_demande(tmp_path):
    pytest.importorskip("matplotlib")
    df = _df(["tail_base", "front_paw_left"])
    out = tmp_path / "S1_front_paw_left.png"
    assert plot_trajectory_qc(df, df, "front_paw_left", out) is True
    assert out.is_file()


def test_keypoint_inconnu_leve_avec_la_liste_des_keypoints(tmp_path):
    """Ne pas retourner False en silence : c'est indistinguable d'un succès
    du point de vue de l'appelant, qui n'affiche alors rien du tout."""
    df = _df(["tail_base", "front_paw_left"])
    with pytest.raises(ValueError) as exc:
        plot_trajectory_qc(df, df, "paw_front_left", tmp_path / "x.png")
    message = str(exc.value)
    assert "paw_front_left" in message
    assert "front_paw_left" in message   # les noms disponibles sont listés
    assert "tail_base" in message
