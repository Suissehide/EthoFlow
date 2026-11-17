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


def test_axes_figes_sur_l_image_de_la_video(tmp_path):
    pytest.importorskip("matplotlib")
    df = _df(["front_paw_left", "nose"])
    out = tmp_path / "qc.png"
    assert plot_trajectory_qc(df, df, "front_paw_left", out, frame_size=(640, 480)) is True
    assert out.stat().st_size > 0


def test_taille_image_lue_dans_la_metadata(tmp_path):
    import yaml
    from prepare_vame_input_custom import taille_image
    (tmp_path / "data" / "raw" / "S1").mkdir(parents=True)
    (tmp_path / "data" / "raw" / "S1" / "metadata.yaml").write_text(
        yaml.safe_dump({"source_video": "/nulle/part.mp4",
                        "camera": {"width": 1280, "height": 720}}))
    assert taille_image(tmp_path, "S1") == (1280, 720)
    (tmp_path / "data" / "raw" / "S2").mkdir()
    (tmp_path / "data" / "raw" / "S2" / "metadata.yaml").write_text(
        yaml.safe_dump({"source_video": "/nulle/part.mp4"}))
    assert taille_image(tmp_path, "S2") is None      # ni metadata, ni vidéo lisible
    assert taille_image(tmp_path, "S3") is None      # pas de metadata du tout
