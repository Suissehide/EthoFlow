"""`inspect_session.py` doit voir les deux voies, pas seulement le topview.

Le script ne cherchait que `<session>_A*.h5` : sur un projet bottomview il
levait « Aucun .h5 single-animal » et `--all` ne listait rien, alors que le
dossier était plein. Même angle mort que celui corrigé dans `find_pairs`.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import inspect_session as IS


DLC_SUFFIX = "DLC_Resnet50_bottomviewMCCshuffle1_snapshot-200"


def _write_h5(path: Path, *, bodyparts=("nose", "tail_base"), n=10) -> Path:
    """Un .h5 DLC plausible : MultiIndex scorer/bodyparts/coords + likelihood."""
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = pd.MultiIndex.from_tuples(
        [("scorer", bp, c) for bp in bodyparts for c in ("x", "y", "likelihood")],
        names=["scorer", "bodyparts", "coords"])
    df = pd.DataFrame(1.0, index=range(n), columns=cols)
    df.to_hdf(path, key="df_with_missing", mode="w", format="table")
    return path


@pytest.fixture
def dlc_out(tmp_path) -> Path:
    d = tmp_path / "data" / "dlc-output"
    d.mkdir(parents=True)
    return d


# ----------------------------------------------------------------------
# Découverte des .h5
# ----------------------------------------------------------------------

def test_bottomview_brut_est_inspectable(dlc_out):
    brut = _write_h5(dlc_out / "BV-961" / f"BV-961{DLC_SUFFIX}.h5")
    assert IS.resolve_h5_files(dlc_out / "BV-961", "BV-961") == [
        ("Single-animal", brut)]


def test_bottomview_prefere_le_clean(dlc_out):
    _write_h5(dlc_out / "BV-961" / f"BV-961{DLC_SUFFIX}.h5")
    clean = _write_h5(dlc_out / "BV-961" / "BV-961_clean.h5")
    assert IS.resolve_h5_files(dlc_out / "BV-961", "BV-961") == [
        ("Single-animal", clean)]


def test_topview_garde_ses_arenes(dlc_out):
    for arena in ("A1", "A2"):
        _write_h5(dlc_out / "S1" / f"S1_{arena}.h5")
    labels = [lab for lab, _ in IS.resolve_h5_files(dlc_out / "S1", "S1")]
    assert labels == ["Arène A1", "Arène A2"]


def test_dossier_sans_h5(dlc_out):
    (dlc_out / "BV-961").mkdir()
    assert IS.resolve_h5_files(dlc_out / "BV-961", "BV-961") == []


# ----------------------------------------------------------------------
# --all
# ----------------------------------------------------------------------

def test_list_sessions_voit_les_deux_voies(dlc_out):
    _write_h5(dlc_out / "BV-961" / f"BV-961{DLC_SUFFIX}.h5")
    _write_h5(dlc_out / "S1" / "S1_A1.h5")
    (dlc_out / "vide").mkdir()
    assert IS.list_sessions(dlc_out) == ["BV-961", "S1"]


# ----------------------------------------------------------------------
# Rapport
# ----------------------------------------------------------------------

def test_rapport_bottomview_sort_un_verdict(dlc_out, tmp_path, capsys):
    _write_h5(dlc_out / "BV-961" / f"BV-961{DLC_SUFFIX}.h5")
    IS.inspect_session(tmp_path, "BV-961", dlc_out, fps=25.0)
    out = capsys.readouterr().out
    assert "Single-animal" in out
    assert "Verdict" in out


def test_session_sans_h5_leve_une_erreur_explicite(dlc_out, tmp_path):
    (dlc_out / "BV-961").mkdir()
    with pytest.raises(FileNotFoundError, match="Aucun .h5"):
        IS.inspect_session(tmp_path, "BV-961", dlc_out, fps=25.0)


def test_inspection_en_tableaux(dlc_out, tmp_path):
    from lib.sessions import inspection
    _write_h5(dlc_out / "BV-961" / f"BV-961{DLC_SUFFIX}.h5")
    sessions, keypoints = inspection(tmp_path)
    assert list(sessions["Session"]) == ["BV-961"]
    assert sessions.loc[0, "Fichier"] == f"BV-961{DLC_SUFFIX}.h5"
    assert "Verdict" in sessions and list(keypoints.columns) == ["BV-961"]


def test_verdict_sur_les_images_completes_pas_sur_la_couverture():
    """Couverture à 100 % mais un tiers d'images complètes : pas « excellent »."""
    v = IS.verdict(33.0, {"nose": 0.99, "tail_mid": 0.29, "tail_base": 1.0})
    assert v.startswith("❌") and "limité par tail_mid (29 %)" in v
    assert IS.verdict(95.0, {"nose": 0.99}).startswith("✅")
    assert "limité" not in IS.verdict(95.0, {"nose": 0.95})


def test_points_ignores_exclus_des_images_completes(tmp_path):
    """Un point flou à 10 % ne doit plus faire tomber le total s'il est ignoré."""
    import numpy as np
    import pandas as pd
    cols = pd.MultiIndex.from_product([["DLC"], ["nose", "tail_mid"], ["x", "y", "likelihood"]],
                                      names=["scorer", "bodyparts", "coords"])
    v = np.ones((20, 6)); v[2:, 3] = np.nan          # tail_mid présent sur 2 images / 20
    h5 = tmp_path / "s.h5"; pd.DataFrame(v, columns=cols).to_hdf(h5, key="df")
    assert IS.analyze_h5(h5)["frames_all_valid"] == 2
    s = IS.analyze_h5(h5, ignorer=("tail_mid",))
    assert s["frames_all_valid"] == 20 and s["n_kp"] == 1
    assert s["per_kp_pct"]["tail_mid"] == 0.1        # toujours affiché
