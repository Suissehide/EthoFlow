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
