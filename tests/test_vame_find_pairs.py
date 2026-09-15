"""Appariement (vidéo, .h5) de `run_vame.py setup`.

Le nettoyage (`prepare_vame_input_custom.py`) est de l'assurance qualité,
pas un passage obligé : `setup` doit savoir repartir des `.h5` bruts que
DeepLabCut dépose dans `dlc-output/<session>/`. Quand plusieurs candidats
cohabitent, l'ordre de préférence est explicite et testé ici.
"""
from __future__ import annotations

import os
from pathlib import Path

import run_vame as RV


DLC_SUFFIX = "DLC_Resnet50_bottomviewMCCshuffle1_snapshot-200"


def _touch(path: Path, mtime: float | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x00")
    if mtime is not None:
        os.utime(path, (mtime, mtime))
    return path


# ----------------------------------------------------------------------
# Choix du .h5 bottomview
# ----------------------------------------------------------------------

def test_h5_dlc_brut_accepte_sans_nettoyage(tmp_path):
    """Le cas qui bloquait : inférence DLC faite, nettoyage sauté."""
    sdir = tmp_path / "BV-961"
    brut = _touch(sdir / f"BV-961{DLC_SUFFIX}.h5")
    assert RV.pick_bottomview_h5(sdir, "BV-961") == brut


def test_clean_prioritaire_sur_le_brut(tmp_path):
    sdir = tmp_path / "BV-961"
    _touch(sdir / f"BV-961{DLC_SUFFIX}.h5")
    clean = _touch(sdir / "BV-961_clean.h5")
    assert RV.pick_bottomview_h5(sdir, "BV-961") == clean


def test_filtered_prefere_au_brut(tmp_path):
    """`dlc.filterpredictions` a tourné : sa sortie vaut mieux que le brut."""
    sdir = tmp_path / "BV-961"
    _touch(sdir / f"BV-961{DLC_SUFFIX}.h5")
    filtered = _touch(sdir / f"BV-961{DLC_SUFFIX}_filtered.h5")
    assert RV.pick_bottomview_h5(sdir, "BV-961") == filtered


def test_entre_bruts_le_plus_recent_gagne(tmp_path):
    """--video-adapt écrit deux .h5 sans suffixe distinctif : l'adapté est le second."""
    sdir = tmp_path / "BV-961"
    _touch(sdir / f"BV-961{DLC_SUFFIX}.h5", mtime=1_000_000)
    adapte = _touch(sdir / f"BV-961{DLC_SUFFIX}_adapted.h5", mtime=2_000_000)
    assert RV.pick_bottomview_h5(sdir, "BV-961") == adapte


def test_dossier_sans_h5(tmp_path):
    sdir = tmp_path / "BV-961"
    sdir.mkdir()
    _touch(sdir / f"BV-961{DLC_SUFFIX}_labeled.mp4")
    assert RV.pick_bottomview_h5(sdir, "BV-961") is None


# ----------------------------------------------------------------------
# find_pairs de bout en bout
# ----------------------------------------------------------------------

def _bottomview_project(tmp_path, *, h5_names: list[str]) -> tuple[Path, Path, Path, Path]:
    import yaml
    data = tmp_path / "data"
    video = tmp_path / "videos" / "961.mp4"
    _touch(video)
    for name in h5_names:
        _touch(data / "dlc-output" / "BV-961" / name)
    raw = data / "raw" / "BV-961"
    raw.mkdir(parents=True)
    (raw / "metadata.yaml").write_text(
        yaml.safe_dump({"id": "BV-961", "source_video": str(video)}),
        encoding="utf-8",
    )
    (data / "cropped").mkdir(parents=True, exist_ok=True)
    return data / "dlc-output", data / "cropped", data / "raw", video


def test_find_pairs_apparie_un_h5_brut(tmp_path):
    dlc_out, cropped, raw, video = _bottomview_project(
        tmp_path, h5_names=[f"BV-961{DLC_SUFFIX}.h5"])
    pairs = RV.find_pairs(dlc_out, cropped, raw_root=raw)
    assert len(pairs) == 1
    assert pairs[0][0] == video
    assert pairs[0][1].name == f"BV-961{DLC_SUFFIX}.h5"


def test_find_pairs_prefere_le_clean(tmp_path):
    dlc_out, cropped, raw, _ = _bottomview_project(
        tmp_path, h5_names=[f"BV-961{DLC_SUFFIX}.h5", "BV-961_clean.h5"])
    pairs = RV.find_pairs(dlc_out, cropped, raw_root=raw)
    assert [p[1].name for p in pairs] == ["BV-961_clean.h5"]


def test_find_pairs_topview_inchange(tmp_path):
    """Le multi-animal continue de passer par les .h5 par arène."""
    data = tmp_path / "data"
    for arena in ("A1", "A2"):
        _touch(data / "dlc-output" / "S1" / f"S1_{arena}.h5")
        _touch(data / "cropped" / "S1" / f"S1_{arena}.mp4")
    (data / "raw").mkdir(parents=True, exist_ok=True)
    pairs = RV.find_pairs(data / "dlc-output", data / "cropped",
                          raw_root=data / "raw")
    assert sorted(p[1].name for p in pairs) == ["S1_A1.h5", "S1_A2.h5"]


# ----------------------------------------------------------------------
# Seuil de likelihood
# ----------------------------------------------------------------------

def test_default_likelihood_a_06():
    import pose_cleaning
    import prepare_vame_input_custom as P
    assert pose_cleaning.DEFAULT_LIKELIHOOD == 0.6
    assert P.DEFAULT_LIKELIHOOD == 0.6


# ----------------------------------------------------------------------
# Rekey non destructif
# ----------------------------------------------------------------------

def _write_h5(path: Path, key: str) -> Path:
    import pandas as pd
    cols = pd.MultiIndex.from_tuples(
        [("scorer", "nose", "x"), ("scorer", "nose", "y")],
        names=["scorer", "bodyparts", "coords"])
    pd.DataFrame([[1.0, 2.0], [3.0, 4.0]], columns=cols).to_hdf(
        path, key=key, mode="w", format="table")
    return path


def test_brut_mal_cle_est_copie_pas_ecrase(tmp_path):
    """Réécrire en place détruirait la sortie DLC d'origine (heures de GPU)."""
    from rekey_h5 import is_already_correct
    sdir = tmp_path / "BV-961"
    sdir.mkdir()
    src = _write_h5(sdir / f"BV-961{DLC_SUFFIX}.h5", key="df")

    out = RV.ensure_vame_key(src, "BV-961")

    assert out == sdir / "BV-961_vame.h5"
    assert is_already_correct(out)
    assert not is_already_correct(src)


def test_fichier_du_pipeline_rekeye_en_place(tmp_path):
    """`_clean.h5` est notre sortie : la réécrire ne perd rien."""
    from rekey_h5 import is_already_correct
    sdir = tmp_path / "BV-961"
    sdir.mkdir()
    src = _write_h5(sdir / "BV-961_clean.h5", key="df")

    assert RV.ensure_vame_key(src, "BV-961") == src
    assert is_already_correct(src)


def test_bonne_cle_ne_touche_a_rien(tmp_path):
    sdir = tmp_path / "BV-961"
    sdir.mkdir()
    src = _write_h5(sdir / f"BV-961{DLC_SUFFIX}.h5", key="df_with_missing")
    mtime = src.stat().st_mtime

    assert RV.ensure_vame_key(src, "BV-961") == src
    assert src.stat().st_mtime == mtime
    assert not (sdir / "BV-961_vame.h5").exists()
