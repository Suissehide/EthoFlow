"""Page Qualité DLC : lib/dlc_qualite.py et rendu, sur un modèle DLC factice."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from streamlit.testing.v1 import AppTest

from lib import project as P

REPO = Path(__file__).resolve().parent.parent
APP_PY = str(REPO / "streamlit_app" / "app.py")
KEYPOINTS = ["nose", "tail_base", "paw_left", "paw_right"]


def _df_dlc(xy: np.ndarray, likelihood: np.ndarray | None, index) -> pd.DataFrame:
    """Format DLC : colonnes (scorer, bodyparts, coords)."""
    coords = ["x", "y"] + (["likelihood"] if likelihood is not None else [])
    cols = pd.MultiIndex.from_product([["DLC"], KEYPOINTS, coords],
                                      names=["scorer", "bodyparts", "coords"])
    blocs = []
    for k in range(len(KEYPOINTS)):
        blocs += [xy[:, k, 0], xy[:, k, 1]]
        if likelihood is not None:
            blocs.append(likelihood[:, k])
    return pd.DataFrame(np.column_stack(blocs), index=index, columns=cols)


@pytest.fixture
def modele(project: Path, tmp_path: Path) -> Path:
    rng = np.random.default_rng(0)
    m = tmp_path / "models" / "souris"
    (m / "labeled-data" / "vid").mkdir(parents=True)
    (m / "config.yaml").write_text("Task: souris\n")
    P.set_dlc_config(project, m)

    train = m / "dlc-models-pytorch" / "iteration-0" / "sourisshuffle1" / "train"
    train.mkdir(parents=True)
    (train / "pytorch_config.yaml").write_text(yaml.safe_dump({"train_settings": {"epochs": 50}}))
    (train / "learning_stats.csv").write_text(
        "step,losses/train.total_loss,losses/eval.total_loss,metrics/test.rmse\n"
        + "".join(f"{e},{1/e:.4f},{1.2/e:.4f},{10/e:.3f}\n" for e in range(1, 11)))

    index = [f"labeled-data/vid/img{i:03d}.png" for i in range(20)]
    gt = rng.uniform(100, 400, (20, len(KEYPOINTS), 2))
    _df_dlc(gt, None, index).to_hdf(m / "labeled-data" / "vid" / "CollectedData_x.h5", key="df")
    pred = gt + rng.normal(0, 2, gt.shape)
    pred[:5, 2] = gt[:5, 3]                       # patte gauche posée sur la droite
    lik = np.full((20, len(KEYPOINTS)), 0.95)
    ev = m / "evaluation-results-pytorch" / "iteration-0"
    ev.mkdir(parents=True)
    _df_dlc(pred, lik, index).to_hdf(ev / "souris-snapshot-best.h5", key="df")

    sortie = project / "data" / "dlc-output" / "S1"
    sortie.mkdir(parents=True)
    lik_video = np.column_stack([np.full(50, 0.9), np.full(50, 0.8),
                                 np.full(50, 0.05), np.full(50, 0.6)])
    _df_dlc(rng.uniform(0, 500, (50, len(KEYPOINTS), 2)), lik_video,
            range(50)).to_hdf(sortie / "S1DLC_souris.h5", key="df")
    (sortie / "S1DLC_souris_filtered.h5").write_bytes(b"")  # jamais choisi
    return m


def test_lib_trois_diagnostics(project, modele):
    import lib.dlc_qualite as Q

    assert Q.modele_dir(project) == modele
    [stats] = Q.stats_entrainement(modele)
    assert stats["epochs_prevues"] == 50 and len(stats["df"]) == 10

    tab, confusions, _ = Q.erreurs_keypoints(modele)
    assert set(tab["keypoint"]) == set(KEYPOINTS) | {"TOTAL"}
    gauche = next(p for p in confusions["paires"] if p["keypoint"] == "paw_left")
    assert gauche["pct_sur_jumeau"] == pytest.approx(25.0)

    assert list(Q.h5_sessions(project)) == ["S1"]
    resume, _, verdict = Q.confiances(Q.h5_sessions(project)["S1"])
    assert resume.index[0] == "nose" and resume.index[-1] == "paw_left"
    assert "paw_left" in verdict


def test_sans_evaluation_le_diagnostic_du_script_est_montre(project, modele):
    import shutil
    import lib.dlc_qualite as Q
    shutil.rmtree(modele / "evaluation-results-pytorch")
    with pytest.raises(FileNotFoundError, match="jamais tourné"):
        Q.erreurs_keypoints(modele)


def _page(tmp_path, monkeypatch, projet) -> AppTest:
    monkeypatch.setattr(P, "PREFS_PATH", tmp_path / "app_prefs.yaml")
    P.save_prefs({"projects_root": str(projet.parent), "models_root": str(tmp_path / "models")})
    at = AppTest.from_file(APP_PY, default_timeout=10)
    at.session_state["current_project_path"] = str(projet)
    at.run()
    at.button(key="nav_qualite").click().run()
    assert not at.exception, at.exception
    return at


def test_page_rendue(tmp_path, monkeypatch, project, modele):
    at = _page(tmp_path, monkeypatch, project)
    metriques = {m.label: m.value for m in at.metric}
    assert metriques["Epoch"] == "10 / 50"
    assert "RMSE (confiantes)" in metriques
    assert len(at.get("vega_lite_chart")) >= 4   # pertes, rmse, erreurs, confiances
    assert any("paw_left" in c.value for c in at.code)


def test_page_sans_modele(tmp_path, monkeypatch, project):
    at = _page(tmp_path, monkeypatch, project)
    assert any("Aucun modèle DLC" in i.value for i in at.info)
