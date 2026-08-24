"""Vérification de bout en bout de la page Nettoyage via AppTest.

Isolation : `lib.project.PREFS_PATH` monkeypatché (comme
`tests/test_app_pose.py`) pour ne jamais toucher `Path.home()` réel ni
`DEFAULT_PROJECTS_ROOT` (`D:\\EthoFlow\\projects`, un nom de dossier
littéral et relatif sur ce runner macOS).
"""
from __future__ import annotations

from pathlib import Path

import yaml
from streamlit.testing.v1 import AppTest

from lib import project as P

REPO = Path(__file__).resolve().parent.parent
APP_PY = str(REPO / "streamlit_app" / "app.py")


def _projet(tmp_path: Path, *, kind: str = "single",
           px_per_cm: float | None = None) -> Path:
    p = tmp_path / "projects" / "test-nettoyage"
    for sub in ("raw", "cropped", "dlc-output", "vame", "results"):
        (p / "data" / sub).mkdir(parents=True)
    (p / "configs").mkdir(parents=True)

    cfg: dict = {"kind": kind}
    if px_per_cm is not None:
        cfg["px_per_cm"] = px_per_cm

    (p / "configs" / "pipeline_config.yaml").write_text(
        yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8",
    )

    session_dir = p / "data" / "raw" / "S1"
    session_dir.mkdir(parents=True)
    (session_dir / "metadata.yaml").write_text(
        yaml.safe_dump({"source_video": str(session_dir / "S1.mp4")}), encoding="utf-8",
    )

    return p


def _lancer_sur_projet(tmp_path: Path, monkeypatch, projet: Path) -> AppTest:
    monkeypatch.setattr(P, "PREFS_PATH", tmp_path / "app_prefs.yaml")
    P.save_prefs({"projects_root": str(projet.parent), "models_root": str(tmp_path / "models")})
    (tmp_path / "models").mkdir(exist_ok=True)
    at = AppTest.from_file(APP_PY)
    at.session_state["current_project_path"] = str(projet)
    at.run()
    assert not at.exception, at.exception
    boutons = {b.key: b for b in at.button}
    assert "nav_nettoyage" in boutons, list(boutons)
    boutons["nav_nettoyage"].click().run()
    assert not at.exception, at.exception
    return at


def test_avertissement_px_per_cm_absent(tmp_path, monkeypatch):
    projet = _projet(tmp_path, kind="single", px_per_cm=None)
    at = _lancer_sur_projet(tmp_path, monkeypatch, projet)

    assert any("Passe 3" in w.value and "désactivée" in w.value for w in at.warning), \
        [w.value for w in at.warning]


def test_pas_avertissement_si_px_per_cm_configure(tmp_path, monkeypatch):
    projet = _projet(tmp_path, kind="single", px_per_cm=12.5)
    at = _lancer_sur_projet(tmp_path, monkeypatch, projet)

    assert not any("Passe 3" in w.value and "désactivée" in w.value for w in at.warning), \
        [w.value for w in at.warning]
    assert any("12.5 px/cm" in c.value for c in at.caption), [c.value for c in at.caption]


def test_section_arenes_visible_pour_projet_multi(tmp_path, monkeypatch):
    projet = _projet(tmp_path, kind="multi", px_per_cm=12.5)
    at = _lancer_sur_projet(tmp_path, monkeypatch, projet)

    assert any("assign_arenas" in c.value for c in at.caption), [c.value for c in at.caption]
    boutons = {b.key: b for b in at.button}
    assert "btn_nettoyage_arenes" in boutons


def test_section_arenes_remplacee_par_explication_pour_projet_single(tmp_path, monkeypatch):
    projet = _projet(tmp_path, kind="single", px_per_cm=12.5)
    at = _lancer_sur_projet(tmp_path, monkeypatch, projet)

    assert any("ne s'applique donc pas" in c.value for c in at.caption), \
        [c.value for c in at.caption]
    boutons = {b.key: b for b in at.button}
    assert "btn_nettoyage_arenes" not in boutons


def test_expander_outils_avances_replie_par_defaut(tmp_path, monkeypatch):
    projet = _projet(tmp_path, kind="single", px_per_cm=12.5)
    at = _lancer_sur_projet(tmp_path, monkeypatch, projet)

    expanders = [e for e in at.expander if "Outils avancés" in (e.label or "")]
    assert len(expanders) == 1
    assert expanders[0].proto.expanded is False
