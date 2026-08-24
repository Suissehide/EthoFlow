"""Vérification de bout en bout de la page Pose (DLC) via AppTest.

Isolation : `lib.project.PREFS_PATH` monkeypatché (comme
`tests/test_app_donnees.py`) pour ne jamais toucher `Path.home()` réel ni
`DEFAULT_PROJECTS_ROOT` (`D:\\EthoFlow\\projects`, un nom de dossier
littéral et relatif sur ce runner macOS — piège déjà tombé plusieurs fois).

Le projet courant est injecté directement dans `at.session_state`
(`current_project_path`), comme le fait `lib.config.set_current_project`.
"""
from __future__ import annotations

from pathlib import Path

import yaml
from streamlit.testing.v1 import AppTest

from lib import project as P

REPO = Path(__file__).resolve().parent.parent
APP_PY = str(REPO / "streamlit_app" / "app.py")


def _projet(tmp_path: Path, *, kind: str = "single",
           modele: str = "absent", avec_crop: bool = False) -> Path:
    """`modele` : "absent" (pas de `dlc_project_config`), "introuvable"
    (une valeur configurée mais aucun fichier à ce chemin — modèle déplacé
    ou supprimé, ruling R12.1), ou "ok" (un vrai fichier existe)."""
    p = tmp_path / "projects" / "test-pose"
    for sub in ("raw", "cropped", "dlc-output", "vame", "results"):
        (p / "data" / sub).mkdir(parents=True)
    (p / "configs").mkdir(parents=True)

    cfg: dict = {"kind": kind}
    if modele == "ok":
        chemin_modele = tmp_path / "modeles" / "souris" / "config.yaml"
        chemin_modele.parent.mkdir(parents=True, exist_ok=True)
        chemin_modele.write_text("dummy: true", encoding="utf-8")
        cfg["dlc_project_config"] = str(chemin_modele)
    elif modele == "introuvable":
        cfg["dlc_project_config"] = str(tmp_path / "modeles" / "disparu" / "config.yaml")
    elif modele != "absent":
        raise ValueError(modele)

    (p / "configs" / "pipeline_config.yaml").write_text(
        yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8",
    )

    # Une session, pour que le sélecteur ait quelque chose à lister.
    session_dir = p / "data" / "raw" / "S1"
    session_dir.mkdir(parents=True)
    (session_dir / "metadata.yaml").write_text(
        yaml.safe_dump({"source_video": str(session_dir / "S1.mp4")}), encoding="utf-8",
    )

    if avec_crop:
        cropped_session = p / "data" / "cropped" / "S1"
        cropped_session.mkdir(parents=True)
        (cropped_session / "S1_A1.mp4").write_bytes(b"\x00")

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
    assert "nav_pose" in boutons, list(boutons)
    boutons["nav_pose"].click().run()
    assert not at.exception, at.exception
    return at


def _radio_mode(at: AppTest):
    return [r for r in at.radio if r.key == "pose_mode"][0]


def test_mode_par_defaut_custom_si_modele_configure(tmp_path, monkeypatch):
    projet = _projet(tmp_path, kind="single", modele="ok")
    at = _lancer_sur_projet(tmp_path, monkeypatch, projet)

    assert _radio_mode(at).value == "custom"
    # Modèle configuré et présent sur disque : pas d'avertissement.
    assert not any("Aucun modèle DLC" in w.value for w in at.warning)
    assert not any("introuvable" in w.value for w in at.warning)


def test_mode_par_defaut_superanimal_si_multi_sans_modele(tmp_path, monkeypatch):
    projet = _projet(tmp_path, kind="multi", modele="absent", avec_crop=False)
    at = _lancer_sur_projet(tmp_path, monkeypatch, projet)

    assert _radio_mode(at).value == "superanimal"


def test_mode_par_defaut_single_animal_si_videos_croppees(tmp_path, monkeypatch):
    projet = _projet(tmp_path, kind="multi", modele="absent", avec_crop=True)
    at = _lancer_sur_projet(tmp_path, monkeypatch, projet)

    assert _radio_mode(at).value == "single-animal"


def test_modele_introuvable_ne_defaut_pas_sur_custom(tmp_path, monkeypatch):
    """Régression R12.1 : un `dlc_project_config` qui pointe vers un fichier
    disparu ne doit pas se comporter comme un modèle configuré — sinon la
    page propose `custom` par défaut alors que le job échouerait à coup sûr
    (--no-prompt). Sans crop, on retombe sur `superanimal`."""
    projet = _projet(tmp_path, kind="single", modele="introuvable")
    at = _lancer_sur_projet(tmp_path, monkeypatch, projet)

    assert _radio_mode(at).value == "superanimal"


def test_custom_sans_modele_avertit_et_desactive_le_bouton(tmp_path, monkeypatch):
    projet = _projet(tmp_path, kind="single", modele="absent")
    at = _lancer_sur_projet(tmp_path, monkeypatch, projet)

    # Défaut single sans modèle -> superanimal ; on force custom.
    _radio_mode(at).set_value("custom").run()
    assert not at.exception, at.exception

    assert any("Aucun modèle DLC" in w.value for w in at.warning), [w.value for w in at.warning]
    assert any("Modèle DLC" in w.value and "Projet" in w.value for w in at.warning)

    boutons = {b.key: b for b in at.button}
    assert boutons["btn_pose_lancer"].disabled


def test_custom_avec_modele_introuvable_pointe_vers_diagnostiquer(tmp_path, monkeypatch):
    """Message distinct de « absent » (ruling R12.1) : le modèle a été
    déplacé/supprimé, on pointe vers Projet -> Modèle DLC -> Diagnostiquer,
    pas vers « désigne un modèle » comme si rien n'avait jamais été fait."""
    projet = _projet(tmp_path, kind="single", modele="introuvable")
    at = _lancer_sur_projet(tmp_path, monkeypatch, projet)

    _radio_mode(at).set_value("custom").run()
    assert not at.exception, at.exception

    assert any("introuvable" in w.value and "Diagnostiquer" in w.value for w in at.warning), \
        [w.value for w in at.warning]
    assert not any("Aucun modèle DLC" in w.value for w in at.warning)

    boutons = {b.key: b for b in at.button}
    assert boutons["btn_pose_lancer"].disabled


def test_batch_size_visible_seulement_si_video_adapt(tmp_path, monkeypatch):
    projet = _projet(tmp_path, kind="single", modele="ok")
    at = _lancer_sur_projet(tmp_path, monkeypatch, projet)

    assert [n for n in at.number_input if n.key == "pose_video_adapt_batch_size"] == []

    case = [c for c in at.checkbox if c.key == "pose_video_adapt"][0]
    case.set_value(True).run()
    assert not at.exception, at.exception

    champs = [n for n in at.number_input if n.key == "pose_video_adapt_batch_size"]
    assert len(champs) == 1
    assert champs[0].value == 2


def test_toutes_sessions_coche_par_defaut_active_le_bouton(tmp_path, monkeypatch):
    projet = _projet(tmp_path, kind="single", modele="ok")
    at = _lancer_sur_projet(tmp_path, monkeypatch, projet)

    boutons = {b.key: b for b in at.button}
    assert not boutons["btn_pose_lancer"].disabled

    case_tout = [c for c in at.checkbox if c.key == "pose_all"][0]
    case_tout.set_value(False).run()
    assert not at.exception, at.exception

    boutons = {b.key: b for b in at.button}
    assert boutons["btn_pose_lancer"].disabled
