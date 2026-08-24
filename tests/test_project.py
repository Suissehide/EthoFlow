from pathlib import Path

import yaml

from lib import project as P


def test_list_projects_ne_retient_que_les_vrais_projets(tmp_path):
    (tmp_path / "vrai" / "data").mkdir(parents=True)
    (tmp_path / "faux").mkdir()
    (tmp_path / "fichier.txt").write_text("x")
    assert [p.name for p in P.list_projects(tmp_path)] == ["vrai"]


def test_list_projects_racine_absente(tmp_path):
    assert P.list_projects(tmp_path / "nexiste-pas") == []


def test_list_dlc_models_veut_un_config_yaml(tmp_path):
    (tmp_path / "modele-a").mkdir()
    (tmp_path / "modele-a" / "config.yaml").write_text("x")
    (tmp_path / "pas-un-modele").mkdir()
    assert [p.name for p in P.list_dlc_models(tmp_path)] == ["modele-a"]


def test_lecture_pipeline_config(project):
    cfg_path = project / "configs" / "pipeline_config.yaml"
    cfg_path.write_text(yaml.safe_dump({
        "kind": "multi",
        "dlc_project_config": "/modeles/souris/config.yaml",
        "px_per_cm": 12.5,
        "default_arenes_coords": {"A1": [0, 0, 512, 540]},
    }))
    assert P.project_kind(project) == "multi"
    assert P.dlc_config_path(project) == "/modeles/souris/config.yaml"
    assert P.px_per_cm(project) == 12.5
    assert P.arena_coords(project) == {"A1": [0, 0, 512, 540]}


def test_projet_sans_config_ne_leve_pas(tmp_path):
    """Un projet fraîchement créé n'a pas encore de pipeline_config.yaml."""
    vide = tmp_path / "vide"
    (vide / "data").mkdir(parents=True)
    assert P.read_pipeline_config(vide) == {}
    assert P.dlc_config_path(vide) is None
    assert P.px_per_cm(vide) is None
    assert P.arena_coords(vide) == {}
    assert P.project_kind(vide) == "single"


def test_prefs_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr(P, "PREFS_PATH", tmp_path / "app_prefs.yaml")
    P.save_prefs({"projects_root": "/data/projets"})
    assert P.load_prefs()["projects_root"] == "/data/projets"
    assert P.projects_root() == Path("/data/projets")


def test_projects_root_defaut_vient_des_scripts(tmp_path, monkeypatch):
    """Sans préférence, on prend la racine que les scripts utilisent."""
    monkeypatch.setattr(P, "PREFS_PATH", tmp_path / "absent.yaml")
    import interactive
    assert P.projects_root() == interactive.DEFAULT_PROJECTS_ROOT
