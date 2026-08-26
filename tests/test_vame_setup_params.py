"""Hyperparamètres VAME demandés au `setup`, pas subis.

`run_vame.py setup` créait le projet avec les défauts VAME sans jamais
mentionner qu'ils existaient : l'utilisateur découvrait `n_clusters=15` en
lisant le README, une fois l'entraînement lancé. Ils sont maintenant
demandés à l'invite (avec explication et défaut), surchargeables par flag,
et écrits dans le config.yaml juste après l'init.
"""
from __future__ import annotations

import argparse

import yaml

import run_vame as RV


def _args(**kw):
    base = dict(n_clusters=None, time_window=None, max_epochs=None,
                pose_confidence=None, no_prompt=True)
    base.update(kw)
    return argparse.Namespace(**base)


def test_no_prompt_prend_les_defauts():
    assert RV.resolve_setup_params(_args()) == {
        "n_clusters": 15, "time_window": 30, "max_epochs": 500,
        "pose_confidence": 0.6,
    }


def test_les_flags_priment_sur_les_defauts():
    params = RV.resolve_setup_params(_args(n_clusters=25, max_epochs=100))
    assert params["n_clusters"] == 25
    assert params["max_epochs"] == 100
    assert params["time_window"] == 30


def test_flag_seul_ne_declenche_pas_l_invite(monkeypatch):
    """Tous les paramètres donnés en flags : rien à demander, donc aucune
    invite — sinon un lancement scripté se bloquerait sur `input()`."""
    def _boom(*a, **k):
        raise AssertionError("ne doit pas demander")
    monkeypatch.setattr(RV, "prompt_int", _boom)
    monkeypatch.setattr(RV, "prompt_float", _boom)
    params = RV.resolve_setup_params(_args(
        n_clusters=8, time_window=60, max_epochs=250, pose_confidence=0.9,
        no_prompt=False))
    assert params == {"n_clusters": 8, "time_window": 60,
                      "max_epochs": 250, "pose_confidence": 0.9}


def test_ecriture_des_parametres_preserve_le_reste(project, vame_project):
    cfg_path = vame_project / "config.yaml"
    cfg = yaml.safe_load(cfg_path.read_text())
    cfg["project_name"] = "vame"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))

    RV.set_vame_params(project, n_clusters=25, max_epochs=100)

    apres = yaml.safe_load(cfg_path.read_text())
    assert apres["n_clusters"] == 25
    assert apres["max_epochs"] == 100
    assert apres["project_name"] == "vame"
    assert apres["segmentation_algorithms"] == ["hmm"]


def test_set_n_clusters_reste_disponible(project, vame_project):
    """`segment --n-clusters` passe toujours par là."""
    RV.set_n_clusters(project, 30)
    assert yaml.safe_load((vame_project / "config.yaml").read_text())[
        "n_clusters"] == 30


# ============================================================
# Dossier data/vame/ vide laissé par create_project.py
# ============================================================

def test_dossier_vame_vide_est_retire_avant_init(project):
    """Régression : `run_vame.py setup` plantait sur « Config file is not
    found » à chaque premier lancement.

    `create_project.py` crée le squelette du projet, `data/vame/` compris —
    vide. `vame.init_new_project` fait alors :

        if project_path.exists():
            return projconfigfile, read_config(projconfigfile)

    Il considère le dossier vide comme un projet déjà initialisé, court-
    circuite la création, puis lève FileNotFoundError en lisant un
    `config.yaml` qui n'a jamais été écrit. Le garde-fou d'EthoFlow ne
    refusait que les dossiers NON vides : le cas vide passait droit dans le
    piège. On retire donc le dossier vide, VAME le recrée."""
    vame = project / "data" / "vame"
    vame.mkdir(parents=True, exist_ok=True)
    assert vame.is_dir() and not any(vame.iterdir())

    RV.ensure_project_dir_libre(vame, force=False)

    assert not vame.exists()


def test_dossier_vame_absent_ne_leve_pas(project):
    absent = project / "data" / "vame"
    if absent.exists():
        absent.rmdir()
    RV.ensure_project_dir_libre(absent, force=False)
    assert not absent.exists()


def test_projet_vame_existant_refuse_sans_force(project, vame_project):
    """Un vrai projet (config.yaml présent) reste protégé."""
    import pytest
    with pytest.raises(SystemExit):
        RV.ensure_project_dir_libre(vame_project, force=False)
    assert (vame_project / "config.yaml").is_file()


def test_force_supprime_le_projet_existant(project, vame_project):
    RV.ensure_project_dir_libre(vame_project, force=True)
    assert not vame_project.exists()
