"""`--config-dir` du Parcours B : demandé comme partout ailleurs.

Les scripts d'entraînement DLC étaient les seuls à ne rien demander quand
leur dossier de config manquait — ils retombaient en silence sur le
`_config.py` template du repo, donc sur la vidéo pilote et le nom de
projet de quelqu'un d'autre. Ils passent maintenant par le même menu
numéroté que `--project-dir` et `--dlc-config`.
"""
from __future__ import annotations

import argparse

import pytest

import interactive as I


def _modele(root, nom, *, avec_config=True):
    d = root / nom
    d.mkdir(parents=True)
    if avec_config:
        (d / "_config.py").write_text("PROJECT_NAME = 'x'\n")
    return d


def test_list_dlc_config_dirs_veut_un_config_py(tmp_path):
    _modele(tmp_path, "souris-bottomview")
    _modele(tmp_path, "pas-encore-init", avec_config=False)
    (tmp_path / "fichier.txt").write_text("x")
    assert [d.name for d in I.list_dlc_config_dirs(tmp_path)] == [
        "souris-bottomview"]


def test_list_dlc_config_dirs_racine_absente(tmp_path):
    assert I.list_dlc_config_dirs(tmp_path / "nexiste-pas") == []


def test_choix_par_numero(tmp_path, monkeypatch):
    a = _modele(tmp_path, "a-modele")
    _modele(tmp_path, "b-modele")
    monkeypatch.setattr("builtins.input", lambda _: "1")
    assert I.prompt_dlc_config_dir(tmp_path) == a


def test_choix_par_nom(tmp_path, monkeypatch):
    _modele(tmp_path, "a-modele")
    b = _modele(tmp_path, "b-modele")
    monkeypatch.setattr("builtins.input", lambda _: "b-modele")
    assert I.prompt_dlc_config_dir(tmp_path) == b


def test_no_prompt_echoue_au_lieu_de_deviner(tmp_path):
    """Le fallback silencieux sur le template du repo est précisément ce
    qu'on supprime : en non-interactif on échoue, on ne devine pas."""
    _modele(tmp_path, "a-modele")
    with pytest.raises(SystemExit):
        I.prompt_dlc_config_dir(tmp_path, no_prompt=True)


def test_load_config_demande_quand_le_flag_manque(tmp_path, monkeypatch):
    """Le helper des scripts numérotés renvoie le dossier résolu ET le
    réinjecte dans args : `01_setup_project` en a besoin pour savoir où
    merger le projet DLC."""
    import _load_config as LC

    modele = _modele(tmp_path, "souris-bottomview")
    monkeypatch.setattr(LC, "DEFAULT_MODELS_ROOT", tmp_path)
    monkeypatch.setattr("builtins.input", lambda _: "1")

    args = argparse.Namespace(config_dir=None, no_prompt=False)
    assert LC.load_config(args) == modele
    assert args.config_dir == modele


def test_load_config_respecte_un_flag_explicite(tmp_path, monkeypatch):
    modele = _modele(tmp_path, "souris-bottomview")
    import _load_config as LC

    def _boom(_):
        raise AssertionError("ne doit pas demander")
    monkeypatch.setattr("builtins.input", _boom)

    args = argparse.Namespace(config_dir=modele, no_prompt=False)
    assert LC.load_config(args) == modele
