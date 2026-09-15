"""Extraction manuelle de frames : toutes les vidéos, pas que la pilote.

`deeplabcut.extract_frames(config, mode="manual")` n'ouvre la GUI que sur
`videos[0]` — la vidéo pilote — et rend la main. Les vidéos ajoutées par
`04_add_videos.py` ne sont jamais proposées, alors que ce sont justement
elles qui portent la diversité inter-individus qu'on cherche à labelliser.

Le script `extract_frames_manual.py` lit la liste qui fait foi
(`video_sets` du `config.yaml` DLC) et lance la GUI une vidéo à la fois.
Les tests ci-dessous couvrent les deux parties pures : lire cette liste,
et la filtrer depuis une saisie utilisateur. La boucle napari elle-même
n'est qu'une fine couche par-dessus.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

import extract_frames_manual as E


def _config_yaml(tmp_path: Path, videos, **extra) -> Path:
    """Fabrique un config.yaml DLC minimal avec le video_sets demandé."""
    cfg = {"Task": "souris", "project_path": str(tmp_path)}
    cfg["video_sets"] = (
        videos if not isinstance(videos, list)
        else {v: {"crop": "0, 1024, 0, 1080"} for v in videos}
    )
    cfg.update(extra)
    p = tmp_path / "config.yaml"
    p.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return p


# ---------------------------------------------------------------- video_sets

def test_toutes_les_videos_du_projet_sont_lues_pas_seulement_la_premiere(tmp_path):
    """Le bug d'origine : DLC s'arrête à videos[0]."""
    cfg = _config_yaml(tmp_path, [
        "/data/pilote.mp4", "/data/souris02.mp4", "/data/souris03.mp4"])
    assert E.read_video_sets(cfg) == [
        Path("/data/pilote.mp4"),
        Path("/data/souris02.mp4"),
        Path("/data/souris03.mp4"),
    ]


def test_l_ordre_du_config_yaml_est_preserve(tmp_path):
    """La pilote reste en tête : c'est l'ordre que l'utilisateur connaît."""
    cfg = _config_yaml(tmp_path, ["/d/z.mp4", "/d/a.mp4", "/d/m.mp4"])
    assert [v.name for v in E.read_video_sets(cfg)] == ["z.mp4", "a.mp4", "m.mp4"]


@pytest.mark.parametrize("video_sets", [None, {}])
def test_projet_sans_video_sets(tmp_path, video_sets):
    cfg = _config_yaml(tmp_path, video_sets)
    assert E.read_video_sets(cfg) == []


def test_les_chemins_windows_ne_sont_pas_reecrits(tmp_path):
    """DLC écrit des chemins absolus Windows dans video_sets. Un `.resolve()`
    bien intentionné les préfixerait du cwd côté POSIX et casserait les
    chemins UNC côté Windows : on rend la clé telle quelle."""
    cfg = _config_yaml(tmp_path, [r"D:\EthoFlow\data\souris02.mp4"])
    assert str(E.read_video_sets(cfg)[0]) == r"D:\EthoFlow\data\souris02.mp4"


# ----------------------------------------------------------------- sélection

VIDEOS = [Path("/d/pilote.mp4"), Path("/d/souris02.mp4"), Path("/d/souris03.avi")]


@pytest.mark.parametrize("reponse", ["", "  ", "all", "ALL", "tout"])
def test_une_reponse_vide_ou_all_prend_tout(reponse):
    assert E.select_videos(VIDEOS, reponse) == VIDEOS


def test_selection_par_numero():
    assert E.select_videos(VIDEOS, "1,3") == [VIDEOS[0], VIDEOS[2]]


@pytest.mark.parametrize("separateur", [",", " ", ", ", ";"])
def test_les_separateurs_courants_marchent(separateur):
    assert E.select_videos(VIDEOS, f"1{separateur}2") == VIDEOS[:2]


@pytest.mark.parametrize("nom", ["souris02", "souris02.mp4", "SOURIS02"])
def test_selection_par_nom(nom):
    """Avec ou sans extension, insensible à la casse : on tape ce qu'on lit."""
    assert E.select_videos(VIDEOS, nom) == [VIDEOS[1]]


def test_numeros_et_noms_peuvent_se_melanger():
    assert E.select_videos(VIDEOS, "1, souris03") == [VIDEOS[0], VIDEOS[2]]


def test_les_doublons_sont_ecrases_et_l_ordre_du_projet_conserve():
    assert E.select_videos(VIDEOS, "3,1,pilote,3") == [VIDEOS[0], VIDEOS[2]]


@pytest.mark.parametrize("reponse", ["0", "4", "souris99", "-1", "1..2"])
def test_une_selection_invalide_est_refusee(reponse):
    """Mieux vaut redemander que d'ouvrir napari sur la mauvaise vidéo."""
    with pytest.raises(ValueError):
        E.select_videos(VIDEOS, reponse)


# ------------------------------------------------------- frames déjà extraites

def test_compte_les_frames_deja_extraites(tmp_path):
    labeled = tmp_path / "labeled-data" / "souris02"
    labeled.mkdir(parents=True)
    for i in range(3):
        (labeled / f"img{i:03d}.png").write_bytes(b"\x89PNG")
    (labeled / "CollectedData_labo.h5").write_bytes(b"\x00")
    assert E.count_extracted_frames(tmp_path, Path("/d/souris02.mp4")) == 3


def test_compte_zero_quand_la_video_n_a_jamais_ete_extraite(tmp_path):
    assert E.count_extracted_frames(tmp_path, Path("/d/jamais_vue.mp4")) == 0


# ------------------------------------------------------------- env manquant

def test_env_dlc_inactif_donne_un_message_pas_un_traceback(monkeypatch, capsys):
    """Oublier `conda activate dlc` est l'erreur la plus fréquente du
    Parcours B : elle doit dire quoi faire, pas dérouler une stack."""
    monkeypatch.setitem(__import__("sys").modules, "deeplabcut", None)
    with pytest.raises(SystemExit):
        E.require_deeplabcut()
    assert "conda activate dlc" in capsys.readouterr().err
