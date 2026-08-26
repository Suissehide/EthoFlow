"""Régénération des `_labeled.mp4` à un seuil de confiance choisi.

Le Parcours A ne savait pas produire de vidéo annotée : `--mode custom`
n'appelle qu'`analyze_videos`. La capacité existait, mais enfermée dans
`prepare_dlc_feedback_kit.py`, dont l'objet est de fabriquer un zip pour
l'équipe VAME — un détour coûteux pour un simple contrôle visuel.

`relabel_video.py` expose le geste seul, et le kit réutilise ses
fonctions plutôt que d'en garder une copie.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest
import yaml

import relabel_video as RV


@pytest.fixture
def deeplabcut_factice(monkeypatch):
    """`deeplabcut` n'est pas installable dans l'env de test.

    Le double écrit le `_labeled.mp4` que DLC produirait, pour que la
    logique de renommage en `_pXX` soit réellement exercée.
    """
    appels = []

    def _create_labeled_video(config, videos, destfolder, pcutoff,
                               draw_skeleton=True, **kw):
        appels.append({"config": config, "videos": list(videos),
                       "destfolder": str(destfolder), "pcutoff": pcutoff})
        stem = Path(videos[0]).stem
        (Path(destfolder) / f"{stem}DLC_resnet50_labeled.mp4").write_bytes(b"mp4")

    faux = types.ModuleType("deeplabcut")
    faux.create_labeled_video = _create_labeled_video
    monkeypatch.setitem(sys.modules, "deeplabcut", faux)
    return appels


@pytest.fixture
def session(project, tmp_path):
    """Session avec sa vidéo source, sa metadata et un .h5 DLC."""
    sid = "BV-970"
    raw = project / "data" / "raw" / sid
    raw.mkdir(parents=True, exist_ok=True)
    video = tmp_path / f"{sid}.mp4"
    video.write_bytes(b"video")
    (raw / "metadata.yaml").write_text(
        yaml.safe_dump({"session_id": sid, "source_video": str(video)}))
    out = project / "data" / "dlc-output" / sid
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{sid}DLC_resnet50.h5").write_bytes(b"h5")
    return sid


def test_tag_du_seuil():
    """Le seuil est dans le nom, sinon deux passes s'écrasent."""
    assert RV.pcutoff_tag(0.3) == "p30"
    assert RV.pcutoff_tag(0.6) == "p60"
    assert RV.pcutoff_tag(0.05) == "p05"
    assert RV.pcutoff_tag(1.0) == "p100"


def test_deux_seuils_cohabitent(project, session, deeplabcut_factice):
    out = project / "data" / "dlc-output" / session
    a = RV.generate_labeled_video("/modele/config.yaml",
                                  Path(f"{session}.mp4"), out, 0.3)
    b = RV.generate_labeled_video("/modele/config.yaml",
                                  Path(f"{session}.mp4"), out, 0.6)
    assert a is not None and b is not None
    assert a != b
    assert a.name.endswith("_p30.mp4") and b.name.endswith("_p60.mp4")
    assert a.is_file() and b.is_file()
    # Le .h5 est relu, pas de ré-inférence : deux appels, deux pcutoff
    assert [c["pcutoff"] for c in deeplabcut_factice] == [0.3, 0.6]


def test_session_sans_h5_est_sautee(project, deeplabcut_factice, capsys):
    """Sans prédictions, il n'y a rien à redessiner — on le dit et on passe
    à la suivante au lieu de laisser DLC lever."""
    vide = project / "data" / "dlc-output" / "BV-999"
    vide.mkdir(parents=True)
    assert RV.generate_labeled_video("/modele/config.yaml",
                                     Path("BV-999.mp4"), vide, 0.6) is None
    assert deeplabcut_factice == []
    assert "h5" in capsys.readouterr().out.lower()


def test_projet_sans_modele_echoue_proprement(project):
    """`create_labeled_video` a besoin du config.yaml d'un projet DLC : en
    mode SuperAnimal il n'y en a pas. Mieux vaut le dire que planter dans
    les entrailles de DLC."""
    with pytest.raises(SystemExit):
        RV.resolve_model_config(project, no_prompt=True)


def test_video_source_lue_dans_la_metadata(project, session, tmp_path):
    assert RV.find_source_video(project, session) == tmp_path / f"{session}.mp4"


def test_video_source_absente(project, session, tmp_path):
    (tmp_path / f"{session}.mp4").unlink()
    assert RV.find_source_video(project, session) is None


def test_le_kit_utilise_la_fonction_partagee():
    """Une seule implémentation du nommage `pXX` et de la réutilisation du
    .h5 — sinon les deux divergent au premier changement."""
    import prepare_dlc_feedback_kit as KIT
    assert KIT.generate_labeled_video is RV.generate_labeled_video
    assert KIT.find_labeled_video is RV.find_labeled_video


def test_le_parser_se_construit(monkeypatch):
    """Régression : `add_project_dir_arg` ajoute déjà `--no-prompt`, le
    rajouter faisait lever argparse à la CONSTRUCTION du parser — donc à
    chaque lancement, y compris `--help`. Tester les fonctions pures ne
    l'attrape pas : il faut monter le parser."""
    monkeypatch.setattr(sys, "argv", ["relabel_video.py", "--help"])
    with pytest.raises(SystemExit) as exc:
        RV.main()
    assert exc.value.code == 0        # --help sort en 0, pas sur une erreur


def test_modele_non_entraine_diagnostique_avant_dlc(project, tmp_path, capsys):
    """`create_labeled_video` dérive le nom du scorer des métadonnées
    d'entraînement et sort « Could not find a shuffle... » sur un modèle
    déplacé ou non entraîné. `run_dlc_inference` sait déjà diagnostiquer
    ce cas : ce script doit passer par les mêmes garde-fous plutôt que de
    laisser remonter l'erreur cryptique de DLC."""
    modele = tmp_path / "modele"
    modele.mkdir()
    (modele / "config.yaml").write_text(yaml.safe_dump({
        "project_path": str(modele), "TrainingFraction": [0.95],
        "iteration": 0,
    }))
    (project / "configs" / "pipeline_config.yaml").write_text(
        yaml.safe_dump({"dlc_project_config": str(modele / "config.yaml")}))

    with pytest.raises(SystemExit):
        RV.resolve_model_config(project, no_prompt=True)
    sortie = capsys.readouterr()
    assert "entraîné" in (sortie.out + sortie.err)
