"""Vérification de bout en bout de la page Vidéos & calibration via AppTest.

Isolation : `lib.project.PREFS_PATH` monkeypatché (comme `test_app_pose.py`)
pour ne jamais toucher `Path.home()` réel ni `DEFAULT_PROJECTS_ROOT`
(`D:\\EthoFlow\\projects`, un nom de dossier littéral et relatif sur ce
runner macOS).

Les tests de sonde/comparaison utilisent une **vraie** petite vidéo générée
avec OpenCV (même fixture que `tests/test_video.py`) : un fichier factice
d'un octet ne prouverait rien sur l'affichage des caractéristiques réelles.
"""
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


@pytest.fixture
def video_reelle(tmp_path):
    cv2 = pytest.importorskip("cv2")

    def _make(dossier: Path, nom: str = "clip.mp4", *,
              fps: float = 10.0, n: int = 20, w: int = 64, h: int = 48) -> Path:
        dossier = Path(dossier)
        dossier.mkdir(parents=True, exist_ok=True)
        chemin = dossier / nom
        writer = cv2.VideoWriter(
            str(chemin), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        for i in range(n):
            frame = np.full((h, w, 3), (i * 10) % 255, dtype=np.uint8)
            writer.write(frame)
        writer.release()
        if not chemin.exists() or chemin.stat().st_size == 0:
            pytest.skip("encodeur mp4v indisponible")
        return chemin

    return _make


def _projet(tmp_path: Path, *, kind: str = "single") -> Path:
    p = tmp_path / "projects" / "test-videos"
    for sub in ("raw", "cropped", "dlc-output", "vame", "results"):
        (p / "data" / sub).mkdir(parents=True)
    (p / "configs").mkdir(parents=True)
    (p / "configs" / "pipeline_config.yaml").write_text(
        yaml.safe_dump({"kind": kind}, sort_keys=False), encoding="utf-8",
    )
    return p


def _ecrire_session(projet: Path, session_id: str, *, source_video,
                    camera: dict | None = None, extra: dict | None = None) -> Path:
    sdir = projet / "data" / "raw" / session_id
    sdir.mkdir(parents=True, exist_ok=True)
    meta: dict = {"id": session_id, "source_video": str(source_video)}
    if camera is not None:
        meta["camera"] = camera
    if extra:
        meta.update(extra)
    (sdir / "metadata.yaml").write_text(
        yaml.safe_dump(meta, sort_keys=False, allow_unicode=True), encoding="utf-8",
    )
    return sdir


def _lancer_sur_projet(tmp_path: Path, monkeypatch, projet: Path) -> AppTest:
    monkeypatch.setattr(P, "PREFS_PATH", tmp_path / "app_prefs.yaml")
    P.save_prefs({"projects_root": str(projet.parent), "models_root": str(tmp_path / "models")})
    (tmp_path / "models").mkdir(exist_ok=True)
    at = AppTest.from_file(APP_PY)
    at.session_state["current_project_path"] = str(projet)
    at.run()
    assert not at.exception, at.exception
    boutons = {b.key: b for b in at.button}
    assert "nav_videos" in boutons, list(boutons)
    boutons["nav_videos"].click().run()
    assert not at.exception, at.exception
    return at


def _compter_elements(at: AppTest, type_recherche: str) -> int:
    """`st.image`/`st.video` n'ont pas de propriété dédiée dans AppTest (pas
    des widgets interactifs) : ils apparaissent comme `UnknownElement` dans
    l'arbre, avec `.type` == "imgs" ou "video" — on les compte à la main."""
    compte = 0

    def _walk(node) -> None:
        nonlocal compte
        if getattr(node, "type", None) == type_recherche:
            compte += 1
        for enfant in getattr(node, "children", {}).values():
            _walk(enfant)

    _walk(at.main)
    return compte


def _tableau_comparaison(at: AppTest) -> pd.DataFrame:
    for d in at.dataframe:
        cols = list(d.value.columns)
        if cols == ["Caractéristique", "Déclaré (metadata)", "Réel (fichier)"]:
            return d.value
    raise AssertionError("tableau de comparaison introuvable")


# ============================================================
# Caractéristiques réelles + comparaison
# ============================================================

def test_caracteristiques_reelles_affichees(tmp_path, monkeypatch, video_reelle):
    projet = _projet(tmp_path, kind="single")
    video = video_reelle(tmp_path / "videos", fps=10.0, n=20, w=64, h=48)
    _ecrire_session(projet, "S1", source_video=video,
                    camera={"fps": 10, "width": 64, "height": 48})

    at = _lancer_sur_projet(tmp_path, monkeypatch, projet)

    table = _tableau_comparaison(at)
    lignes = table.set_index("Caractéristique")
    assert lignes.loc["Largeur (px)", "Réel (fichier)"] == "64"
    assert lignes.loc["Hauteur (px)", "Réel (fichier)"] == "48"
    assert lignes.loc["Frames", "Réel (fichier)"] == "20"
    assert abs(float(lignes.loc["fps", "Réel (fichier)"]) - 10.0) < 0.5

    # Vignette + lecteur présents, aucune alerte de décalage fps (déclaré == réel).
    assert _compter_elements(at, "imgs") >= 1
    assert _compter_elements(at, "video") >= 1
    assert not any("Écart de fps" in e.value for e in at.error)


def test_ecart_fps_signale(tmp_path, monkeypatch, video_reelle):
    """Metadata déclare 25 fps, le fichier réel tourne à 10 fps — écart > tolérance."""
    projet = _projet(tmp_path, kind="single")
    video = video_reelle(tmp_path / "videos", fps=10.0, n=20, w=64, h=48)
    _ecrire_session(projet, "S1", source_video=video,
                    camera={"fps": 25, "width": 64, "height": 48})

    at = _lancer_sur_projet(tmp_path, monkeypatch, projet)

    assert any("Écart de fps" in e.value for e in at.error), [e.value for e in at.error]
    table = _tableau_comparaison(at)
    lignes = table.set_index("Caractéristique")
    assert lignes.loc["fps", "Déclaré (metadata)"] == "25"
    assert abs(float(lignes.loc["fps", "Réel (fichier)"]) - 10.0) < 0.5


# ============================================================
# Vidéo manquante -> re-pointage, jamais d'exception
# ============================================================

def test_session_video_manquante_affiche_repointage_sans_exception(tmp_path, monkeypatch):
    projet = _projet(tmp_path, kind="single")
    _ecrire_session(projet, "S-perdue",
                    source_video=tmp_path / "chemin" / "qui" / "nexiste-pas.mp4")

    at = _lancer_sur_projet(tmp_path, monkeypatch, projet)

    assert not at.exception, at.exception
    assert any("introuvable" in w.value.lower() for w in at.warning), \
        [w.value for w in at.warning]
    # La section de re-pointage est bien affichée, avec son champ de saisie.
    champs = {t.key: t for t in at.text_input}
    assert "videos_relink_dossier" in champs


# ============================================================
# Onglet Crop : présent en multi, remplacé par une explication en single
# ============================================================

def test_onglet_crop_multi_explique_les_deux_voies(tmp_path, monkeypatch):
    projet = _projet(tmp_path, kind="multi")
    _ecrire_session(projet, "S1", source_video=tmp_path / "videos" / "S1.mp4")

    at = _lancer_sur_projet(tmp_path, monkeypatch, projet)

    textes = [m.value for m in at.markdown] + [c.value for c in at.caption]
    assert any("Voie A" in t for t in textes)
    assert any("Voie B" in t for t in textes)
    boutons = {b.key: b for b in at.button}
    assert "btn_videos_crop" in boutons


def test_onglet_crop_single_explique_pourquoi_absent(tmp_path, monkeypatch):
    projet = _projet(tmp_path, kind="single")
    _ecrire_session(projet, "S1", source_video=tmp_path / "videos" / "S1.mp4")

    at = _lancer_sur_projet(tmp_path, monkeypatch, projet)

    textes = [c.value for c in at.caption]
    assert any("n'a de sens que pour isoler" in t for t in textes), textes
    boutons = {b.key: b for b in at.button}
    assert "btn_videos_crop" not in boutons


# ============================================================
# Re-pointage de bout en bout : écriture réelle sur disque
# ============================================================

def test_repointage_bout_en_bout_ecrit_metadata_et_preserve_le_reste(
    tmp_path, monkeypatch, video_reelle,
):
    projet = _projet(tmp_path, kind="single")
    chemin_mort = tmp_path / "D" / "ancien" / "chemin" / "BV-970.mp4"
    _ecrire_session(
        projet, "BV-970", source_video=chemin_mort,
        camera={"fps": 25, "width": 640, "height": 480},
        extra={"date": "2025-10-10", "notes": "essai pilote"},
    )
    meta_avant = yaml.safe_load(
        (projet / "data" / "raw" / "BV-970" / "metadata.yaml").read_text())

    nouveau_dossier = tmp_path / "videos-retrouvees"
    video_reelle(nouveau_dossier, nom="BV-970.mp4", fps=10.0, n=20, w=64, h=48)

    at = _lancer_sur_projet(tmp_path, monkeypatch, projet)

    # Le champ "dossier des vidéos" pointe vers l'endroit où le fichier a été replacé.
    champs = {t.key: t for t in at.text_input}
    champs["videos_relink_dossier"].set_value(str(nouveau_dossier)).run()
    assert not at.exception, at.exception

    boutons = {b.key: b for b in at.button}
    assert "btn_relink_demander" in boutons, list(boutons)
    boutons["btn_relink_demander"].click().run()
    assert not at.exception, at.exception

    boutons = {b.key: b for b in at.button}
    assert "btn_relink_confirmer" in boutons, list(boutons)
    boutons["btn_relink_confirmer"].click().run()
    assert not at.exception, at.exception

    meta_apres = yaml.safe_load(
        (projet / "data" / "raw" / "BV-970" / "metadata.yaml").read_text())
    assert meta_apres["source_video"] == str(nouveau_dossier / "BV-970.mp4")

    # Toutes les autres clés survivent, inchangées.
    for cle, valeur in meta_avant.items():
        if cle == "source_video":
            continue
        assert meta_apres[cle] == valeur, cle
    assert set(meta_apres) == set(meta_avant)
