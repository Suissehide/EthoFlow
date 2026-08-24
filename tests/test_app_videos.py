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


def _pipeline_config(projet: Path) -> dict:
    chemin = projet / "configs" / "pipeline_config.yaml"
    return yaml.safe_load(chemin.read_text()) if chemin.is_file() else {}


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


def test_repointage_preserve_accents_et_structures_imbriquees(
    tmp_path, monkeypatch, video_reelle,
):
    """Le re-pointage est le seul chemin destructif de cette page (il
    réécrit `metadata.yaml`) : on vérifie explicitement que des valeurs
    accentuées et une structure `arenes` imbriquée le traversent intactes,
    pas seulement des clés scalaires ASCII comme dans le test ci-dessus."""
    projet = _projet(tmp_path, kind="multi")
    chemin_mort = tmp_path / "D" / "ancien" / "chemin" / "OF-M1.mp4"
    arenes = [
        {"id": "A1", "mouse_id": 15, "condition": "SHAM", "coords": [0, 0, 5, 5]},
        {"id": "A2", "mouse_id": None, "condition": "CUS+ANGII", "coords": None},
    ]
    _ecrire_session(
        projet, "OF-M1", source_video=chemin_mort,
        camera={"fps": 25, "width": 640, "height": 480},
        extra={
            "opérateur": "Léo Couffinhal",
            "notes": "arène désinfectée à l'éthanol après chaque essai",
            "arenes": arenes,
        },
    )
    meta_avant = yaml.safe_load(
        (projet / "data" / "raw" / "OF-M1" / "metadata.yaml").read_text())

    nouveau_dossier = tmp_path / "videos-retrouvees"
    video_reelle(nouveau_dossier, nom="OF-M1.mp4", fps=10.0, n=20, w=64, h=48)

    at = _lancer_sur_projet(tmp_path, monkeypatch, projet)
    assert not at.exception, at.exception  # la page rend déjà le tableau des arènes ici

    champs = {t.key: t for t in at.text_input}
    champs["videos_relink_dossier"].set_value(str(nouveau_dossier)).run()
    boutons = {b.key: b for b in at.button}
    boutons["btn_relink_demander"].click().run()
    boutons = {b.key: b for b in at.button}
    boutons["btn_relink_confirmer"].click().run()
    assert not at.exception, at.exception

    meta_apres = yaml.safe_load(
        (projet / "data" / "raw" / "OF-M1" / "metadata.yaml").read_text())
    assert meta_apres["source_video"] == str(nouveau_dossier / "OF-M1.mp4")
    assert meta_apres["opérateur"] == "Léo Couffinhal"
    assert meta_apres["notes"] == meta_avant["notes"]
    assert meta_apres["arenes"] == arenes          # structure imbriquée intacte
    assert set(meta_apres) == set(meta_avant)


# ============================================================
# Task 20 — Calibration arènes / Échelle px/cm
# ============================================================
#
# `streamlit_image_coordinates` est un composant custom (iframe) : AppTest
# n'a pas de méthode pour lui simuler un clic de souris (contrairement à
# `.button.click()` ou `.text_input.set_value()`, qui existent pour les
# widgets natifs). Vérifié empiriquement : dans l'arbre d'éléments, il
# apparaît comme `UnknownElement` avec `.type == "component_instance"`, et
# cette classe n'expose qu'une propriété `.value` en lecture seule.
#
# Ce que le composant retourne à Streamlit est en revanche un contrat
# documenté et stable : un dict `{"x", "y", "width", "height", "unix_time"}`
# stocké sous `st.session_state[cle_du_widget]` — exactement ce que
# `streamlit.testing.v1.AppTest` permet d'écrire directement AVANT
# `.run()`. On s'en sert ici pour piloter le vrai code de production (la
# détection de « nouveau clic » via `unix_time`, l'accumulation en
# rectangle/segment, l'écriture finale) sans jamais avoir à interagir avec
# le rendu de l'image elle-même. Ce n'est pas un test qui « fait semblant » :
# c'est la même frontière que le composant utilise pour parler à Streamlit,
# poussée à la main plutôt que par un vrai navigateur.
#
# La géométrie pure (deux clics → rectangle, deux clics → distance) est
# testée séparément et sans AppTest dans `tests/test_video.py`.

def _cle_clic_arenes(chemin, frame_idx: int) -> str:
    return f"calib_arenes_clic__session:{chemin}#{frame_idx}"


def _cle_clic_echelle_video(chemin, frame_idx: int) -> str:
    return f"echelle_clic__session:{chemin}#{frame_idx}"


def test_calibration_arenes_quatre_clics_ajustement_et_enregistrement(
    tmp_path, monkeypatch, video_reelle,
):
    projet = _projet(tmp_path, kind="multi")
    video = video_reelle(tmp_path / "videos", fps=10.0, n=20, w=64, h=48)
    _ecrire_session(projet, "S1", source_video=video)

    at = _lancer_sur_projet(tmp_path, monkeypatch, projet)

    onglets = {t.proto.label: t for t in at.tabs}
    assert "Calibration arènes" in onglets, list(onglets)

    # Frame par défaut = min(n_frames // 2, n_frames - 1) = 10 pour n=20.
    cle_clic = _cle_clic_arenes(video, 10)

    # Quatre paires de clics (coins opposés) → quatre rectangles A1..A4.
    points = [
        ((5, 5), (15, 15)),
        ((20, 5), (30, 20)),
        ((5, 25), (15, 35)),
        ((20, 25), (35, 40)),
    ]
    unix_time = 1.0
    for p1, p2 in points:
        at.session_state[cle_clic] = {"x": p1[0], "y": p1[1], "unix_time": unix_time}
        at.run()
        assert not at.exception, at.exception
        unix_time += 1
        at.session_state[cle_clic] = {"x": p2[0], "y": p2[1], "unix_time": unix_time}
        at.run()
        assert not at.exception, at.exception
        unix_time += 1

    attendu = {
        "A1": [5, 5, 10, 10],
        "A2": [20, 5, 10, 15],
        "A3": [5, 25, 10, 10],
        "A4": [20, 25, 15, 15],
    }
    assert at.session_state["calib_arenes_rects"] == [attendu[f"A{i+1}"] for i in range(4)]

    # Ajustement fin : nudge de A1.x de +3 px via le number_input dédié.
    nums = {n.key: n for n in at.number_input}
    assert "calib_arenes_A1_x" in nums, list(nums)
    nums["calib_arenes_A1_x"].set_value(8).run()
    assert not at.exception, at.exception

    boutons = {b.key: b for b in at.button}
    assert "calib_arenes_enregistrer" in boutons, list(boutons)
    boutons["calib_arenes_enregistrer"].click().run()
    assert not at.exception, at.exception

    cfg = _pipeline_config(projet)
    ecrites = cfg["default_arenes_coords"]
    assert ecrites["A1"] == [8, 5, 10, 10]   # x nudgé, reste inchangé
    assert ecrites["A2"] == attendu["A2"]
    assert ecrites["A3"] == attendu["A3"]
    assert ecrites["A4"] == attendu["A4"]


def test_calibration_arenes_bouton_enregistrer_absent_avant_quatre_arenes(
    tmp_path, monkeypatch, video_reelle,
):
    projet = _projet(tmp_path, kind="multi")
    video = video_reelle(tmp_path / "videos", fps=10.0, n=20, w=64, h=48)
    _ecrire_session(projet, "S1", source_video=video)

    at = _lancer_sur_projet(tmp_path, monkeypatch, projet)
    cle_clic = _cle_clic_arenes(video, 10)

    at.session_state[cle_clic] = {"x": 5, "y": 5, "unix_time": 1.0}
    at.run()
    at.session_state[cle_clic] = {"x": 15, "y": 15, "unix_time": 2.0}
    at.run()
    assert not at.exception, at.exception

    assert len(at.session_state["calib_arenes_rects"]) == 1
    boutons = {b.key: b for b in at.button}
    assert "calib_arenes_enregistrer" not in boutons
    assert "calib_arenes_recommencer" in boutons
    assert _pipeline_config(projet).get("default_arenes_coords") in (None, {})


# ------------------------------------------------------------
# Ruling R20.1 (fix round 1/5) — paire de clics dégénérée sur les arènes.
#
# Avant le correctif : deux clics au même endroit produisent un rectangle
# [x, y, 0, 0] qui est ajouté à `calib_arenes_rects` tel quel. Le rendu
# suivant de l'ajustement fin appelle `number_input("largeur", value=0,
# min_value=1, ...)` → `StreamlitValueBelowMinError` non rattrapée, qui fait
# planter toute la page, pas seulement ce widget. Le correctif refuse la
# paire à la source (avant qu'elle n'entre dans `rects`) plutôt que
# d'élargir `min_value`, ce qui laisserait enregistrer une arène d'aire
# nulle.
# ------------------------------------------------------------

def test_calibration_arenes_clics_identiques_refuses_sans_crash(
    tmp_path, monkeypatch, video_reelle,
):
    projet = _projet(tmp_path, kind="multi")
    video = video_reelle(tmp_path / "videos", fps=10.0, n=20, w=64, h=48)
    _ecrire_session(projet, "S1", source_video=video)

    at = _lancer_sur_projet(tmp_path, monkeypatch, projet)
    cle_clic = _cle_clic_arenes(video, 10)

    at.session_state[cle_clic] = {"x": 10, "y": 10, "unix_time": 1.0}
    at.run()
    assert not at.exception, at.exception
    at.session_state[cle_clic] = {"x": 10, "y": 10, "unix_time": 2.0}  # même pixel
    at.run()
    assert not at.exception, at.exception  # ne doit plus planter la page

    # Aucun rectangle dégénéré enregistré, aucune écriture de config...
    assert at.session_state["calib_arenes_rects"] == []
    assert _pipeline_config(projet).get("default_arenes_coords") in (None, {})
    # ... et le premier clic reste enregistré : pas besoin de tout recommencer.
    assert at.session_state["calib_arenes_premier_clic"] == (10, 10)
    erreurs = [e.value for e in at.error]
    assert erreurs, "un message explicite doit signaler le clic refusé"

    # La reprise fonctionne : un second clic distinct complète bien A1.
    at.session_state[cle_clic] = {"x": 20, "y": 20, "unix_time": 3.0}
    at.run()
    assert not at.exception, at.exception
    assert at.session_state["calib_arenes_rects"] == [[10, 10, 10, 10]]


def test_calibration_arenes_clics_identiques_sur_un_seul_axe_refuses(
    tmp_path, monkeypatch, video_reelle,
):
    """Une paire qui ne diverge que sur un axe (même x, y différent, ou
    l'inverse) donne une largeur ou une hauteur nulle — tout aussi inutile
    qu'une paire totalement identique ; même garde attendue."""
    projet = _projet(tmp_path, kind="multi")
    video = video_reelle(tmp_path / "videos", fps=10.0, n=20, w=64, h=48)
    _ecrire_session(projet, "S1", source_video=video)

    at = _lancer_sur_projet(tmp_path, monkeypatch, projet)
    cle_clic = _cle_clic_arenes(video, 10)

    at.session_state[cle_clic] = {"x": 10, "y": 10, "unix_time": 1.0}
    at.run()
    assert not at.exception, at.exception
    at.session_state[cle_clic] = {"x": 10, "y": 25, "unix_time": 2.0}  # même x → largeur nulle
    at.run()
    assert not at.exception, at.exception

    assert at.session_state["calib_arenes_rects"] == []
    assert at.session_state["calib_arenes_premier_clic"] == (10, 10)
    erreurs = [e.value for e in at.error]
    assert erreurs, "un message explicite doit signaler le clic refusé"
    assert _pipeline_config(projet).get("default_arenes_coords") in (None, {})


def test_echelle_deux_clics_calcule_et_enregistre_px_per_cm(
    tmp_path, monkeypatch, video_reelle,
):
    projet = _projet(tmp_path, kind="single")
    video = video_reelle(tmp_path / "videos", fps=10.0, n=20, w=64, h=48)
    _ecrire_session(projet, "S1", source_video=video)

    at = _lancer_sur_projet(tmp_path, monkeypatch, projet)

    radios = {r.key: r for r in at.radio}
    assert "echelle_source_mode" in radios
    radios["echelle_source_mode"].set_value("Frame d'une vidéo de session").run()
    assert not at.exception, at.exception

    cle_clic = _cle_clic_echelle_video(video, 10)
    at.session_state[cle_clic] = {"x": 10, "y": 10, "unix_time": 1.0}
    at.run()
    at.session_state[cle_clic] = {"x": 10, "y": 30, "unix_time": 2.0}  # 20 px verticaux
    at.run()
    assert not at.exception, at.exception

    nums = {n.key: n for n in at.number_input}
    assert "echelle_known_cm" in nums, list(nums)
    nums["echelle_known_cm"].set_value(10.0).run()
    assert not at.exception, at.exception

    # Distance mesurée affichée : 20 px pour 10 cm déclarés → 2 px/cm.
    textes = [w.value for w in at.markdown] + [c.value for c in at.caption]
    assert any("20.0" in t and "px" in t for t in textes), textes

    boutons = {b.key: b for b in at.button}
    assert "echelle_enregistrer_clics" in boutons, list(boutons)
    boutons["echelle_enregistrer_clics"].click().run()
    assert not at.exception, at.exception

    assert _pipeline_config(projet)["px_per_cm"] == pytest.approx(2.0)


def test_echelle_clics_identiques_narrete_pas_le_bouton_enregistrer(
    tmp_path, monkeypatch, video_reelle,
):
    """Ruling R20.1 (fix round 1/5), volet mineur : deux clics au même
    endroit donnent une distance nulle donc `px_per_cm = 0.0`. Avant le
    correctif, rien n'empêchait d'enregistrer cette valeur — un zéro
    silencieux dans `pipeline_config.yaml` casserait le filtre de vitesses
    aberrantes du nettoyage VAME sans aucun message. Le bouton d'enregistrement
    ne doit plus exister dans ce cas, avec une explication à la place."""
    projet = _projet(tmp_path, kind="single")
    video = video_reelle(tmp_path / "videos", fps=10.0, n=20, w=64, h=48)
    _ecrire_session(projet, "S1", source_video=video)

    at = _lancer_sur_projet(tmp_path, monkeypatch, projet)

    radios = {r.key: r for r in at.radio}
    radios["echelle_source_mode"].set_value("Frame d'une vidéo de session").run()
    assert not at.exception, at.exception

    cle_clic = _cle_clic_echelle_video(video, 10)
    at.session_state[cle_clic] = {"x": 10, "y": 10, "unix_time": 1.0}
    at.run()
    at.session_state[cle_clic] = {"x": 10, "y": 10, "unix_time": 2.0}  # même pixel
    at.run()
    assert not at.exception, at.exception

    boutons = {b.key: b for b in at.button}
    assert "echelle_enregistrer_clics" not in boutons, list(boutons)
    erreurs = [e.value for e in at.error]
    assert erreurs, "un message explicite doit signaler la distance nulle"
    assert _pipeline_config(projet).get("px_per_cm") in (None,)


def test_echelle_photo_importee_comme_source_alternative(tmp_path, monkeypatch):
    """Le brief demande explicitement une source « photo de règle », pas
    seulement des frames de vidéo (conseil du README sur la distorsion de
    lentille)."""
    cv2 = pytest.importorskip("cv2")
    projet = _projet(tmp_path, kind="single")
    at = _lancer_sur_projet(tmp_path, monkeypatch, projet)

    radios = {r.key: r for r in at.radio}
    radios["echelle_source_mode"].set_value("Photo importée (règle)").run()
    assert not at.exception, at.exception

    uploaders = {u.key: u for u in at.file_uploader}
    assert "echelle_upload" in uploaders, list(uploaders)

    ok, buf = cv2.imencode(".png", np.zeros((30, 30, 3), dtype=np.uint8))
    assert ok
    uploaders["echelle_upload"].set_value(("regle.png", buf.tobytes(), "image/png")).run()
    assert not at.exception, at.exception

    cle_clic = "echelle_clic__image:regle.png#" + str(len(buf.tobytes()))
    at.session_state[cle_clic] = {"x": 2, "y": 2, "unix_time": 1.0}
    at.run()
    at.session_state[cle_clic] = {"x": 2, "y": 12, "unix_time": 2.0}
    at.run()
    assert not at.exception, at.exception
    assert at.session_state["echelle_points"] == [(2, 2), (2, 12)]


def test_echelle_saisie_directe_equivalent_du_set(tmp_path, monkeypatch):
    """Équivalent de `calibrate_scale.py --set 12.5` : pas de clic du tout."""
    projet = _projet(tmp_path, kind="single")
    at = _lancer_sur_projet(tmp_path, monkeypatch, projet)

    nums = {n.key: n for n in at.number_input}
    assert "echelle_valeur_directe" in nums
    nums["echelle_valeur_directe"].set_value(12.5).run()
    assert not at.exception, at.exception

    boutons = {b.key: b for b in at.button}
    assert "echelle_enregistrer_directe" in boutons
    boutons["echelle_enregistrer_directe"].click().run()
    assert not at.exception, at.exception

    assert _pipeline_config(projet)["px_per_cm"] == pytest.approx(12.5)


def test_echelle_valeur_actuelle_affichee_si_deja_configuree(tmp_path, monkeypatch):
    projet = _projet(tmp_path, kind="single")
    (projet / "configs" / "pipeline_config.yaml").write_text(
        yaml.safe_dump({"kind": "single", "px_per_cm": 4.0}), encoding="utf-8",
    )
    at = _lancer_sur_projet(tmp_path, monkeypatch, projet)
    textes = [c.value for c in at.caption]
    assert any("4.000 px/cm" in t for t in textes), textes
