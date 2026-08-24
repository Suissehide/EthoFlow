import numpy as np
import pytest
import yaml

from lib import video as V


@pytest.fixture
def petite_video(tmp_path):
    cv2 = pytest.importorskip("cv2")
    chemin = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(
        str(chemin), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (64, 48))
    for i in range(20):
        frame = np.full((48, 64, 3), i * 10, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    if not chemin.exists() or chemin.stat().st_size == 0:
        pytest.skip("encodeur mp4v indisponible")
    return chemin


def test_probe(petite_video):
    info = V.probe(petite_video)
    assert info.exists
    assert info.width == 64 and info.height == 48
    assert abs(info.fps - 10.0) < 0.5
    assert info.n_frames == 20
    assert abs(info.duration_s - 2.0) < 0.3


def test_probe_fichier_absent(tmp_path):
    info = V.probe(tmp_path / "nexiste-pas.mp4")
    assert not info.exists
    assert info.fps is None and info.n_frames is None


def test_grab_frame(petite_video):
    frame = V.grab_frame(petite_video, index=5)
    assert frame is not None and frame.shape == (48, 64, 3)


def test_grab_frame_hors_bornes(petite_video):
    assert V.grab_frame(petite_video, index=9999) is None


def test_frame_png_bytes(petite_video):
    data = V.frame_png_bytes(petite_video, index=0)
    assert data.startswith(b"\x89PNG")


def test_draw_arenas_ne_modifie_pas_loriginal():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    sortie = V.draw_arenas(frame, {"A1": [10, 10, 30, 30]})
    assert frame.sum() == 0            # l'original est intact
    assert sortie.sum() > 0            # le rectangle est dessiné


def test_relink_retrouve_les_videos_deplacees(project, session_factory, tmp_path):
    """Cas du README : metadata avec chemins Windows sur machine Linux."""
    session_factory("BV-970", video=False)
    (project / "data" / "raw" / "BV-970" / "metadata.yaml").write_text(
        yaml.safe_dump({"id": "BV-970",
                        "source_video": r"D:\ancien\chemin\BV-970.mp4"}))
    nouveau = tmp_path / "videos"
    nouveau.mkdir()
    (nouveau / "BV-970.mp4").write_bytes(b"\x00")

    relinks = V.find_relinks(project, nouveau)
    assert relinks == [("BV-970", nouveau / "BV-970.mp4")]
    assert V.apply_relinks(project, relinks) == 1

    meta = yaml.safe_load(
        (project / "data" / "raw" / "BV-970" / "metadata.yaml").read_text())
    assert meta["source_video"] == str(nouveau / "BV-970.mp4")


def test_relink_ignore_les_sessions_deja_ok(project, session_factory, tmp_path):
    session_factory("BV-971")           # vidéo présente et valide
    assert V.find_relinks(project, tmp_path) == []


# ============================================================
# Géométrie des clics (Task 20) — logique pure sous les onglets de calibration.
# ============================================================

def test_to_rgb_inverse_les_canaux():
    # Pixel bleu pur en BGR (convention OpenCV) : B=255, G=0, R=0.
    frame_bgr = np.zeros((2, 2, 3), dtype=np.uint8)
    frame_bgr[:, :, 0] = 255
    rgb = V.to_rgb(frame_bgr)
    # En RGB, le même pixel doit porter la valeur sur le canal R (indice 0).
    assert (rgb[:, :, 0] == 0).all()
    assert (rgb[:, :, 2] == 255).all()


def test_rect_from_two_points_coins_opposes_ordre_quelconque():
    # Coin bas-droit cliqué en premier, coin haut-gauche en second.
    assert V.rect_from_two_points((30, 40), (10, 10)) == [10, 10, 20, 30]


def test_rect_from_two_points_meme_resultat_quel_que_soit_lordre_des_clics():
    a, b = (10, 10), (30, 40)
    assert V.rect_from_two_points(a, b) == V.rect_from_two_points(b, a)


def test_distance_from_two_points_pythagore():
    assert V.distance_from_two_points((0, 0), (3, 4)) == 5.0


def test_distance_from_two_points_symetrique():
    p1, p2 = (5, 5), (12, 20)
    assert V.distance_from_two_points(p1, p2) == V.distance_from_two_points(p2, p1)


def test_draw_scale_line_ne_modifie_pas_loriginal():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    sortie = V.draw_scale_line(frame, (10, 10), (60, 10))
    assert frame.sum() == 0
    assert sortie.sum() > 0
