"""Nettoyage des poses : excursions vers un point fixe, immobilité réelle."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from pose_cleaning import detect_sticky_points, detect_velocity_outliers  # noqa: E402

FPS, PX_PER_CM = 50.0, 40.0   # seuil 5 m/s = 400 px par image
POINT_FAUX = (1990.0, 340.0)


def _immobile(n=6000, seed=0):
    """Souris immobile dans un coin : tremblement de ~2 px."""
    rng = np.random.default_rng(seed)
    return 1800 + rng.normal(0, 2, n), 1700 + rng.normal(0, 2, n)


def _avec_excursions(x, y, debuts, duree):
    x, y, faux = x.copy(), y.copy(), np.zeros(len(x), bool)
    for d in debuts:
        x[d:d + duree], y[d:d + duree] = POINT_FAUX
        faux[d:d + duree] = True
    return x, y, faux


def test_immobilite_reelle_jamais_marquee():
    x, y = _immobile()
    assert not detect_velocity_outliers(x, y, FPS, PX_PER_CM).any()
    flag, points = detect_sticky_points(x, y)
    assert not flag.any() and points == []


def test_excursion_de_plusieurs_images_entierement_marquee():
    x, y = _immobile()
    x, y, faux = _avec_excursions(x, y, range(200, 5800, 300), duree=10)
    flag = detect_velocity_outliers(x, y, FPS, PX_PER_CM)
    assert (flag == faux).all()   # tout le séjour, et pas l'image du retour


def test_saut_court_vers_point_fixe_attrape_par_les_collants():
    # Souris qui se promène près du point faux : sauts < 400 px, la
    # vitesse ne les voit pas, mais on arrive toujours sur ce point de loin.
    rng = np.random.default_rng(1)
    t = np.arange(6000)
    x = 1900 + 60 * np.sin(t / 200) + rng.normal(0, 2, 6000)
    y = 500 + 150 * np.cos(t / 300) + rng.normal(0, 2, 6000)
    x, y, faux = _avec_excursions(x, y, range(100, 5900, 250), duree=8)
    assert not detect_velocity_outliers(x, y, FPS, PX_PER_CM)[faux].all()
    flag, points = detect_sticky_points(x, y)
    assert flag[faux].all()
    assert flag[~faux].sum() == 0
    assert points and np.allclose(points[0], POINT_FAUX)


def test_saut_sans_retour_non_marque():
    x, y = _immobile(1000)
    x[500:] += 900   # vrai déplacement ou reprise de suivi
    assert not detect_velocity_outliers(x, y, FPS, PX_PER_CM).any()
