"""views/_activite.temps_restant : extrapolation du temps écoulé."""
from datetime import datetime

from views._activite import temps_restant

DEBUT = "2026-01-01T14:00:00"


def test_temps_restant():
    a = lambda minutes, v: temps_restant(DEBUT, v, datetime(2026, 1, 1, 14, minutes))
    assert a(30, 0.5) == "≈ 30 min"
    assert a(10, 0.1) == "≈ 1 h 30"
    assert a(59, 0.999) == "< 1 min"
    assert a(1, 0.01) is None          # trop tôt pour estimer
    assert temps_restant("pas une date", 0.5) is None


def test_periode_debut_fin():
    from views._activite import periode
    assert periode("2026-11-17T10:02:00", "2026-11-17T10:41:00") == "17/11 10:02 → 10:41"
    assert periode("2026-11-17T23:50:00", "2026-11-18T01:10:00") == "17/11 23:50 → 18/11 01:10"
    assert periode("2026-11-17T10:02:00", None) == "17/11 10:02"
