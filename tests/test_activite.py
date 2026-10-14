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
