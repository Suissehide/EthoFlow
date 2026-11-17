import math

from lib import schema_souris as SCH


def test_couleur_rouge_a_vert_et_gris_sans_valeur():
    assert SCH.couleur(0) == "#e34948" and SCH.couleur(100) == "#2f9567"
    assert SCH.couleur(50) == "#eda100"
    assert SCH.couleur(None) == SCH.couleur(math.nan) == "#b9b5ac"
    assert SCH.style_cellule(100) == "background-color: #d6efdf"


def test_svg_place_les_points_connus_et_liste_les_autres():
    s = SCH.svg({"nose": 95.0, "tail_tip": 12.3, "truc": 50.0}, "S1")
    assert "nez" in s and "95 %" in s and "bout queue" in s and "12 %" in s
    assert "Hors schéma : truc 50 %" in s
    assert s.count("<circle") == 2 + 2  # 2 points + 2 oreilles de la silhouette
