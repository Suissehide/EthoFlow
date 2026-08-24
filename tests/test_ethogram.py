from lib import config as C
from lib import motif_labels as ML


def test_categories_viennent_des_scripts():
    """`category` est une liste fermée, groupée par les analyses."""
    assert C.categories() == ML.categories()
    assert "Arena-specific" not in C.categories()   # n'existait que côté app
    assert "Catch-all" not in C.categories()


def test_vocabulaire_suggere_est_libre():
    """Le vocabulaire aide à remplir `label`, qui lui est libre."""
    voc = C.VOCABULAIRE_SUGGERE
    assert "grooming face" in voc["Grooming"]
    assert "thigmotaxis" in voc["Arena-specific"]


def test_vocabulaire_et_categories_sont_deux_choses():
    """Aucun test d'égalité entre les deux : ce sont des rôles distincts."""
    assert set(C.VOCABULAIRE_SUGGERE) != set(C.categories())
