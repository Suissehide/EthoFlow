from lib.analysis import parse_list_columns

SORTIE = """
Colonnes exploitables comme axe de comparaison (3) :

  captopril                2 groupes : Captopril (8 sessions), Control (8 sessions)
  condition                2 groupes : MCCiECKO (8 sessions), MCCf/f (8 sessions)
  cage                     4 groupes : C0 (4 sessions), C1 (4 sessions), C2 (4 sessions), C3 (4 sessions)
"""


def test_extraction_des_colonnes():
    cols = parse_list_columns(SORTIE)
    assert [c["nom"] for c in cols] == ["captopril", "condition", "cage"]
    assert cols[0]["n_groupes"] == 2
    assert cols[2]["n_groupes"] == 4


def test_resume_des_groupes_conserve():
    cols = parse_list_columns(SORTIE)
    assert "Captopril (8 sessions)" in cols[0]["groupes"]


def test_sortie_vide():
    assert parse_list_columns("") == []
    assert parse_list_columns("Aucune colonne exploitable.") == []


def test_sortie_avec_bruit_conda():
    """`conda run` préfixe parfois des lignes qui ne sont pas du script."""
    bruit = "WARNING: overwriting environment variables\n" + SORTIE
    assert len(parse_list_columns(bruit)) == 3
