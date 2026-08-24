from lib import sessions as S


def test_projet_vide(project):
    assert S.list_sessions(project).empty
    assert S.session_ids(project) == []


def test_inventaire_de_base(project, session_factory):
    session_factory("BV-970")
    session_factory("BV-971", video=False)
    df = S.list_sessions(project)
    assert list(df["session_id"]) == ["BV-970", "BV-971"]
    assert df.loc[0, "vidéo"] == "OK"
    assert df.loc[1, "vidéo"] == "manque"


def test_statut_dlc_puis_nettoyage(project, session_factory):
    session_factory("BV-970")
    dlc = project / "data" / "dlc-output" / "BV-970"
    dlc.mkdir(parents=True)
    assert S.list_sessions(project).loc[0, "DLC"] == "—"   # dossier vide
    (dlc / "abcDLC_resnet50.h5").write_bytes(b"\x00")
    assert S.list_sessions(project).loc[0, "DLC"] == "OK"
    assert S.list_sessions(project).loc[0, "nettoyage"] == "—"
    (dlc / "BV-970_clean.h5").write_bytes(b"\x00")
    assert S.list_sessions(project).loc[0, "nettoyage"] == "OK"


def test_statut_vame_teste_le_vrai_artefact(project, session_factory, vame_project):
    """L'ancien code testait data/vame/<session>, chemin inexistant."""
    session_factory("S1")
    session_factory("S2")
    df = S.list_sessions(project).set_index("session_id")
    assert df.loc["S1", "VAME"] == "OK"     # 15_hmm_label_S1.npy dans la fixture
    assert df.loc["S2", "VAME"] == "—"


def test_aucune_colonne_en_dur(project, session_factory):
    """L'Excel est à colonnes libres : rien ne doit être présupposé."""
    session_factory("S1", regime_alimentaire="gras", operateur="Leo", group="MCC")
    champs = S.metadata_fields(S.load_metadata(project, "S1"))
    assert champs["regime_alimentaire"] == "gras"
    assert champs["operateur"] == "Leo"
    assert "source_video" not in champs          # chemin, pas un facteur
    assert list(champs) == ["id", "regime_alimentaire", "operateur", "group"]


def test_metadata_fields_ignore_les_structures(project, session_factory):
    session_factory("S1", arenes=[{"id": "A1"}], camera={"fps": 30}, sexe="F")
    champs = S.metadata_fields(S.load_metadata(project, "S1"))
    assert champs == {"id": "S1", "sexe": "F"}


def test_arenes_dataframe_affiche_toutes_les_cles(project, session_factory):
    meta = {"arenes": [
        {"id": "A1", "mouse_id": "970", "condition": "MCC", "coords": [0, 0, 5, 5]},
        {"id": "A2", "mouse_id": "971", "condition": "WT", "coords": None},
    ]}
    df = S.arenes_dataframe(meta)
    assert list(df.columns) == ["id", "mouse_id", "condition", "coords"]
    assert df.loc[1, "coords"] == "(à définir)"
    assert "Stress" not in df.columns and "ANGII" not in df.columns


def test_arenes_dataframe_sans_arenes():
    assert S.arenes_dataframe({}).empty
