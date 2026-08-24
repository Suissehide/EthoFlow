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


def test_colonne_video_en_tant_que_facteur(project):
    """Une colonne 'video' nommée par l'utilisateur survit : elle n'est pas en dur.

    session_factory(`video=True/False`) crée un fichier, mais si l'Excel contient
    une colonne `video` scalaire, elle doit être un facteur expérimental valide.
    Écrit le metadata.yaml directement pour éviter la collision avec le paramètre.
    """
    import yaml
    sdir = project / "data" / "raw" / "S1"
    sdir.mkdir(parents=True)
    meta = {"id": "S1", "video": "HD", "source_video": "/tmp/video.mp4"}
    (sdir / "metadata.yaml").write_text(
        yaml.safe_dump(meta, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    champs = S.metadata_fields(S.load_metadata(project, "S1"))
    assert "video" in champs
    assert champs["video"] == "HD"


def test_arenes_et_camera_exclues_par_type(project):
    """Après suppression de 'arenes' et 'camera' de _NON_FACTEURS, le filtre isinstance()
    les exclut encore. C'est une régression : le type check doit suffire.
    """
    import yaml
    sdir = project / "data" / "raw" / "S1"
    sdir.mkdir(parents=True)
    meta = {
        "id": "S1",
        "arenes": [{"id": "A1"}],
        "camera": {"fps": 30, "resolution": "1080p"},
        "source_video": "/tmp/video.mp4",
        "temperature": 22.5,
    }
    (sdir / "metadata.yaml").write_text(
        yaml.safe_dump(meta, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    champs = S.metadata_fields(S.load_metadata(project, "S1"))
    # Structurées (dicts, listes) exclues par isinstance
    assert "arenes" not in champs
    assert "camera" not in champs
    # source_video exclu explicitement
    assert "source_video" not in champs
    # Scalaire reste
    assert champs["temperature"] == 22.5
