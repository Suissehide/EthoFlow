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


def test_arenes_dataframe_colonnes_uniformement_typees(project, session_factory):
    """Ruling R19.1 : une arène vide a un `mouse_id` légitimement `null`
    (scripts/crop_arenes.py) — la colonne ne doit pas mélanger int et "" au
    risque de casser la sérialisation Arrow de st.dataframe (reproduit sur
    la page Vidéos & calibration, streamlit_app/views/videos.py:227)."""
    meta = {"arenes": [
        {"id": "A1", "mouse_id": 15, "condition": "MCC", "coords": [0, 0, 5, 5]},
        {"id": "A2", "mouse_id": None, "condition": "vide", "coords": None},
    ]}
    df = S.arenes_dataframe(meta)
    types_mouse_id = {type(v) for v in df["mouse_id"]}
    assert types_mouse_id == {str}, types_mouse_id
    assert df.loc[0, "mouse_id"] == "15"
    assert df.loc[1, "mouse_id"] == ""


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


# ============================================================
# Task 21 — galerie QC : parsing des noms de fichiers _qc_trajectories/
# ============================================================

def test_parse_qc_trajectory_filename_session_simple():
    """Cas de base : session_id sans underscore, keypoint sans underscore."""
    assert S.parse_qc_trajectory_filename(
        "BV-970_tail_base.png", ["BV-970"],
    ) == ("BV-970", "tail_base")


def test_parse_qc_trajectory_filename_session_id_avec_underscore():
    """Session éclatée par arène (BV-970_A1) : un split naïf sur `_`
    couperait après `BV-970`, pas après `BV-970_A1`."""
    assert S.parse_qc_trajectory_filename(
        "BV-970_A1_tail_base.png", ["BV-970_A1"],
    ) == ("BV-970_A1", "tail_base")


def test_parse_qc_trajectory_filename_session_id_avec_underscore_prefere_le_plus_long():
    """Si BV-970 ET BV-970_A1 sont tous deux des session_id connus, le
    préfixe le plus long et le plus spécifique l'emporte."""
    connus = ["BV-970", "BV-970_A1"]
    assert S.parse_qc_trajectory_filename(
        "BV-970_A1_tail_base.png", connus,
    ) == ("BV-970_A1", "tail_base")
    assert S.parse_qc_trajectory_filename(
        "BV-970_tail_base.png", connus,
    ) == ("BV-970", "tail_base")


def test_parse_qc_trajectory_filename_keypoint_avec_underscores():
    """Keypoint lui-même composé (paw_front_left) : tout ce qui suit le
    session_id connu est le keypoint, quel que soit son nombre de `_`."""
    assert S.parse_qc_trajectory_filename(
        "BV-970_paw_front_left.png", ["BV-970"],
    ) == ("BV-970", "paw_front_left")


def test_parse_qc_trajectory_filename_session_inconnue():
    """Aucun session_id connu ne préfixe le nom : None, jamais une supposition."""
    assert S.parse_qc_trajectory_filename(
        "XYZ-999_tail_base.png", ["BV-970"],
    ) is None


def test_list_qc_trajectories(project, session_factory, tmp_path):
    session_factory("BV-970")
    session_factory("BV-970_A1")
    qc_dir = S.qc_trajectories_dir(project)
    qc_dir.mkdir(parents=True)
    (qc_dir / "BV-970_tail_base.png").write_bytes(b"\x00")
    (qc_dir / "BV-970_A1_tail_base.png").write_bytes(b"\x00")
    (qc_dir / "BV-970_paw_front_left.png").write_bytes(b"\x00")
    (qc_dir / "orpheline_tail_base.png").write_bytes(b"\x00")  # session inconnue

    galerie = S.list_qc_trajectories(project)
    assert set(galerie) == {"tail_base", "paw_front_left"}
    assert set(galerie["tail_base"]) == {"BV-970", "BV-970_A1"}
    assert galerie["tail_base"]["BV-970"] == qc_dir / "BV-970_tail_base.png"
    assert set(galerie["paw_front_left"]) == {"BV-970"}


def test_list_qc_trajectories_dossier_absent(project):
    assert S.list_qc_trajectories(project) == {}
