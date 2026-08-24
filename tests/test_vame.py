from lib import vame as V


def test_racine_plate(project, vame_project):
    assert V.vame_project(project) == project / "data" / "vame"
    assert V.is_initialised(project)


def test_projet_sans_vame(project):
    assert not V.is_initialised(project)
    assert V.list_algos(project) == []
    assert V.list_sessions(project) == []
    assert V.n_clusters(project) is None
    assert V.motif_usage_df(project, "hmm-15").empty


def test_lecture_du_config(project, vame_project):
    assert V.n_clusters(project) == 15
    assert V.read_config(project)["segmentation_algorithms"] == ["hmm"]


def test_detection_des_algos(project, vame_project):
    assert V.list_algos(project) == ["hmm-15"]
    (vame_project / "results" / "S1" / "VAME" / "kmeans-25").mkdir(parents=True)
    assert V.list_algos(project) == ["hmm-15", "kmeans-25"]


def test_parse_algo_n():
    assert V.parse_algo_n("hmm-15") == ("hmm", 15)
    assert V.parse_algo_n("kmeans-25") == ("kmeans", 25)


def test_parse_algo_n_invalide():
    import pytest
    with pytest.raises(ValueError):
        V.parse_algo_n("hmm")


def test_motif_usage_df(project, vame_project):
    df = V.motif_usage_df(project, "hmm-15")
    assert set(df.columns) == {"session", "motif", "count", "frequency"}
    assert len(df) == 15
    assert abs(df["frequency"].sum() - 1.0) < 1e-9


def test_stage_status_progression(project, vame_project):
    """Le stepper de la page VAME lit ça pour savoir où on en est."""
    etat = V.stage_status(project)
    assert etat["setup"] is True
    assert etat["segment"] is True      # 15_hmm_label_S1.npy présent
    assert etat["train"] is False       # pas de model/


def test_stage_status_train_not_started(project, vame_project):
    """Répertoire vide ne compte pas comme entraîné."""
    (vame_project / "model" / "best_model").mkdir(parents=True)
    assert V.stage_status(project)["train"] is False


def test_stage_status_train_complete(project, vame_project):
    """Modèle sauvegardé marque l'entraînement comme fait."""
    (vame_project / "model" / "best_model").mkdir(parents=True)
    (vame_project / "model" / "best_model" / "rnn_vae_test_project.pkl").write_text("mock")
    assert V.stage_status(project)["train"] is True


def test_stage_status_align_no_project(project):
    """Projet sans VAME ne marque pas align comme fait."""
    assert V.stage_status(project)["align"] is False


def test_stage_status_align_complete(project, vame_project):
    """Fichier processed marque align comme fait."""
    (vame_project / "data" / "processed").mkdir(parents=True)
    (vame_project / "data" / "processed" / "S1_processed.nc").write_text("mock")
    assert V.stage_status(project)["align"] is True
