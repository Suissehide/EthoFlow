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
    """Le témoin de `run_vame.py align` marque align comme fait."""
    (vame_project / "data" / "processed").mkdir(parents=True)
    (vame_project / "data" / "processed" / ".ethoflow_align_ok").touch()
    assert V.stage_status(project)["align"] is True


def test_stage_status_align_pas_fait_juste_apres_setup(project, vame_project):
    """`init_new_project` écrit déjà les *_processed.nc : ça ne prouve pas
    l'alignement (l'app disait « Aligner les poses » fait avant tout clic)."""
    (vame_project / "data" / "processed").mkdir(parents=True)
    (vame_project / "data" / "processed" / "S1_processed.nc").write_text("mock")
    assert V.stage_status(project)["align"] is False


def test_training_losses_et_meilleur_epoch(project, vame_project):
    import numpy as np
    from lib import vame as VA
    d = vame_project / "model" / "model_losses"
    d.mkdir(parents=True)
    np.save(d / "train_losses_VAME.npy", np.array([9.0, 5, 4, 3, 2.5]))
    np.save(d / "test_losses_VAME.npy", np.array([1.0, 6, 5, 3.5, 4]))   # 1.0 : pendant l'annealing
    np.save(d / "weight_values_VAME.npy", np.array([0.2, 0.6, 1, 1, 1]))
    np.save(d / "kl_losses_VAME.npy", np.array([1.0, 1, 1, 1]))          # une epoch de retard
    (d / "kmeans_losses_VAME.npy").write_bytes(b"illisible")

    df = VA.training_losses(project)
    assert list(df["epoch"]) == [1, 2, 3, 4]
    assert "kmeans" not in df and "kl" in df
    assert VA.best_epoch(df) == 4


def test_training_losses_absent(project, vame_project):
    from lib import vame as VA
    assert VA.training_losses(project) is None


def test_session_bottomview_nommee_d_apres_le_clean_h5(project, vame_project):
    """VAME nomme la session `BV-X_clean` (nom du .h5) : elle doit compter
    pour la session `BV-X` du projet, sinon l'étape Motifs reste fermée."""
    import numpy as np
    d = vame_project / "results" / "BV-X_clean" / "VAME" / "hmm-15"
    d.mkdir(parents=True)
    np.save(d / "15_hmm_label_BV-X_clean.npy", np.zeros(10, dtype=int))
    np.save(d / "motif_usage_BV-X_clean.npy", np.ones(15))
    assert V.session_has_labels(project, "BV-X")
    assert "BV-X" in set(V.motif_usage_df(project, "hmm-15")["session"])


def test_parse_session_name_retire_clean():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from analyze_vame import parse_session_name
    assert parse_session_name("BV-OF-M1-M01_clean") == ("BV-OF-M1-M01", "")
    assert parse_session_name("OF-M1-V01_A1") == ("OF-M1-V01", "A1")
