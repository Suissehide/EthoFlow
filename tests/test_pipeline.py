import pytest

from lib import pipeline as PL


def test_toutes_les_commandes_portent_project_dir_et_no_prompt(project):
    """La règle qui empêche les scripts de se figer sur input()."""
    commandes = [
        PL.sync_from_excel(project, videos_dir="/videos"),
        PL.crop_arenes(project, all_sessions=True),
        PL.run_dlc_inference(project, mode="custom", all_sessions=True),
        PL.diagnose_dlc_model(project),
        PL.prepare_vame_input(project, likelihood_threshold=0.7, max_speed=5.0),
        PL.assign_arenas(project, all_sessions=True),
        PL.inspect_session(project, all_sessions=True),
        PL.vame_stage(project, "train"),
        PL.analyze_vame(project),
        PL.motif_gif(project, session="S1"),
        PL.behavior_structure_gif(project, session="S1"),
        PL.community_dendrogram(project),
    ]
    for cmd in commandes:
        assert "--project-dir" in cmd.args, cmd.script
        assert str(project) in cmd.args, cmd.script
        assert "--no-prompt" in cmd.args, cmd.script


def test_env_par_script():
    """Se tromper d'env produit un ImportError après des minutes d'attente."""
    assert PL.SCRIPT_ENVS["run_dlc_inference.py"] == "dlc"
    # Importe deeplabcut pour dlc.filterpredictions
    assert PL.SCRIPT_ENVS["prepare_vame_input_custom.py"] == "dlc"
    # matplotlib + scipy, absents de l'env ethoflow
    assert PL.SCRIPT_ENVS["analyze_vame.py"] == "vame"
    assert PL.SCRIPT_ENVS["behavior_structure_gif.py"] == "vame"
    assert PL.SCRIPT_ENVS["community_dendrogram.py"] == "vame"
    assert PL.SCRIPT_ENVS["run_vame.py"] == "vame"
    assert PL.SCRIPT_ENVS["sync_from_excel.py"] == "ethoflow"
    assert PL.SCRIPT_ENVS["motif_gif.py"] == "ethoflow"


def test_to_argv_sans_no_capture_output(project):
    """--no-capture-output renverrait la sortie au terminal, pas au pipe."""
    argv = PL.to_argv(PL.vame_stage(project, "align"))
    assert argv[:4] == ["conda", "run", "-n", "vame"]
    assert "--no-capture-output" not in argv
    assert argv[4] == "python"
    assert argv[5].endswith("run_vame.py")


def test_vame_stage_project_dir_avant_la_sous_commande(project):
    """argparse exige --project-dir avant le sous-parseur."""
    args = PL.vame_stage(project, "segment", n_clusters=25).args
    assert args.index("--project-dir") < args.index("segment")
    assert args[args.index("segment") + 1:] == ["--n-clusters", "25"]


def test_create_project_ne_prend_pas_le_projet_courant(tmp_path):
    """Le projet n'existe pas encore : --project-dir est la cible à créer."""
    cible = tmp_path / "nouveau"
    cmd = PL.create_project(cible, kind="multi", dlc_config="/m/config.yaml")
    assert cmd.env == "ethoflow"
    assert cmd.args == [
        "--project-dir", str(cible),
        "--kind", "multi",
        "--dlc-config", "/m/config.yaml",
        "--no-prompt",
    ]


def test_create_project_sans_modele_dlc(tmp_path):
    cmd = PL.create_project(tmp_path / "n", kind="single")
    assert "--dlc-config" not in cmd.args


def test_dlc_inference_sessions_positionnelles(project):
    args = PL.run_dlc_inference(project, mode="custom", sessions=["S1", "S2"]).args
    assert "S1" in args and "S2" in args
    assert "--all" not in args


def test_dlc_inference_all_exclut_les_sessions(project):
    args = PL.run_dlc_inference(
        project, mode="superanimal", sessions=["S1"], all_sessions=True
    ).args
    assert "--all" in args
    assert "S1" not in args


def test_video_adapt_batch_size_seulement_si_video_adapt(project):
    sans = PL.run_dlc_inference(project, mode="custom", all_sessions=True).args
    assert "--video-adapt-batch-size" not in sans
    avec = PL.run_dlc_inference(
        project, mode="single-animal", all_sessions=True,
        video_adapt=True, video_adapt_batch_size=2,
    ).args
    assert avec[avec.index("--video-adapt-batch-size") + 1] == "2"


def test_prepare_vame_input_passe_les_seuils_explicitement(project):
    """Sans valeurs explicites le script les demande à l'invite."""
    args = PL.prepare_vame_input(
        project, likelihood_threshold=0.7, max_speed=4.0, px_per_cm=12.5,
        sticky_detection=False, qc_bodypart="paw_front_left",
    ).args
    assert args[args.index("--likelihood-threshold") + 1] == "0.7"
    assert args[args.index("--max-speed") + 1] == "4.0"
    assert args[args.index("--px-per-cm") + 1] == "12.5"
    assert "--no-sticky-detection" in args
    assert args[args.index("--qc-bodypart") + 1] == "paw_front_left"


def test_analyze_vame_group_by_et_cross(project):
    args = PL.analyze_vame(
        project, group_by=["sex", "cage"],
        cross=[("condition", "captopril")], extended=True,
        extended_by="condition_x_captopril",
    ).args
    assert args[args.index("--group-by") + 1:args.index("--group-by") + 3] == ["sex", "cage"]
    assert args[args.index("--cross") + 1:args.index("--cross") + 3] == ["condition", "captopril"]
    assert "--extended" in args
    assert args[args.index("--extended-by") + 1] == "condition_x_captopril"


def test_analyze_vame_cross_multiple(project):
    """--cross est action='append' : un flag par croisement."""
    args = PL.analyze_vame(
        project, cross=[("condition", "captopril"), ("sex", "cage")]
    ).args
    assert args.count("--cross") == 2


def test_analyze_vame_list_columns_est_isole(project):
    """--list-columns sort la liste et rend la main, sans produire de figures."""
    args = PL.analyze_vame(project, list_columns=True).args
    assert "--list-columns" in args
    assert "--group-by" not in args
    assert "--extended" not in args


def test_script_inconnu_refuse():
    """Un script absent de SCRIPT_ENVS doit échouer bruyamment, pas silencieusement."""
    with pytest.raises(KeyError):
        PL.to_argv(PL.Command("ethoflow", "inexistant.py", [], "x"))
