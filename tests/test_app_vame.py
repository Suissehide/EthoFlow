"""Vérification de bout en bout de la page VAME via AppTest.

Isolation : `lib.project.PREFS_PATH` monkeypatché (comme
`tests/test_app_nettoyage.py`) pour ne jamais toucher `Path.home()` réel ni
`DEFAULT_PROJECTS_ROOT` (`D:\\EthoFlow\\projects`, un nom de dossier
littéral et relatif sur ce runner macOS).

Le point central de ces tests, en écho à la consigne d'honnêteté de la
Task 15 : `train: True` ne doit JAMAIS s'afficher comme « entraînement
terminé », et `align: True` ne doit jamais laisser croire que toutes les
sessions sont alignées (c'est un `any()`, pas un `all()`).
"""
from __future__ import annotations

from pathlib import Path

import yaml
from streamlit.testing.v1 import AppTest

from lib import project as P

REPO = Path(__file__).resolve().parent.parent
APP_PY = str(REPO / "streamlit_app" / "app.py")


def _projet(tmp_path: Path) -> Path:
    p = tmp_path / "projects" / "test-vame"
    for sub in ("raw", "cropped", "dlc-output", "vame", "results"):
        (p / "data" / sub).mkdir(parents=True)
    (p / "configs").mkdir(parents=True)
    (p / "configs" / "pipeline_config.yaml").write_text(
        yaml.safe_dump({"kind": "single"}, sort_keys=False), encoding="utf-8",
    )
    session_dir = p / "data" / "raw" / "S1"
    session_dir.mkdir(parents=True)
    (session_dir / "metadata.yaml").write_text(
        yaml.safe_dump({"source_video": str(session_dir / "S1.mp4")}), encoding="utf-8",
    )
    # Étapes précédentes faites (vidéo, pose, nettoyage) : VAME n'est pas
    # verrouillée, ces tests portent sur la logique propre à la page.
    (session_dir / "S1.mp4").write_bytes(b"\x00")
    sortie = p / "data" / "dlc-output" / "S1"
    sortie.mkdir(parents=True)
    (sortie / "S1DLC.h5").write_bytes(b"\x00")
    (sortie / "S1_clean.h5").write_bytes(b"\x00")
    return p


def _init_vame(projet: Path, n_clusters: int = 15) -> Path:
    """Simule un `setup` déjà passé : `data/vame/config.yaml` présent."""
    vame = projet / "data" / "vame"
    (vame / "config.yaml").write_text(
        yaml.safe_dump(
            {"n_clusters": n_clusters, "segmentation_algorithms": ["hmm"]},
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return vame


def _lancer_sur_projet(tmp_path: Path, monkeypatch, projet: Path) -> AppTest:
    monkeypatch.setattr(P, "PREFS_PATH", tmp_path / "app_prefs.yaml")
    P.save_prefs({"projects_root": str(projet.parent), "models_root": str(tmp_path / "models")})
    (tmp_path / "models").mkdir(exist_ok=True)
    at = AppTest.from_file(APP_PY)
    at.session_state["current_project_path"] = str(projet)
    at.run()
    assert not at.exception, at.exception
    boutons = {b.key: b for b in at.button}
    assert "nav_vame" in boutons, list(boutons)
    boutons["nav_vame"].click().run()
    assert not at.exception, at.exception
    return at


def _bouton(at: AppTest, cle: str):
    boutons = {b.key: b for b in at.button}
    return boutons[cle]


def test_projet_sans_vame_seul_setup_actif(tmp_path, monkeypatch):
    """Sur un projet sans dossier VAME, seul `setup` est actionnable ; les
    cinq autres boutons sont grisés, chacun avec sa raison."""
    projet = _projet(tmp_path)
    at = _lancer_sur_projet(tmp_path, monkeypatch, projet)

    assert not _bouton(at, "btn_vame_setup").disabled

    autres = ["btn_vame_align", "btn_vame_trainset", "btn_vame_train",
              "btn_vame_evaluate", "btn_vame_segment"]
    for cle in autres:
        bouton = _bouton(at, cle)
        assert bouton.disabled, cle
        assert bouton.help, f"{cle} grisé sans raison affichée"

    assert "projet VAME" in _bouton(at, "btn_vame_align").help
    assert "Aligne" in _bouton(at, "btn_vame_trainset").help
    assert "jeu d'entraînement" in _bouton(at, "btn_vame_train").help
    assert "modèle entraîné" in _bouton(at, "btn_vame_evaluate").help
    assert "modèle entraîné" in _bouton(at, "btn_vame_segment").help


def test_progression_complete_debloque_toutes_les_etapes(tmp_path, monkeypatch):
    """Reproduit l'état d'avancement complet (setup->train) et vérifie que
    chaque bouton se débloque en conséquence."""
    projet = _projet(tmp_path)
    vame = _init_vame(projet)
    (vame / "data" / "processed").mkdir(parents=True)
    (vame / "data" / "processed" / "S1_processed.nc").write_text("mock")
    (vame / "data" / "train").mkdir(parents=True)
    (vame / "data" / "train" / "train_seq.npy").write_bytes(b"\x00")
    (vame / "model" / "best_model").mkdir(parents=True)
    (vame / "model" / "best_model" / "rnn_vae.pkl").write_text("mock")

    at = _lancer_sur_projet(tmp_path, monkeypatch, projet)

    for cle in ("btn_vame_setup", "btn_vame_align", "btn_vame_trainset",
                "btn_vame_train", "btn_vame_evaluate", "btn_vame_segment"):
        assert not _bouton(at, cle).disabled, cle


def test_train_n_affirme_jamais_que_lentrainement_est_termine(tmp_path, monkeypatch):
    """Le coeur de la consigne d'honnêteté : `train: True` (un .pkl existe)
    ne doit jamais se lire comme « entraînement terminé »."""
    projet = _projet(tmp_path)
    vame = _init_vame(projet)
    (vame / "model" / "best_model").mkdir(parents=True)
    (vame / "model" / "best_model" / "rnn_vae.pkl").write_text("mock")

    at = _lancer_sur_projet(tmp_path, monkeypatch, projet)

    textes = [i.value for i in at.info] + [w.value for w in at.warning] + \
             [c.value for c in at.caption]
    assert not any("entraînement terminé" in t.lower() for t in textes), textes
    # Le message honnête attendu : un modèle existe, mais rien ne garantit
    # que l'entraînement a fini.
    assert any("modèle existe" in t.lower() for t in textes), textes
    assert any("fin du run" in t or "convergence" in t for t in textes), textes


def test_train_sans_modele_pas_de_pretention_non_plus(tmp_path, monkeypatch):
    """État `train: False` : le texte doit rester neutre."""
    projet = _projet(tmp_path)
    _init_vame(projet)

    at = _lancer_sur_projet(tmp_path, monkeypatch, projet)

    textes = [c.value for c in at.caption]
    assert any("Aucun modèle sauvegardé" in t for t in textes), textes


def test_align_ne_pretend_pas_que_toutes_les_sessions_sont_faites(tmp_path, monkeypatch):
    """`align: True` vient d'un `any()` — une seule session traitée suffit.
    Le texte doit le dire, pas laisser entendre que tout est fait."""
    projet = _projet(tmp_path)
    vame = _init_vame(projet)
    (vame / "data" / "processed").mkdir(parents=True)
    (vame / "data" / "processed" / "S1_processed.nc").write_text("mock")
    (vame / "data" / "processed" / ".ethoflow_align_ok").touch()

    at = _lancer_sur_projet(tmp_path, monkeypatch, projet)

    captions = [c.value for c in at.caption]
    assert any("Au moins une session alignée" in c for c in captions), captions
    assert not any("Toutes les sessions" in c and "aligné" in c for c in captions)


def test_evaluate_etat_annonce_comme_non_detectable(tmp_path, monkeypatch):
    """`stage_status` n'a pas de clé `evaluate` : la page ne doit pas
    inventer un état fait/à faire pour cette sous-étape."""
    projet = _projet(tmp_path)
    _init_vame(projet)

    at = _lancer_sur_projet(tmp_path, monkeypatch, projet)

    captions = [c.value for c in at.caption]
    assert any("ne peut pas détecter" in c for c in captions), captions


def test_segment_affiche_n_clusters_courant_et_avertit_sur_le_csv(tmp_path, monkeypatch):
    projet = _projet(tmp_path)
    vame = _init_vame(projet, n_clusters=15)
    (vame / "model" / "best_model").mkdir(parents=True)
    (vame / "model" / "best_model" / "rnn_vae.pkl").write_text("mock")
    (vame / "motif_labels.csv").write_text("motif_id;label\n0;\n", encoding="utf-8")

    at = _lancer_sur_projet(tmp_path, monkeypatch, projet)

    captions = [c.value for c in at.caption]
    assert any("Nombre de motifs actuel" in c and "15" in c for c in captions), captions

    warnings = [w.value for w in at.warning]
    assert any("unique par projet" in w for w in warnings), warnings
    assert any("Sauvegarde-le d'abord" in w for w in warnings), warnings


def test_segment_champ_n_clusters_champ_libre(tmp_path, monkeypatch):
    """Le champ `--n-clusters` reste modifiable même sans CSV existant, et
    l'avertissement CSV n'apparaît pas quand il n'y a rien à perdre."""
    projet = _projet(tmp_path)
    vame = _init_vame(projet, n_clusters=15)
    (vame / "model" / "best_model").mkdir(parents=True)
    (vame / "model" / "best_model" / "rnn_vae.pkl").write_text("mock")

    at = _lancer_sur_projet(tmp_path, monkeypatch, projet)

    champ = [n for n in at.number_input if n.key == "vame_segment_n_clusters"][0]
    assert champ.value == 15
    assert not any("unique par projet" in w.value for w in at.warning)


def test_setup_ne_demande_jamais_rien_depuis_l_app(tmp_path, monkeypatch):
    """`run_vame.py setup` pose désormais quatre questions à l'invite. Un job
    de l'app tourne sans terminal : sans `--no-prompt`, le premier `input()`
    lèverait EOFError. Les hyperparamètres se règlent donc dans la page, et
    la commande porte à la fois leurs valeurs et `--no-prompt`."""
    projet = _projet(tmp_path)
    at = _lancer_sur_projet(tmp_path, monkeypatch, projet)

    cles = {n.key for n in at.number_input}
    for attendu in ("vame_setup_n_clusters", "vame_setup_time_window",
                    "vame_setup_max_epochs", "vame_setup_pose_confidence"):
        assert attendu in cles, sorted(cles)

    from lib import pipeline as PL
    args = PL.vame_stage(projet, "setup", extra=[
        "--n-clusters", "15", "--time-window", "30", "--max-epochs", "500",
        "--pose-confidence", "0.6", "--no-prompt",
    ]).args
    assert "--no-prompt" in args
    assert args.index("setup") < args.index("--n-clusters")


def test_courbes_d_entrainement_affichees(tmp_path, monkeypatch):
    import numpy as np
    projet = _projet(tmp_path)
    vame = _init_vame(projet)
    d = vame / "model" / "model_losses"
    d.mkdir(parents=True)
    np.save(d / "train_losses_VAME.npy", np.linspace(10, 2, 30))
    np.save(d / "test_losses_VAME.npy", np.r_[np.linspace(9, 3, 20), np.full(10, 3.5)])
    np.save(d / "weight_values_VAME.npy", np.minimum(1, np.arange(30) / 10))
    np.save(d / "kl_losses_VAME.npy", np.linspace(1, 0.2, 30))

    at = _lancer_sur_projet(tmp_path, monkeypatch, projet)
    metriques = {m.label: m.value for m in at.metric}
    assert metriques["Epoch"] == "30"
    assert metriques["Meilleur modèle"] == "epoch 20"
    assert metriques["Sans amélioration"] == "10"
    assert len(at.get("vega_lite_chart")) >= 2   # train/test + composante KL


def test_vame_verrouillee_tant_que_le_nettoyage_n_est_pas_fait(tmp_path, monkeypatch):
    """Règle de la refonte : lancement bloqué, raison écrite sous le bouton."""
    projet = _projet(tmp_path)
    (projet / "data" / "dlc-output" / "S1" / "S1_clean.h5").unlink()
    at = _lancer_sur_projet(tmp_path, monkeypatch, projet)
    assert _bouton(at, "btn_vame_setup").disabled
    assert any("Nettoyage" in c.value and "0/1" in c.value for c in at.caption)


def _job_echoue(projet):
    import json
    jobs = projet / ".ethoflow" / "jobs"
    jobs.mkdir(parents=True, exist_ok=True)
    job_id = "20260101-161400-000000-run_vame"
    (jobs / f"{job_id}.json").write_text(json.dumps({
        "job_id": job_id, "script": "run_vame.py", "env": "vame", "label": "VAME train",
        "argv": ["python", "run_vame.py", "train"], "started_at": "2026-01-01T16:02:00",
        "ended_at": "2026-01-01T16:14:00", "returncode": 1, "pid": None,
        "state": "failed", "cancel_requested": False, "owner_pid": None}))
    (jobs / f"{job_id}.log").write_text("torch.OutOfMemoryError: CUDA out of memory.\n")
    return job_id


def test_relancer_verrouille_comme_lancer(tmp_path, monkeypatch):
    projet = _projet(tmp_path)
    job_id = _job_echoue(projet)
    (projet / "data" / "dlc-output" / "S1" / "S1_clean.h5").unlink()
    at = _lancer_sur_projet(tmp_path, monkeypatch, projet)
    assert _bouton(at, f"relancer_{job_id}").disabled


def test_banniere_d_echec_explique_en_clair(tmp_path, monkeypatch):
    import json
    projet = _projet(tmp_path)
    jobs = projet / ".ethoflow" / "jobs"
    jobs.mkdir(parents=True)
    job_id = "20260101-161400-000000-run_vame"
    (jobs / f"{job_id}.json").write_text(json.dumps({
        "job_id": job_id, "script": "run_vame.py", "env": "vame", "label": "VAME train",
        "argv": ["python", "run_vame.py", "train"], "started_at": "2026-01-01T16:02:00",
        "ended_at": "2026-01-01T16:14:00", "returncode": 1, "pid": None,
        "state": "failed", "cancel_requested": False, "owner_pid": None}))
    (jobs / f"{job_id}.log").write_text(
        "Epoch 15/200\ntorch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB\n")

    at = _lancer_sur_projet(tmp_path, monkeypatch, projet)
    texte = " ".join(m.value for m in at.markdown)
    assert "« VAME train » a échoué après 12 min" in texte
    assert "carte graphique" in texte and "taille de lot" in texte
    assert "code de retour 1" in texte
    assert f"relancer_{job_id}" in {b.key for b in at.button}


def test_banniere_seulement_sur_la_page_de_son_etape(tmp_path, monkeypatch):
    projet = _projet(tmp_path)
    job_id = _job_echoue(projet)
    at = _lancer_sur_projet(tmp_path, monkeypatch, projet)
    assert f"relancer_{job_id}" in {b.key for b in at.button}
    at.button(key="nav_pose").click().run()
    assert not at.exception, at.exception
    assert f"relancer_{job_id}" not in {b.key for b in at.button}
