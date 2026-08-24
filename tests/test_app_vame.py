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

    assert "setup" in _bouton(at, "btn_vame_align").help
    assert "align" in _bouton(at, "btn_vame_trainset").help
    assert "trainset" in _bouton(at, "btn_vame_train").help
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
    assert any("n_clusters` actuel" in c and "15" in c for c in captions), captions

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
