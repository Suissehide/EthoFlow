"""Vérification de bout en bout de la page Projet via AppTest.

Isolation : `lib.project.PREFS_PATH` est monkeypatché comme dans
`tests/test_project.py::test_prefs_round_trip` — jamais de dépendance à
`Path.home()` réel ni à `DEFAULT_PROJECTS_ROOT` (`D:\\EthoFlow\\projects`,
un chemin Windows qui n'est pas absolu du tout sur ce runner macOS/Linux :
un harnais qui ne l'isole pas crée un dossier littéral `D:\\EthoFlow\\
projects` dans le dépôt — piège déjà tombé sur une revue précédente).

`lib.runner._argv` est monkeypatché pour lancer `create_project.py`
directement via `sys.executable` plutôt que par un `conda run -n ethoflow`
imbriqué (on tourne déjà dans cet env) : le job réel devient quasi
instantané, donc ce test reste rapide malgré un vrai script + un vrai
sous-process.
"""
from __future__ import annotations

import sys
from pathlib import Path

import yaml
from streamlit.testing.v1 import AppTest

from lib import pipeline as PL
from lib import project as P
from lib import runner as R

REPO = Path(__file__).resolve().parent.parent
APP_PY = str(REPO / "streamlit_app" / "app.py")


def _argv_sans_conda(cmd: PL.Command) -> list[str]:
    """`conda run -n <env> python <script> <args>` -> `[sys.executable, script, *args]`."""
    argv = PL.to_argv(cmd)
    return [sys.executable] + argv[5:]


def _attendre_job_termine(project: Path, timeout: float = 20.0) -> None:
    import time
    fin = time.time() + timeout
    while time.time() < fin:
        job = R.current(project)
        if job and job.state != "running":
            return
        time.sleep(0.05)
    raise AssertionError("job de création toujours 'running' après timeout")


def test_creer_supprimer_retaper_ne_ressuscite_pas_le_projet(tmp_path, monkeypatch):
    """Séquence exacte du Critical remonté par la revue : créer X (bannière
    de succès), le supprimer via le flux guardé de la page, retaper le même
    nom X. Avant le ruling R10.6 : la bannière "créé avec succès" revenait,
    le bouton "Créer le projet" disparaissait pour de bon, et le dossier
    revenait comme un squelette vide sur disque."""
    monkeypatch.setattr(P, "PREFS_PATH", tmp_path / "app_prefs.yaml")
    monkeypatch.setattr(R, "_argv", _argv_sans_conda)

    projects_root = tmp_path / "projects"
    models_root = tmp_path / "models"
    projects_root.mkdir()
    models_root.mkdir()
    P.save_prefs({"projects_root": str(projects_root), "models_root": str(models_root)})

    nom = "test-zombie"
    cible = projects_root / nom

    # ---- 1) Créer X : le projet s'ouvre tout seul à la fin ----
    at = AppTest.from_file(APP_PY)
    at.run()
    assert not at.exception, at.exception
    at.text_input(key="creation_nom").set_value(nom).run()
    at.button(key="btn_creer").click().run()
    assert not at.exception, at.exception

    _attendre_job_termine(cible.parent)
    # Le clic relance le script (carte de création à la place du formulaire) ;
    # l'arbre d'AppTest garde pourtant les anciens champs, dont l'état a
    # été nettoyé — on les remet pour que `run()` puisse les relire.
    at.session_state["creation_kind"] = "single"
    at.session_state["creation_modele"] = "(choisir plus tard)"
    at.run()
    assert not at.exception, at.exception
    assert (cible / "configs" / "pipeline_config.yaml").is_file()
    assert Path(at.session_state["current_project_path"]).name == nom
    assert at.session_state["current_page"] == "Projet"   # ouvert directement

    # ---- 2) Revenir aux projets, supprimer X (menu ⋯ -> Supprimer -> confirmer) ----
    at.button(key="nav_projets").click().run()
    supprimer = [b for b in at.button if b.key.startswith("supprimer_")]
    assert supprimer, [b.key for b in at.button]
    supprimer[0].click().run()
    at.button(key="btn_confirmer_suppression").click().run()
    assert not at.exception, at.exception
    assert not cible.exists(), "le rmtree n'a pas eu lieu"

    # ---- 3) Retaper le même nom ----
    at.text_input(key="creation_nom").set_value(nom).run()
    assert not at.exception, at.exception
    # Le job de création réussi (toujours le dernier de la racine) ne doit
    # pas être repris pour ce nouveau dossier : pas d'ouverture fantôme…
    assert at.session_state["current_page"] == "Projets"
    # …et le bouton de création reste là, utilisable.
    bouton = at.button(key="btn_creer")
    assert not bouton.disabled, bouton.help
    assert not cible.exists(), "le dossier a été ressuscité rien qu'en retapant le nom"
