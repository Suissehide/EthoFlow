import subprocess
import sys
from pathlib import Path

import yaml

from lib import project as P


def test_list_projects_ne_retient_que_les_vrais_projets(tmp_path):
    (tmp_path / "vrai" / "data").mkdir(parents=True)
    (tmp_path / "faux").mkdir()
    (tmp_path / "fichier.txt").write_text("x")
    assert [p.name for p in P.list_projects(tmp_path)] == ["vrai"]


def test_list_projects_racine_absente(tmp_path):
    assert P.list_projects(tmp_path / "nexiste-pas") == []


def test_list_dlc_models_veut_un_config_yaml(tmp_path):
    (tmp_path / "modele-a").mkdir()
    (tmp_path / "modele-a" / "config.yaml").write_text("x")
    (tmp_path / "pas-un-modele").mkdir()
    assert [p.name for p in P.list_dlc_models(tmp_path)] == ["modele-a"]


def test_lecture_pipeline_config(project):
    cfg_path = project / "configs" / "pipeline_config.yaml"
    cfg_path.write_text(yaml.safe_dump({
        "kind": "multi",
        "dlc_project_config": "/modeles/souris/config.yaml",
        "px_per_cm": 12.5,
        "default_arenes_coords": {"A1": [0, 0, 512, 540]},
    }))
    assert P.project_kind(project) == "multi"
    assert P.dlc_config_path(project) == "/modeles/souris/config.yaml"
    assert P.px_per_cm(project) == 12.5
    assert P.arena_coords(project) == {"A1": [0, 0, 512, 540]}


def test_excel_path_trouve_le_classeur_du_projet(project):
    """`excel_path()` délègue à `sync_from_excel.find_project_excel` — même
    convention de nommage que `create_project.py` (`<nom_du_projet>_sessions.xlsx`)."""
    cible = project / f"{project.name}_sessions.xlsx"
    cible.write_bytes(b"contenu xlsx factice")
    assert P.excel_path(project) == cible


def test_excel_path_none_sans_classeur(project):
    """Aucun Excel à la racine : `None`, pas d'exception (ruling R11.1)."""
    assert P.excel_path(project) is None


def test_projet_sans_config_ne_leve_pas(tmp_path):
    """Un projet fraîchement créé n'a pas encore de pipeline_config.yaml."""
    vide = tmp_path / "vide"
    (vide / "data").mkdir(parents=True)
    assert P.read_pipeline_config(vide) == {}
    assert P.dlc_config_path(vide) is None
    assert P.px_per_cm(vide) is None
    assert P.arena_coords(vide) == {}
    assert P.project_kind(vide) == "single"


def test_dlc_config_status_absent(project):
    assert P.dlc_config_status(project) == ("absent", None)


def test_dlc_config_status_introuvable(project):
    """Un modèle configuré mais déplacé/supprimé ne doit PAS se présenter
    comme utilisable (ruling R12.1) : `run_dlc_inference.py --mode custom`
    fait le même test d'existence et échoue vite en --no-prompt."""
    P.set_dlc_config(project, "/modeles/disparu/config.yaml")
    statut, chemin = P.dlc_config_status(project)
    assert statut == "introuvable"
    assert chemin == "/modeles/disparu/config.yaml"


def test_dlc_config_status_ok(project, tmp_path):
    modele = tmp_path / "modeles" / "souris" / "config.yaml"
    modele.parent.mkdir(parents=True)
    modele.write_text("dummy: true")
    P.set_dlc_config(project, modele)
    statut, chemin = P.dlc_config_status(project)
    assert statut == "ok"
    assert chemin == str(modele)


def test_set_dlc_config_preserve_le_reste(project):
    """Régression Critical 1+2 : désigner un modèle ne doit PAS repasser par
    create_project.py --force, qui régénère pipeline_config.yaml en entier
    (arènes perdues) et l'Excel de démarrage (données du chercheur perdues).
    set_dlc_config() ne touche qu'à sa propre clé."""
    cfg_path = project / "configs" / "pipeline_config.yaml"
    cfg_path.write_text(yaml.safe_dump({
        "default_arenes_coords": {"A1": [0, 0, 512, 540]},
        "px_per_cm": 12.5,
    }))
    P.set_dlc_config(project, "/modeles/souris/config.yaml")
    cfg = P.read_pipeline_config(project)
    assert cfg["dlc_project_config"] == "/modeles/souris/config.yaml"
    assert cfg["default_arenes_coords"] == {"A1": [0, 0, 512, 540]}
    assert cfg["px_per_cm"] == 12.5


def test_set_dlc_config_deux_fois_ne_duplique_pas(project):
    P.set_dlc_config(project, "/modeles/a/config.yaml")
    P.set_dlc_config(project, "/modeles/b/config.yaml")
    cfg = P.read_pipeline_config(project)
    assert cfg["dlc_project_config"] == "/modeles/b/config.yaml"
    assert list(cfg.keys()).count("dlc_project_config") == 1
    # Pas de bloc dupliqué/imbriqué : une seule paire clé/valeur pour
    # dlc_project_config, le reste (posé par le fixture `project`) intact.
    assert cfg["kind"] == "single"
    assert len(cfg) == 2


def test_set_dlc_config_cree_le_fichier_si_absent(tmp_path):
    projet = tmp_path / "sans-config"
    (projet / "data").mkdir(parents=True)
    assert not (projet / "configs" / "pipeline_config.yaml").exists()
    resultat = P.set_dlc_config(projet, "/modeles/x/config.yaml")
    assert resultat == projet / "configs" / "pipeline_config.yaml"
    assert resultat.is_file()
    assert P.dlc_config_path(projet) == "/modeles/x/config.yaml"


def _creer_projet_reel(tmp_path: Path, nom: str, kind: str) -> Path:
    """Lance le vrai create_project.py — pas un fixture qui invente la
    forme du YAML. Régression Critical 5 : tests/test_project.py écrivait
    à la main `kind: multi`, une clé que create_project.py n'écrit jamais ;
    project_kind() semblait donc marcher alors qu'elle lisait une clé
    fantôme."""
    projet = tmp_path / nom
    resultat = subprocess.run(
        [sys.executable, str(P.SCRIPTS_DIR / "create_project.py"),
         "--project-dir", str(projet), "--kind", kind, "--no-prompt"],
        capture_output=True, text=True,
    )
    assert resultat.returncode == 0, resultat.stderr
    return projet


def test_project_kind_single_sur_sortie_reelle_de_create_project(tmp_path):
    projet = _creer_projet_reel(tmp_path, "proj-single", "single")
    # create_project.py n'écrit aucune clé `kind` pour single (dict vide) —
    # c'est exactement le cas que la lecture littérale de la clé ratait.
    assert P.read_pipeline_config(projet) == {}
    assert P.project_kind(projet) == "single"


def test_project_kind_multi_sur_sortie_reelle_de_create_project(tmp_path):
    projet = _creer_projet_reel(tmp_path, "proj-multi", "multi")
    # Pour multi, seule `default_arenes_coords` est écrite — pas `kind` non
    # plus. C'est elle qui doit faire déduire "multi".
    cfg = P.read_pipeline_config(projet)
    assert "kind" not in cfg
    assert cfg.get("default_arenes_coords")
    assert P.project_kind(projet) == "multi"


def test_project_kind_deduit_depuis_les_arenes_de_session(project, session_factory):
    """Sans clé `kind` explicite ni default_arenes_coords, mais avec des
    sessions déjà synchronisées qui ont des arènes : vérité de terrain
    consultée par crop_arenes.py / assign_arenas.py.

    Le fixture `project` écrit `kind: single` dans pipeline_config.yaml —
    pratique pour les autres tests, mais faux ici : create_project.py
    n'écrit jamais cette clé (Critical 5). On l'enlève pour retomber sur
    la situation réelle : un pipeline_config.yaml sans `kind`."""
    (project / "configs" / "pipeline_config.yaml").unlink()
    assert P.project_kind(project) == "single"
    session_factory("S1", arenes=[{"id": "A1", "coords": [0, 0, 100, 100]}])
    assert P.project_kind(project) == "multi"


def test_prefs_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr(P, "PREFS_PATH", tmp_path / "app_prefs.yaml")
    P.save_prefs({"projects_root": "/data/projets"})
    assert P.load_prefs()["projects_root"] == "/data/projets"
    assert P.projects_root() == Path("/data/projets")


def test_projects_root_defaut_vient_des_scripts(tmp_path, monkeypatch):
    """Sans préférence, on prend la racine que les scripts utilisent."""
    monkeypatch.setattr(P, "PREFS_PATH", tmp_path / "absent.yaml")
    import interactive
    assert P.projects_root() == interactive.DEFAULT_PROJECTS_ROOT
