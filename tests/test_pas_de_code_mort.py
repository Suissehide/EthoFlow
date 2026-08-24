"""Garde-fous contre le retour des modèles périmés.

Chacun de ces symboles correspond à un bug corrigé. Les revoir apparaître
signifierait qu'on a réintroduit l'ancien modèle de données.

Deux corrections au jeu de tokens d'origine (ruling P2, Task 23) :

- `"motif_labels_"` est RETIRÉ : `lib/motif_labels.py::legacy_yaml_files`
  contient légitimement le glob `motif_labels_*.yaml` (fichiers YAML par
  algo de l'app v1, qu'on détecte justement pour proposer une migration —
  voir aussi `views/motifs.py`, `views/vame_page.py`, `views/about.py` qui
  en parlent en commentaire/UI). Le garder aurait fait échouer le test sur
  du code que le plan a lui-même demandé.
- `"--no-capture-output"` (présent dans le jeu de tokens du brief) est
  aussi RETIRÉ, pour la même raison structurelle : `lib/pipeline.py`
  documente dans le docstring de `to_argv()` que ce flag n'est
  délibérément PAS utilisé — la phrase explicative contient donc le
  symbole littéral. Un garde-fou qui interdit la chaîne interdit du même
  coup la phrase qui explique pourquoi elle est interdite.
- `"save_labels"` et `"labels_path"` sont AJOUTÉS : les deux noms de
  fonction de l'ancien `lib/labels.py` (format YAML par motif, supprimé),
  qui identifient le format mort sans collision avec du code vivant
  (vérifié par grep avant ajout — aucune occurrence actuelle dans
  `streamlit_app/`).
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
APP = ROOT / "streamlit_app"

INTERDITS = {
    "vame_projects_root": "le layout VAME est plat dans le projet",
    "discover_projects": "il n'y a qu'un projet VAME, rien à découvrir",
    "DEFAULT_VAME_PROJECTS_ROOT": "racine VAME externe supprimée",
    ".vame_config_path": "l'app ne lit plus le pointeur de config VAME",
    "ANGII": "l'Excel est à colonnes libres",
    "save_labels": "ancien lib/labels.py supprimé — labels dans motif_labels.csv",
    "labels_path": "ancien lib/labels.py supprimé — labels dans motif_labels.csv",
}


def _sources():
    return [p for p in APP.rglob("*.py") if "__pycache__" not in p.parts]


def test_aucun_symbole_perime():
    fautes = []
    for chemin in _sources():
        texte = chemin.read_text(encoding="utf-8")
        for symbole, raison in INTERDITS.items():
            if symbole in texte:
                fautes.append(f"{chemin.relative_to(ROOT)} : {symbole} — {raison}")
    assert not fautes, "\n".join(fautes)


def test_seul_config_importe_streamlit_dans_lib():
    """`lib/` doit rester testable sans Streamlit lancé."""
    fautes = []
    for chemin in (APP / "lib").glob("*.py"):
        if chemin.name in ("config.py", "__init__.py"):
            continue
        if "import streamlit" in chemin.read_text(encoding="utf-8"):
            fautes.append(chemin.name)
    assert not fautes, f"modules lib/ important streamlit : {fautes}"


def test_tous_les_scripts_appeles_sont_dans_la_table_des_envs():
    import sys
    sys.path.insert(0, str(APP))
    from lib.pipeline import SCRIPT_ENVS
    for script in SCRIPT_ENVS:
        assert (ROOT / "scripts" / script).is_file(), f"scripts/{script} manquant"
