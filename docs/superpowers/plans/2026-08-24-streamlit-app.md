# App Streamlit EthoFlow — Plan d'implémentation

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Faire de `streamlit_app/` la porte d'entrée des étapes 1 à 9 du README, en remplaçant une couche d'exécution qui ne peut rien lancer et un modèle de données périmé.

**Architecture:** Une couche `lib/` sans Streamlit qui contient toute la logique (chemins, construction des commandes, runner de jobs, lecture des sorties VAME) et qui est testée ; des `views/` minces qui ne font que placer des widgets. Les scripts CLI ne sont pas modifiés et restent la seule autorité sur les formats de fichiers.

**Tech Stack:** Python 3.10, Streamlit ≥ 1.30, pandas, PyYAML, OpenCV, `streamlit-image-coordinates`, pytest, `conda run` pour l'exécution inter-environnements.

**Spec:** `docs/superpowers/specs/2026-08-24-streamlit-app-design.md`

## Global Constraints

- **Toute commande construite pour un script projet-aware porte `--project-dir <chemin absolu>` et `--no-prompt`.** Sans ça le script appelle `input()` et le subprocess se fige (spec §5.1).
- **`conda run` est invoqué SANS `--no-capture-output`.** Ce flag renvoie la sortie au terminal au lieu du pipe (spec §4, constat n°2).
- **`PYTHONUNBUFFERED=1` dans l'environnement de tout subprocess**, sinon les logs arrivent par blocs de 8 ko (spec §15).
- **Env conda par script** — table de la spec §5.2, à respecter exactement :
  - `ethoflow` : `create_project`, `excel_templates`, `sync_from_excel`, `crop_arenes`, `assign_arenas`, `inspect_session`, `diagnose_dlc_model`, `motif_gif`, `post_process_cropped`, `filter_keypoints`, `fill_nan_h5`, `trim_empty_arena`, `rekey_h5`
  - `dlc` : `run_dlc_inference`, `prepare_vame_input_custom`
  - `vame` : `run_vame`, `analyze_vame`, `behavior_structure_gif`, `community_dendrogram`, `inspect_vame_project`, `reencode_vame_videos`
- **Aucun module `lib/` autre que `lib/config.py` et `views/*` n'importe `streamlit`.** C'est ce qui rend `lib/` testable.
- **L'app n'écrit jamais un format de fichier défini par un script.** Elle importe la fonction d'écriture du script (`calibrate_arenes.save_coords_default`, `calibrate_scale.write_scale`) ou lance le script.
- **Le layout VAME est plat** : `<projet>/data/vame/` *est* le projet VAME. Pas de découverte, pas de sélection.
- **`motif_labels.csv`** : séparateur `;`, encodage `utf-8-sig`, colonnes de `run_vame.MOTIF_LABELS_COLUMNS`, colonnes supplémentaires de l'utilisateur préservées.
- **Coordonnées d'arène** : `dict[str, list[int]]`, clés `A1`…`A4`, valeur `[x, y, w, h]`.
- **Langue de l'interface : français**, comme l'existant et le README.
- Commande de test : `conda activate ethoflow && pytest tests/ -v`

---

# Phase 1 — Socle

Rien de visible pendant cette phase, mais toutes les pages en dépendent. Une page écrite avant serait à réécrire.

---

### Task 1: Infrastructure de test

`tests/` ne contient qu'un squelette qui teste l'arborescence du dépôt, et il n'y a ni `conftest.py` ni configuration pytest. Impossible d'importer `streamlit_app/lib/` depuis un test. Cette tâche pose les fondations dont les 22 suivantes se servent.

**Files:**
- Create: `tests/conftest.py`
- Create: `pytest.ini`
- Modify: `tests/test_skeleton.py`

**Interfaces:**
- Consumes: rien
- Produces: fixtures `project` (arborescence de projet factice), `session_factory` (crée une session avec metadata), `vame_project` (projet VAME segmenté factice). Toutes les tâches suivantes les utilisent.

- [ ] **Step 1: Créer la configuration pytest**

`pytest.ini` à la racine du dépôt :

```ini
[pytest]
testpaths = tests
python_files = test_*.py
addopts = -q
```

- [ ] **Step 2: Écrire les fixtures**

`tests/conftest.py` :

```python
"""Fixtures partagées : arborescences de projet factices.

Les tests de `streamlit_app/lib/` doivent tourner sans conda, sans GPU et
sans données réelles. On fabrique donc des projets EthoFlow minimaux mais
structurellement exacts.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent

# `lib.*` et `paths` doivent être importables comme dans l'app.
for extra in (ROOT / "streamlit_app", ROOT / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))


@pytest.fixture
def project(tmp_path: Path) -> Path:
    """Projet EthoFlow vide mais complet : dossiers + pipeline_config.yaml."""
    p = tmp_path / "projet-test"
    for sub in ("raw", "cropped", "dlc-output", "vame", "results"):
        (p / "data" / sub).mkdir(parents=True)
    (p / "configs").mkdir(parents=True)
    (p / "configs" / "pipeline_config.yaml").write_text(
        yaml.safe_dump({"kind": "single"}, sort_keys=False),
        encoding="utf-8",
    )
    return p


@pytest.fixture
def session_factory(project: Path):
    """Fabrique des sessions : metadata.yaml + vidéo source factice."""
    def _make(session_id: str, *, video: bool = True, **meta_extra) -> Path:
        sdir = project / "data" / "raw" / session_id
        sdir.mkdir(parents=True, exist_ok=True)
        video_path = project / "videos" / f"{session_id}.mp4"
        if video:
            video_path.parent.mkdir(parents=True, exist_ok=True)
            video_path.write_bytes(b"\x00")
        meta = {"id": session_id, "source_video": str(video_path)}
        meta.update(meta_extra)
        (sdir / "metadata.yaml").write_text(
            yaml.safe_dump(meta, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        return sdir
    return _make


@pytest.fixture
def vame_project(project: Path) -> Path:
    """Projet VAME segmenté factice à <projet>/data/vame/ (layout plat)."""
    import numpy as np

    vame = project / "data" / "vame"
    vame.mkdir(parents=True, exist_ok=True)
    (vame / "config.yaml").write_text(
        yaml.safe_dump(
            {"n_clusters": 15, "segmentation_algorithms": ["hmm"]},
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    algo_dir = vame / "results" / "S1" / "VAME" / "hmm-15"
    algo_dir.mkdir(parents=True)
    np.save(algo_dir / "motif_usage_S1.npy", np.arange(15, dtype=float))
    (algo_dir / "15_hmm_label_S1.npy").write_bytes(b"\x00")
    return vame
```

- [ ] **Step 3: Corriger le test d'arborescence obsolète**

`test_data_directories_exist` vérifie `<repo>/data/` — legacy d'avant les projets auto-suffisants, et `configs/pipeline_config.yaml.example` n'a plus lieu d'être testé à la racine. Remplacer ces deux tests dans `tests/test_skeleton.py` :

```python
def test_scripts_partages_presents():
    """Les modules partagés entre CLI et app doivent exister."""
    for s in ("paths.py", "interactive.py", "run_vame.py", "analyze_vame.py"):
        assert (ROOT / "scripts" / s).is_file(), f"scripts/{s} manquant"


def test_streamlit_lib_importable():
    """lib/ doit s'importer sans Streamlit lancé."""
    import lib.project  # noqa: F401
```

Supprimer `test_data_directories_exist` et `test_configs_present`.

- [ ] **Step 4: Vérifier que la suite échoue pour la bonne raison**

Run: `pytest tests/ -v`
Expected: `test_streamlit_lib_importable` FAIL avec `ModuleNotFoundError: No module named 'lib.project'`. Les autres tests PASS. C'est le point de départ de la Task 2.

- [ ] **Step 5: Commit**

```bash
git add pytest.ini tests/conftest.py tests/test_skeleton.py
git commit -m "tests: fixtures de projet factice + config pytest"
```

---

### Task 2: `lib/project.py` — chemins et préférences, sans Streamlit

`lib/config.py` mélange aujourd'hui trois choses : des chemins purs, la lecture de `st.session_state`, et un vocabulaire éthologique. Le tout importe `streamlit` au niveau module, donc rien n'est testable. On extrait la partie pure.

**Files:**
- Create: `streamlit_app/lib/project.py`
- Create: `tests/test_project.py`

**Interfaces:**
- Consumes: `scripts/paths.py`, `scripts/interactive.py` (constantes de racines)
- Produces:
  - `prefs_path() -> Path`, `load_prefs() -> dict`, `save_prefs(dict) -> None`
  - `projects_root() -> Path`, `models_root() -> Path`
  - `list_projects(root: Path) -> list[Path]`
  - `list_dlc_models(root: Path) -> list[Path]`
  - `read_pipeline_config(project: Path) -> dict`
  - `dlc_config_path(project: Path) -> str | None`
  - `project_kind(project: Path) -> str`
  - `px_per_cm(project: Path) -> float | None`
  - `arena_coords(project: Path) -> dict[str, list[int]]`

- [ ] **Step 1: Écrire les tests qui échouent**

`tests/test_project.py` :

```python
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


def test_projet_sans_config_ne_leve_pas(tmp_path):
    """Un projet fraîchement créé n'a pas encore de pipeline_config.yaml."""
    vide = tmp_path / "vide"
    (vide / "data").mkdir(parents=True)
    assert P.read_pipeline_config(vide) == {}
    assert P.dlc_config_path(vide) is None
    assert P.px_per_cm(vide) is None
    assert P.arena_coords(vide) == {}
    assert P.project_kind(vide) == "single"


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
```

- [ ] **Step 2: Lancer pour vérifier l'échec**

Run: `pytest tests/test_project.py -v`
Expected: FAIL avec `ModuleNotFoundError: No module named 'lib.project'`

- [ ] **Step 3: Implémenter**

`streamlit_app/lib/project.py` :

```python
"""Chemins projet et préférences d'interface — sans Streamlit.

Ce module ne connaît que des `Path`. Toute la logique testable de
localisation vit ici ; `lib/config.py` se contente d'y brancher le projet
courant lu dans le `session_state`.

La résolution structurelle (`<projet>/data/raw/` etc.) vient de
`scripts/paths.py`, source unique de vérité partagée avec les CLI.
"""
from __future__ import annotations

import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPTS_DIR = ROOT / "scripts"

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import paths as _paths  # noqa: E402
from interactive import (  # noqa: E402
    DEFAULT_MODELS_ROOT,
    DEFAULT_PROJECTS_ROOT,
)

# Préférences d'interface uniquement (racines, dernier projet ouvert).
# Jamais lues par les scripts CLI.
PREFS_PATH = Path.home() / ".ethoflow" / "app_prefs.yaml"


# ---------------------------------------------------------------- préférences

def prefs_path() -> Path:
    return PREFS_PATH


def load_prefs() -> dict:
    if not PREFS_PATH.exists():
        return {}
    try:
        return yaml.safe_load(PREFS_PATH.read_text(encoding="utf-8")) or {}
    except Exception:
        # Un fichier de préférences corrompu ne doit pas empêcher de
        # démarrer l'app : on repart des défauts.
        return {}


def save_prefs(prefs: dict) -> None:
    PREFS_PATH.parent.mkdir(parents=True, exist_ok=True)
    PREFS_PATH.write_text(
        yaml.safe_dump(prefs, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def projects_root() -> Path:
    return Path(load_prefs().get("projects_root", DEFAULT_PROJECTS_ROOT))


def models_root() -> Path:
    return Path(load_prefs().get("models_root", DEFAULT_MODELS_ROOT))


# ------------------------------------------------------------------ inventaire

def list_projects(root: Path) -> list[Path]:
    """Dossiers de `root` qui ressemblent à un projet EthoFlow."""
    root = Path(root)
    if not root.is_dir():
        return []
    return sorted(d for d in root.iterdir() if d.is_dir() and (d / "data").is_dir())


def list_dlc_models(root: Path) -> list[Path]:
    """Dossiers de `root` contenant un `config.yaml` DLC."""
    root = Path(root)
    if not root.is_dir():
        return []
    return sorted(
        d for d in root.iterdir() if d.is_dir() and (d / "config.yaml").is_file()
    )


# --------------------------------------------------------- pipeline_config.yaml

def read_pipeline_config(project: Path) -> dict:
    """Contenu de `configs/pipeline_config.yaml`, `{}` s'il n'existe pas."""
    cfg = _paths.pipeline_config_path(Path(project))
    if not cfg.exists():
        return {}
    try:
        return yaml.safe_load(cfg.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}


def dlc_config_path(project: Path) -> str | None:
    value = read_pipeline_config(project).get("dlc_project_config")
    return str(value) if value else None


def project_kind(project: Path) -> str:
    """'single' ou 'multi'. 'single' par défaut : pas d'arena splitting."""
    kind = read_pipeline_config(project).get("kind")
    return kind if kind in ("single", "multi") else "single"


def px_per_cm(project: Path) -> float | None:
    value = read_pipeline_config(project).get("px_per_cm")
    return float(value) if value is not None else None


def arena_coords(project: Path) -> dict[str, list[int]]:
    return read_pipeline_config(project).get("default_arenes_coords") or {}
```

- [ ] **Step 4: Lancer les tests**

Run: `pytest tests/test_project.py tests/test_skeleton.py -v`
Expected: PASS, y compris `test_streamlit_lib_importable` de la Task 1.

- [ ] **Step 5: Commit**

```bash
git add streamlit_app/lib/project.py tests/test_project.py
git commit -m "lib: module project.py sans Streamlit (chemins + preferences)"
```

---

### Task 3: `lib/pipeline.py` — construction des commandes

Le cœur de la correction. Aujourd'hui `lib/pipeline.py` exécute directement, sans `--project-dir`, avec un `--no-capture-output` qui vide `stdout`. On le remplace par des constructeurs d'argv purs, sans exécution — donc testables sans conda.

**Files:**
- Modify: `streamlit_app/lib/pipeline.py` (réécriture complète)
- Create: `tests/test_pipeline.py`

**Interfaces:**
- Consumes: `lib/project.SCRIPTS_DIR`
- Produces:
  - `SCRIPT_ENVS: dict[str, str]`
  - `@dataclass(frozen=True) Command(env: str, script: str, args: list[str], label: str)`
  - `to_argv(cmd: Command) -> list[str]` — argv complet `conda run …`
  - un constructeur par script, tous retournant `Command` : `create_project`, `sync_from_excel`, `crop_arenes`, `run_dlc_inference`, `diagnose_dlc_model`, `prepare_vame_input`, `assign_arenas`, `inspect_session`, `vame_stage`, `analyze_vame`, `motif_gif`, `behavior_structure_gif`, `community_dendrogram`

- [ ] **Step 1: Écrire les tests qui échouent**

`tests/test_pipeline.py` :

```python
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
```

- [ ] **Step 2: Lancer pour vérifier l'échec**

Run: `pytest tests/test_pipeline.py -v`
Expected: FAIL — `AttributeError: module 'lib.pipeline' has no attribute 'SCRIPT_ENVS'`

- [ ] **Step 3: Implémenter**

Réécrire `streamlit_app/lib/pipeline.py`. La structure, avec les constructeurs complets :

```python
"""Construction des commandes du pipeline — sans exécution.

Chaque fonction publique retourne un `Command` décrivant quoi lancer et
dans quel env conda. C'est `lib/runner.py` qui exécute. Séparer les deux
rend la construction testable sans conda, sans GPU et sans données.

Trois règles valent pour toute commande (voir la spec §5.1) :
  1. `--project-dir` toujours passé, sinon le script demande le projet à
     l'invite et le subprocess se fige.
  2. `--no-prompt` toujours passé, pour un échec franc au lieu d'une
     attente silencieuse.
  3. Tout paramètre que le script demanderait est fourni explicitement.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from lib.project import SCRIPTS_DIR

# Mapping script -> env conda. Voir spec §5.2. Se tromper d'env produit un
# ImportError après plusieurs minutes d'attente.
SCRIPT_ENVS: dict[str, str] = {
    # env ethoflow : pandas / yaml / openpyxl / cv2
    "create_project.py": "ethoflow",
    "excel_templates.py": "ethoflow",
    "sync_from_excel.py": "ethoflow",
    "crop_arenes.py": "ethoflow",
    "assign_arenas.py": "ethoflow",
    "inspect_session.py": "ethoflow",
    "diagnose_dlc_model.py": "ethoflow",
    "motif_gif.py": "ethoflow",
    "post_process_cropped.py": "ethoflow",
    "filter_keypoints.py": "ethoflow",
    "fill_nan_h5.py": "ethoflow",
    "trim_empty_arena.py": "ethoflow",
    "rekey_h5.py": "ethoflow",
    # env dlc : importe deeplabcut
    "run_dlc_inference.py": "dlc",
    "prepare_vame_input_custom.py": "dlc",   # dlc.filterpredictions
    # env vame : vame, matplotlib, scipy, umap, sklearn
    "run_vame.py": "vame",
    "analyze_vame.py": "vame",               # matplotlib + scipy
    "behavior_structure_gif.py": "vame",     # umap + sklearn
    "community_dendrogram.py": "vame",       # scipy
    "inspect_vame_project.py": "vame",
    "reencode_vame_videos.py": "vame",
}


@dataclass(frozen=True)
class Command:
    env: str
    script: str
    args: list[str] = field(default_factory=list)
    label: str = ""


def to_argv(cmd: Command) -> list[str]:
    """argv complet pour `subprocess.Popen`.

    Pas de `--no-capture-output` : ce flag renvoie la sortie au terminal
    au lieu du pipe, et c'est précisément le bug de l'ancienne version.
    """
    env = SCRIPT_ENVS[cmd.script]   # KeyError volontaire si script inconnu
    return [
        "conda", "run", "-n", env,
        "python", str(SCRIPTS_DIR / cmd.script),
        *cmd.args,
    ]


def _base(project: Path) -> list[str]:
    return ["--project-dir", str(Path(project).resolve()), "--no-prompt"]


def _cmd(script: str, args: list[str], label: str) -> Command:
    return Command(env=SCRIPT_ENVS[script], script=script, args=args, label=label)
```

Puis les constructeurs. `create_project` est le seul à ne pas utiliser `_base` : le projet n'existe pas encore, `--project-dir` désigne la cible à créer.

```python
def create_project(project: Path, *, kind: str,
                   dlc_config: str | None = None,
                   force: bool = False) -> Command:
    args = ["--project-dir", str(Path(project).resolve()), "--kind", kind]
    if dlc_config:
        args += ["--dlc-config", str(dlc_config)]
    if force:
        args.append("--force")
    args.append("--no-prompt")
    return _cmd("create_project.py", args, f"Créer le projet {Path(project).name}")


def sync_from_excel(project: Path, *, videos_dir: str | Path,
                    excel: str | Path | None = None,
                    video_ext: str = "mp4",
                    overwrite: bool = False,
                    dry_run: bool = False) -> Command:
    args = _base(project) + ["--videos-dir", str(videos_dir), "--video-ext", video_ext]
    if excel:
        args += ["--excel", str(excel)]
    if overwrite:
        args.append("--overwrite")
    if dry_run:
        args.append("--dry-run")
    return _cmd("sync_from_excel.py", args,
                "Aperçu du sync" if dry_run else "Sync depuis Excel")


def crop_arenes(project: Path, *, sessions: list[str] | None = None,
                all_sessions: bool = False, all_new: bool = False) -> Command:
    args = _base(project)
    if all_sessions:
        args.append("--all")
    elif all_new:
        args.append("--all-new")
    else:
        args += list(sessions or [])
    return _cmd("crop_arenes.py", args, "Crop des arènes")


def run_dlc_inference(project: Path, *, mode: str,
                      sessions: list[str] | None = None,
                      all_sessions: bool = False,
                      skip_existing: bool = True,
                      video_adapt: bool = False,
                      video_adapt_batch_size: int = 2) -> Command:
    args = _base(project)
    if all_sessions:
        args.append("--all")
    else:
        args += list(sessions or [])
    args += ["--mode", mode]
    if skip_existing:
        args.append("--skip-existing")
    if video_adapt:
        args += ["--video-adapt",
                 "--video-adapt-batch-size", str(video_adapt_batch_size)]
    return _cmd("run_dlc_inference.py", args, f"Inférence DLC ({mode})")


def diagnose_dlc_model(project: Path, *, model_dir: str | Path | None = None,
                       fix: bool = False) -> Command:
    args = _base(project)
    if model_dir:
        args += ["--model-dir", str(model_dir)]
    if fix:
        args.append("--fix")
    return _cmd("diagnose_dlc_model.py", args,
                "Réparer le modèle DLC" if fix else "Diagnostiquer le modèle DLC")


def prepare_vame_input(project: Path, *, likelihood_threshold: float,
                       max_speed: float,
                       px_per_cm: float | None = None,
                       sessions: list[str] | None = None,
                       sticky_detection: bool = True,
                       qc_plot: bool = True,
                       qc_bodypart: str = "tail_base",
                       interp_limit: int = 25,
                       window_length: int = 5,
                       skip_existing: bool = False) -> Command:
    args = _base(project) + list(sessions or [])
    args += ["--likelihood-threshold", str(likelihood_threshold),
             "--max-speed", str(max_speed),
             "--interp-limit", str(interp_limit),
             "--window-length", str(window_length),
             "--qc-bodypart", qc_bodypart]
    if px_per_cm is not None:
        args += ["--px-per-cm", str(px_per_cm)]
    if not sticky_detection:
        args.append("--no-sticky-detection")
    if not qc_plot:
        args.append("--no-qc-plot")
    if skip_existing:
        args.append("--skip-existing")
    return _cmd("prepare_vame_input_custom.py", args, "Nettoyage des poses")


def assign_arenas(project: Path, *, sessions: list[str] | None = None,
                  all_sessions: bool = False, all_new: bool = False,
                  likelihood_threshold: float = 0.6,
                  interp_limit: int = 25, clean: bool = True) -> Command:
    args = _base(project)
    if all_sessions:
        args.append("--all")
    elif all_new:
        args.append("--all-new")
    else:
        args += list(sessions or [])
    args += ["--likelihood-threshold", str(likelihood_threshold),
             "--interp-limit", str(interp_limit)]
    if not clean:
        args.append("--no-clean")
    return _cmd("assign_arenas.py", args, "Split par arène")


def inspect_session(project: Path, *, sessions: list[str] | None = None,
                    all_sessions: bool = False,
                    input_dir: str | Path | None = None,
                    fps: float | None = None) -> Command:
    args = _base(project)
    if all_sessions or not sessions:
        args.append("--all")
    else:
        args += list(sessions)
    if input_dir:
        args += ["--input-dir", str(input_dir)]
    if fps is not None:
        args += ["--fps", str(fps)]
    return _cmd("inspect_session.py", args, "Inspection qualité")


def vame_stage(project: Path, stage: str, *,
               n_clusters: int | None = None,
               regen_labels: bool = False,
               extra: list[str] | None = None) -> Command:
    """`--project-dir` doit précéder la sous-commande (contrainte argparse)."""
    args = _base(project) + [stage]
    if stage == "segment" and n_clusters is not None:
        args += ["--n-clusters", str(n_clusters)]
    if stage in ("motif-videos", "motif-labels") and regen_labels:
        args.append("--regen-labels")
    args += list(extra or [])
    return _cmd("run_vame.py", args, f"VAME {stage}")


def analyze_vame(project: Path, *, algo: str = "hmm",
                 n_clusters: int | None = None,
                 labels: str | Path | None = None,
                 group_by: list[str] | None = None,
                 cross: list[tuple[str, str]] | None = None,
                 extended: bool = False,
                 extended_by: str | None = None,
                 mask_empty: bool = False,
                 validity_source: str | Path | None = None,
                 min_edge_frames: int = 25,
                 fps: float = 30.0,
                 list_columns: bool = False) -> Command:
    args = _base(project)
    if list_columns:
        # Sort la liste des axes et rend la main : tout autre flag serait
        # ignoré, autant ne pas les construire.
        return _cmd("analyze_vame.py", args + ["--list-columns"],
                    "Axes de comparaison disponibles")
    args += ["--algo", algo, "--min-edge-frames", str(min_edge_frames),
             "--fps", str(fps)]
    if n_clusters is not None:
        args += ["--n-clusters", str(n_clusters)]
    if labels:
        args += ["--labels", str(labels)]
    if group_by:
        args += ["--group-by", *group_by]
    for pair in (cross or []):
        args += ["--cross", pair[0], pair[1]]
    if extended:
        args.append("--extended")
        if extended_by:
            args += ["--extended-by", extended_by]
    if validity_source:
        args += ["--validity-source", str(validity_source)]
    if mask_empty:
        args.append("--mask-empty")
    return _cmd("analyze_vame.py", args, "Analyses statistiques")


def motif_gif(project: Path, *, session: str, algo: str = "hmm",
              start: float = 0.0, duration: float | None = None,
              output_format: str = "mp4",
              labels: str | Path | None = None) -> Command:
    args = _base(project) + ["--session", session, "--algo", algo,
                             "--start", str(start),
                             "--output-format", output_format]
    if duration is not None:
        args += ["--duration", str(duration)]
    if labels:
        args += ["--labels", str(labels)]
    return _cmd("motif_gif.py", args, f"Bande de motifs — {session}")


def behavior_structure_gif(project: Path, *, session: str, algo: str = "hmm",
                           projection: str = "umap",
                           start: float = 0.0,
                           duration: float | None = None,
                           output_format: str = "gif",
                           with_video: bool = False,
                           pool_all_sessions: bool = False,
                           labels: str | Path | None = None) -> Command:
    args = _base(project) + ["--session", session, "--algo", algo,
                             "--projection", projection,
                             "--start", str(start),
                             "--output-format", output_format]
    if duration is not None:
        args += ["--duration", str(duration)]
    if with_video:
        args.append("--with-video")
    if pool_all_sessions:
        args.append("--pool-all-sessions")
    if labels:
        args += ["--labels", str(labels)]
    return _cmd("behavior_structure_gif.py", args, f"Manifold — {session}")


def community_dendrogram(project: Path, *, algo: str = "hmm",
                         group: str | None = None,
                         linkage: str = "ward",
                         labels: str | Path | None = None) -> Command:
    args = _base(project) + ["--algo", algo, "--linkage", linkage]
    if group:
        args += ["--group", group]
    if labels:
        args += ["--labels", str(labels)]
    return _cmd("community_dendrogram.py", args, "Dendrogramme des communautés")
```

- [ ] **Step 4: Lancer les tests**

Run: `pytest tests/test_pipeline.py -v`
Expected: PASS (16 tests)

- [ ] **Step 5: Commit**

```bash
git add streamlit_app/lib/pipeline.py tests/test_pipeline.py
git commit -m "lib: pipeline.py construit des argv testables, avec --project-dir partout"
```

---

### Task 4: `lib/runner.py` — exécution des jobs

Un job doit survivre à la navigation entre pages, au rafraîchissement du navigateur et à la fermeture de l'onglet. Un `session_state` Streamlit ne survit à aucun des trois. L'état vit donc sur disque.

**Files:**
- Create: `streamlit_app/lib/runner.py`
- Create: `tests/test_runner.py`

**Interfaces:**
- Consumes: `lib.pipeline.Command`, `lib.pipeline.to_argv`
- Produces:
  - `@dataclass Job` avec `job_id, script, env, label, argv, started_at, ended_at, returncode, pid, state`
  - `class JobBusy(RuntimeError)`
  - `jobs_dir(project) -> Path`, `start(project, cmd) -> Job`, `current(project) -> Job | None`
  - `history(project, limit=20) -> list[Job]`, `read_log(project, job_id, tail=None) -> str`
  - `cancel(project, job_id) -> None`, `is_running(project) -> bool`

- [ ] **Step 1: Écrire les tests qui échouent**

`tests/test_runner.py`. On teste la machine à états avec de vraies commandes triviales, pas avec conda.

```python
import time

import pytest

from lib import pipeline as PL
from lib import runner as R


def _echo(msg: str = "bonjour") -> PL.Command:
    """Command factice court-circuitant conda, pour tester la machine à états."""
    return PL.Command(env="ethoflow", script="__test__",
                      args=["python", "-c", f"print({msg!r})"], label="echo")


@pytest.fixture(autouse=True)
def _argv_direct(monkeypatch):
    """`__test__` n'est pas dans SCRIPT_ENVS : on court-circuite to_argv."""
    monkeypatch.setattr(
        R, "_argv", lambda cmd: cmd.args if cmd.script == "__test__" else PL.to_argv(cmd)
    )


def _attendre_fin(project, timeout=15.0):
    fin = time.time() + timeout
    while time.time() < fin:
        job = R.current(project)
        if job and job.state != "running":
            return job
        time.sleep(0.05)
    raise AssertionError("job toujours running après timeout")


def test_job_reussi(project):
    R.start(project, _echo("salut"))
    job = _attendre_fin(project)
    assert job.state == "succeeded"
    assert job.returncode == 0
    assert "salut" in R.read_log(project, job.job_id)


def test_job_echoue(project):
    R.start(project, PL.Command("ethoflow", "__test__",
                                ["python", "-c", "raise SystemExit(3)"], "boom"))
    job = _attendre_fin(project)
    assert job.state == "failed"
    assert job.returncode == 3


def test_un_seul_job_a_la_fois(project):
    R.start(project, PL.Command("ethoflow", "__test__",
                                ["python", "-c", "import time; time.sleep(5)"], "long"))
    with pytest.raises(R.JobBusy):
        R.start(project, _echo())
    R.cancel(project, R.current(project).job_id)
    _attendre_fin(project)


def test_annulation(project):
    R.start(project, PL.Command("ethoflow", "__test__",
                                ["python", "-c", "import time; time.sleep(30)"], "long"))
    R.cancel(project, R.current(project).job_id)
    job = _attendre_fin(project)
    assert job.state == "cancelled"


def test_verrou_libere_apres_la_fin(project):
    R.start(project, _echo())
    _attendre_fin(project)
    assert not R.is_running(project)
    R.start(project, _echo("deuxieme"))     # ne doit pas lever
    _attendre_fin(project)


def test_job_running_dont_le_process_a_disparu(project):
    """App tuée pendant un job : le JSON dit running, le pid n'existe plus."""
    R.start(project, _echo())
    job = _attendre_fin(project)
    R._write_job(project, R.replace(job, state="running", pid=999999, ended_at=None))
    releve = R.current(project)
    assert releve.state == "interrupted"


def test_historique_du_plus_recent_au_plus_ancien(project):
    for i in range(3):
        R.start(project, _echo(f"job{i}"))
        _attendre_fin(project)
    labels = [j.label for j in R.history(project)]
    assert len(labels) == 3
    assert R.history(project, limit=2) == R.history(project)[:2]


def test_read_log_tail(project):
    R.start(project, PL.Command(
        "ethoflow", "__test__",
        ["python", "-c", "[print(i) for i in range(100)]"], "cent"))
    job = _attendre_fin(project)
    assert R.read_log(project, job.job_id, tail=5).strip().split("\n") == \
        ["95", "96", "97", "98", "99"]


def test_pythonunbuffered_dans_lenvironnement(project, monkeypatch):
    captured = {}
    vrai_popen = R.subprocess.Popen

    def espion(argv, **kw):
        captured.update(kw.get("env") or {})
        return vrai_popen(argv, **kw)

    monkeypatch.setattr(R.subprocess, "Popen", espion)
    R.start(project, _echo())
    _attendre_fin(project)
    assert captured.get("PYTHONUNBUFFERED") == "1"
```

- [ ] **Step 2: Lancer pour vérifier l'échec**

Run: `pytest tests/test_runner.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'lib.runner'`

- [ ] **Step 3: Implémenter**

`streamlit_app/lib/runner.py` :

```python
"""Exécution des scripts en tâche de fond, avec état persisté sur disque.

Un `session_state` Streamlit ne survit ni à la navigation, ni au
rafraîchissement du navigateur, ni à la fermeture de l'onglet. L'état d'un
job vit donc dans `<projet>/.ethoflow/jobs/` :

    <job_id>.json   métadonnées et état
    <job_id>.log    stdout + stderr fusionnés, écrits au fil de l'eau
    current.lock    job_id du job en cours, s'il y en a un

Un seul job à la fois par projet : DLC et VAME veulent tous les deux le
GPU, les laisser tourner ensemble est une source de plantage.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import threading
from dataclasses import asdict, dataclass, replace as _dc_replace
from datetime import datetime
from pathlib import Path

from lib.pipeline import Command, to_argv

replace = _dc_replace


class JobBusy(RuntimeError):
    """Un job tourne déjà pour ce projet."""


@dataclass(frozen=True)
class Job:
    job_id: str
    script: str
    env: str
    label: str
    argv: list[str]
    started_at: str
    ended_at: str | None = None
    returncode: int | None = None
    pid: int | None = None
    state: str = "running"     # running|succeeded|failed|cancelled|interrupted


# ------------------------------------------------------------------- chemins

def jobs_dir(project: Path) -> Path:
    d = Path(project) / ".ethoflow" / "jobs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _job_json(project: Path, job_id: str) -> Path:
    return jobs_dir(project) / f"{job_id}.json"


def _job_log(project: Path, job_id: str) -> Path:
    return jobs_dir(project) / f"{job_id}.log"


def _lock(project: Path) -> Path:
    return jobs_dir(project) / "current.lock"


def _write_job(project: Path, job: Job) -> None:
    _job_json(project, job.job_id).write_text(
        json.dumps(asdict(job), indent=2), encoding="utf-8")


def _read_job(project: Path, job_id: str) -> Job | None:
    path = _job_json(project, job_id)
    if not path.exists():
        return None
    try:
        return Job(**json.loads(path.read_text(encoding="utf-8")))
    except Exception:
        return None


def _argv(cmd: Command) -> list[str]:
    """Indirection pour que les tests puissent court-circuiter conda."""
    return to_argv(cmd)


def _pid_vivant(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except (OSError, ProcessLookupError):
        return False
    return True


def _reconcilier(project: Path, job: Job) -> Job:
    """Un job `running` dont le process a disparu = app tuée en cours de route."""
    if job.state != "running" or _pid_vivant(job.pid):
        return job
    job = replace(job, state="interrupted",
                  ended_at=datetime.now().isoformat(timespec="seconds"))
    _write_job(project, job)
    _lock(project).unlink(missing_ok=True)
    return job


# ------------------------------------------------------------------ lecture

def current(project: Path) -> Job | None:
    lock = _lock(project)
    if lock.exists():
        job = _read_job(project, lock.read_text(encoding="utf-8").strip())
        if job:
            return _reconcilier(project, job)
    hist = history(project, limit=1)
    return hist[0] if hist else None


def is_running(project: Path) -> bool:
    job = current(project)
    return bool(job and job.state == "running")


def history(project: Path, limit: int = 20) -> list[Job]:
    """Du plus récent au plus ancien. Les job_id sont horodatés donc triables."""
    jobs = []
    for path in sorted(jobs_dir(project).glob("*.json"), reverse=True):
        job = _read_job(project, path.stem)
        if job:
            jobs.append(_reconcilier(project, job))
    return jobs[:limit]


def read_log(project: Path, job_id: str, tail: int | None = None) -> str:
    path = _job_log(project, job_id)
    if not path.exists():
        return ""
    texte = path.read_text(encoding="utf-8", errors="replace")
    if tail is None:
        return texte
    return "\n".join(texte.splitlines()[-tail:])


# ---------------------------------------------------------------- exécution

def start(project: Path, cmd: Command) -> Job:
    project = Path(project)
    if is_running(project):
        en_cours = current(project)
        raise JobBusy(
            f"« {en_cours.label} » tourne déjà (démarré à {en_cours.started_at}). "
            "Attends la fin ou annule-le."
        )

    horodatage = datetime.now().strftime("%Y%m%d-%H%M%S")
    job_id = f"{horodatage}-{Path(cmd.script).stem}"
    argv = _argv(cmd)
    log_path = _job_log(project, job_id)

    env = dict(os.environ)
    # Sans ça Python tamponne sa sortie quand elle n'est pas un terminal,
    # et les logs arrivent par blocs au lieu d'arriver ligne à ligne.
    env["PYTHONUNBUFFERED"] = "1"

    log_file = open(log_path, "w", encoding="utf-8", buffering=1)
    proc = subprocess.Popen(
        argv,
        stdout=log_file, stderr=subprocess.STDOUT,
        env=env, cwd=str(project), text=True,
        start_new_session=True,   # groupe de process propre, pour l'annulation
    )

    job = Job(
        job_id=job_id, script=cmd.script, env=cmd.env, label=cmd.label,
        argv=argv, started_at=datetime.now().isoformat(timespec="seconds"),
        pid=proc.pid, state="running",
    )
    _write_job(project, job)
    _lock(project).write_text(job_id, encoding="utf-8")

    def _surveiller() -> None:
        code = proc.wait()
        log_file.close()
        courant = _read_job(project, job_id) or job
        if courant.state == "cancelled":
            etat = "cancelled"
        else:
            etat = "succeeded" if code == 0 else "failed"
        _write_job(project, replace(
            courant, state=etat, returncode=code,
            ended_at=datetime.now().isoformat(timespec="seconds"),
        ))
        _lock(project).unlink(missing_ok=True)

    threading.Thread(target=_surveiller, daemon=True).start()
    return job


def cancel(project: Path, job_id: str) -> None:
    """Tue le groupe de process.

    Le process visible est `conda run`, pas le Python qui travaille : il
    faut descendre au groupe entier, sinon l'enfant survit à son parent.
    """
    job = _read_job(project, job_id)
    if not job or job.state != "running" or not job.pid:
        return
    _write_job(project, replace(job, state="cancelled"))
    try:
        os.killpg(os.getpgid(job.pid), signal.SIGTERM)
    except (OSError, ProcessLookupError):
        pass
```

- [ ] **Step 4: Lancer les tests**

Run: `pytest tests/test_runner.py -v`
Expected: PASS (9 tests). S'ils sont lents, c'est normal : ils lancent de vrais process courts.

- [ ] **Step 5: Commit**

```bash
git add streamlit_app/lib/runner.py tests/test_runner.py
git commit -m "lib: runner de jobs avec etat persiste, verrou projet et annulation"
```

---

### Task 5: `lib/vame.py` — lecture des sorties VAME re-racinées

`lib/vame_projects.py` scanne `~/Inserm/vame-projects` pour trouver des projets VAME. Le layout est plat depuis la fusion : `<projet>/data/vame/` *est* le projet VAME. La logique de détection des algos et d'agrégation des `.npy` est correcte et se garde ; seule sa racine change.

**Files:**
- Create: `streamlit_app/lib/vame.py`
- Delete: `streamlit_app/lib/vame_projects.py`
- Create: `tests/test_vame.py`

**Interfaces:**
- Consumes: `paths.vame_dir`
- Produces:
  - `vame_project(project) -> Path`, `is_initialised(project) -> bool`
  - `read_config(project) -> dict`, `n_clusters(project) -> int | None`
  - `list_algos(project) -> list[str]`, `parse_algo_n(algo) -> tuple[str, int]`
  - `list_sessions(project) -> list[str]`, `motif_usage_df(project, algo) -> pd.DataFrame`
  - `analysis_dir(project) -> Path`, `stage_status(project) -> dict[str, bool]`

- [ ] **Step 1: Écrire les tests qui échouent**

`tests/test_vame.py` :

```python
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


def test_stage_status_train(project, vame_project):
    (vame_project / "model" / "best_model").mkdir(parents=True)
    assert V.stage_status(project)["train"] is True
```

- [ ] **Step 2: Lancer pour vérifier l'échec**

Run: `pytest tests/test_vame.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'lib.vame'`

- [ ] **Step 3: Implémenter**

`streamlit_app/lib/vame.py`. Reprendre `list_algos`, `parse_algo_n`, `motif_usage_df` de `lib/vame_projects.py:56-157` en remplaçant le paramètre « projet VAME » par « projet EthoFlow » et en dérivant la racine :

```python
"""Lecture des sorties VAME du projet courant.

Layout plat : `<projet>/data/vame/` EST le projet VAME. Il y en a un par
projet EthoFlow — rien à découvrir, rien à sélectionner.

    data/vame/config.yaml
    data/vame/motif_labels.csv
    data/vame/model/
    data/vame/results/<session>/<model>/<algo>-<n>/
    data/vame/analysis/
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from lib.project import SCRIPTS_DIR

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
from paths import vame_dir  # noqa: E402

_ALGO_RE = re.compile(r"^(hmm|kmeans)-(\d+)$")


def vame_project(project: Path) -> Path:
    return vame_dir(Path(project))


def is_initialised(project: Path) -> bool:
    return (vame_project(project) / "config.yaml").is_file()


def read_config(project: Path) -> dict:
    cfg = vame_project(project) / "config.yaml"
    if not cfg.exists():
        return {}
    try:
        return yaml.safe_load(cfg.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}


def n_clusters(project: Path) -> int | None:
    cfg = read_config(project)
    n = cfg.get("n_clusters", cfg.get("n_cluster"))
    return int(n) if n else None


def analysis_dir(project: Path) -> Path:
    return vame_project(project) / "analysis"


def list_sessions(project: Path) -> list[str]:
    results = vame_project(project) / "results"
    if not results.is_dir():
        return []
    return sorted(d.name for d in results.iterdir()
                  if d.is_dir() and not d.name.startswith("."))


def list_algos(project: Path) -> list[str]:
    """Cherche `results/<session>/<model>/<algo>-<n>/`."""
    results = vame_project(project) / "results"
    if not results.is_dir():
        return []
    trouves: set[str] = set()
    for session_dir in results.iterdir():
        if not session_dir.is_dir():
            continue
        for model_dir in session_dir.iterdir():
            if not model_dir.is_dir():
                continue
            for algo_dir in model_dir.iterdir():
                if algo_dir.is_dir() and _ALGO_RE.match(algo_dir.name):
                    trouves.add(algo_dir.name)
    return sorted(trouves)


def parse_algo_n(algo: str) -> tuple[str, int]:
    m = _ALGO_RE.match(algo)
    if not m:
        raise ValueError(f"Format d'algo invalide : {algo!r} (attendu 'hmm-N')")
    return m.group(1), int(m.group(2))


def motif_usage_df(project: Path, algo: str) -> pd.DataFrame:
    """Agrège les `motif_usage_*.npy` en table longue."""
    colonnes = ["session", "motif", "count", "frequency"]
    results = vame_project(project) / "results"
    if not results.is_dir():
        return pd.DataFrame(columns=colonnes)
    lignes: list[dict] = []
    for npy in sorted(results.glob(f"*/*/{algo}/motif_usage_*.npy")):
        session = npy.parents[2].name
        try:
            arr = np.asarray(np.load(npy), dtype=float).ravel()
        except Exception:
            continue
        total = float(arr.sum()) or 1.0
        for motif_id, count in enumerate(arr):
            lignes.append({"session": session, "motif": int(motif_id),
                           "count": float(count), "frequency": float(count) / total})
    return pd.DataFrame(lignes, columns=colonnes)


def session_has_labels(project: Path, session: str) -> bool:
    results = vame_project(project) / "results" / session
    return results.is_dir() and any(results.glob("*/*/*_label_*.npy"))


def stage_status(project: Path) -> dict[str, bool]:
    """Ce qui est déjà fait, pour le stepper de la page VAME."""
    vp = vame_project(project)
    return {
        "setup": is_initialised(project),
        "align": any(vp.glob("data/*/*-PE-seq.npy")),
        "trainset": (vp / "data" / "train" / "train_seq.npy").exists(),
        "train": (vp / "model").is_dir() and any((vp / "model").iterdir()),
        "segment": any(vp.glob("results/*/*/*/*_label_*.npy")),
    }
```

- [ ] **Step 4: Lancer les tests**

Run: `pytest tests/test_vame.py -v`
Expected: PASS (9 tests)

- [ ] **Step 5: Supprimer l'ancien module**

```bash
git rm streamlit_app/lib/vame_projects.py
```

`views/results.py`, `views/label_motifs.py` et `views/run_pipeline.py` l'importent encore — ils seront réécrits aux Tasks 12 à 17. Pour que la suite reste verte entre-temps, ils sont retirés de la navigation à la Task 13.

- [ ] **Step 6: Commit**

```bash
git add streamlit_app/lib/vame.py tests/test_vame.py
git commit -m "lib: vame.py sur le layout plat <projet>/data/vame/"
```

---

### Task 6: `lib/motif_labels.py` — le vrai format de labels

La page de labellisation écrit `analysis/motif_labels_<algo>.yaml`. Ce format n'est lu par rien : `analyze_vame.py:1230` cherche `<vame>/motif_labels.csv`. Tout le travail d'annotation fait dans l'app actuelle est invisible aux analyses.

**Files:**
- Create: `streamlit_app/lib/motif_labels.py`
- Delete: `streamlit_app/lib/labels.py`
- Create: `tests/test_motif_labels.py`

**Interfaces:**
- Consumes: `lib.vame.vame_project`, `run_vame.MOTIF_LABELS_COLUMNS`, `run_vame.ETHOGRAM_CATEGORIES`
- Produces:
  - `COLUMNS: list[str]`, `categories() -> list[str]`
  - `path(project) -> Path`, `exists(project) -> bool`
  - `load(project) -> pd.DataFrame | None`, `save(project, df) -> None`
  - `set_fields(project, motif_id, **champs) -> None`
  - `legacy_yaml_files(project) -> list[Path]`, `migrate_from_yaml(project, yaml_path) -> int`
  - `video_path(project, row) -> Path | None`

- [ ] **Step 1: Écrire les tests qui échouent**

`tests/test_motif_labels.py` :

```python
import pandas as pd
import yaml

from lib import motif_labels as ML


def _csv_factice(vame_project, lignes=3):
    contenu = "motif_id;label;category;confidence;qc_inspected_sessions;notes;usage_pct;video\n"
    for i in range(lignes):
        contenu += f"{i};;;;;;{10.0 - i:.2f};results/community_videos/motif_{i}.mp4\n"
    (vame_project / "motif_labels.csv").write_text(contenu, encoding="utf-8-sig")


def test_colonnes_viennent_des_scripts():
    """Une copie divergerait au premier changement de run_vame.py."""
    import run_vame
    assert ML.COLUMNS == run_vame.MOTIF_LABELS_COLUMNS
    assert ML.categories() == run_vame.ETHOGRAM_CATEGORIES
    assert "Transitions" in ML.categories()
    assert len(ML.categories()) == 8


def test_absent_retourne_none(project, vame_project):
    assert not ML.exists(project)
    assert ML.load(project) is None


def test_chargement(project, vame_project):
    _csv_factice(vame_project)
    df = ML.load(project)
    assert list(df.columns) == ML.COLUMNS
    assert len(df) == 3
    assert df.loc[0, "usage_pct"] == "10.00"


def test_ecriture_conserve_separateur_et_encodage(project, vame_project):
    _csv_factice(vame_project)
    ML.set_fields(project, 0, label="grooming_face", category="Grooming")
    brut = (vame_project / "motif_labels.csv").read_bytes()
    assert brut.startswith(b"\xef\xbb\xbf")            # BOM utf-8-sig
    assert b";" in brut.split(b"\n")[0]
    assert b"," not in brut.split(b"\n")[0]
    assert ML.load(project).loc[0, "label"] == "grooming_face"


def test_colonne_utilisateur_preservee(project, vame_project):
    """Quelqu'un ajoute une colonne dans Excel : elle ne doit pas disparaître."""
    (vame_project / "motif_labels.csv").write_text(
        "motif_id;label;category;confidence;qc_inspected_sessions;notes;"
        "usage_pct;video;observateur\n"
        "0;;;;;;10.00;v0.mp4;Leo\n",
        encoding="utf-8-sig",
    )
    ML.set_fields(project, 0, label="walking")
    df = ML.load(project)
    assert df.loc[0, "observateur"] == "Leo"
    assert df.loc[0, "label"] == "walking"
    assert list(df.columns)[-1] == "observateur"


def test_set_fields_ne_touche_pas_les_autres_lignes(project, vame_project):
    _csv_factice(vame_project)
    ML.set_fields(project, 1, category="Locomotion")
    df = ML.load(project)
    assert df.loc[1, "category"] == "Locomotion"
    assert df.loc[0, "category"] == ""
    assert df.loc[2, "usage_pct"] == "8.00"


def test_reprise_depuis_ancien_yaml(project, vame_project):
    _csv_factice(vame_project)
    ancien = vame_project / "analysis" / "motif_labels_hmm-15.yaml"
    ancien.parent.mkdir(parents=True, exist_ok=True)
    ancien.write_text(yaml.safe_dump({0: "grooming", 2: "walking"}))
    assert ML.migrate_from_yaml(project, ancien) == 2
    df = ML.load(project)
    assert df.loc[0, "label"] == "grooming"
    assert df.loc[2, "label"] == "walking"
    assert df.loc[1, "label"] == ""


def test_detection_des_anciens_yaml(project, vame_project):
    (vame_project / "analysis").mkdir(parents=True, exist_ok=True)
    (vame_project / "analysis" / "motif_labels_hmm-15.yaml").write_text("{}")
    assert [p.name for p in ML.legacy_yaml_files(project)] == \
        ["motif_labels_hmm-15.yaml"]


def test_video_path_resolu_depuis_la_colonne(project, vame_project):
    _csv_factice(vame_project)
    clip = vame_project / "results" / "community_videos" / "motif_0.mp4"
    clip.parent.mkdir(parents=True, exist_ok=True)
    clip.write_bytes(b"\x00")
    df = ML.load(project)
    assert ML.video_path(project, df.loc[0]) == clip
    assert ML.video_path(project, df.loc[1]) is None    # fichier absent
```

- [ ] **Step 2: Lancer pour vérifier l'échec**

Run: `pytest tests/test_motif_labels.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'lib.motif_labels'`

- [ ] **Step 3: Implémenter**

`streamlit_app/lib/motif_labels.py` :

```python
"""Accès unique à `<projet>/data/vame/motif_labels.csv`.

Format défini par `run_vame.py` : séparateur `;`, encodage `utf-8-sig`
(Excel sur Windows ouvre correctement les accents), 8 colonnes.

L'app ne fabrique pas ce fichier à partir de rien — c'est
`run_vame motif-videos` ou `run_vame motif-labels` qui le génère avec
`usage_pct` et `video` pré-remplis. L'app met à jour des lignes
existantes, et reprend un ancien YAML si l'utilisateur le demande.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import yaml

from lib.project import SCRIPTS_DIR
from lib.vame import vame_project

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
import run_vame as _run_vame  # noqa: E402

SEP = ";"
ENCODING = "utf-8-sig"

# Source de vérité côté scripts — une copie divergerait au premier changement.
COLUMNS: list[str] = list(_run_vame.MOTIF_LABELS_COLUMNS)


def categories() -> list[str]:
    """Liste fermée écrite dans la colonne `category`, groupée par les analyses."""
    return list(_run_vame.ETHOGRAM_CATEGORIES)


def path(project: Path) -> Path:
    return vame_project(project) / "motif_labels.csv"


def exists(project: Path) -> bool:
    return path(project).is_file()


def load(project: Path) -> pd.DataFrame | None:
    p = path(project)
    if not p.is_file():
        return None
    # dtype=str + keep_default_na : une cellule vide reste "" et non NaN,
    # sinon on réécrirait "nan" dans le fichier de l'utilisateur.
    return pd.read_csv(p, sep=SEP, dtype=str, keep_default_na=False,
                       encoding=ENCODING)


def save(project: Path, df: pd.DataFrame) -> None:
    df.to_csv(path(project), sep=SEP, index=False, encoding=ENCODING)


def set_fields(project: Path, motif_id: int, **champs: str) -> None:
    """Met à jour une ligne. Les colonnes inconnues du CSV sont ignorées."""
    df = load(project)
    if df is None:
        return
    masque = df["motif_id"].astype(str) == str(motif_id)
    for colonne, valeur in champs.items():
        if colonne in df.columns:
            df.loc[masque, colonne] = "" if valeur is None else str(valeur)
    save(project, df)


def video_path(project: Path, row) -> Path | None:
    """Chemin absolu du clip, ou None s'il n'existe pas.

    La colonne `video` contient un chemin relatif au projet VAME, déjà
    résolu par `run_vame.find_motif_video()` — inutile de le deviner.
    """
    rel = str(row.get("video") or "").strip()
    if not rel:
        return None
    candidat = vame_project(project) / rel
    return candidat if candidat.is_file() else None


# ------------------------------------------------------- reprise ancien format

def legacy_yaml_files(project: Path) -> list[Path]:
    """Anciens `analysis/motif_labels_<algo>.yaml` écrits par l'app v1."""
    analysis = vame_project(project) / "analysis"
    if not analysis.is_dir():
        return []
    return sorted(analysis.glob("motif_labels_*.yaml"))


def migrate_from_yaml(project: Path, yaml_path: Path) -> int:
    """Recopie les labels d'un ancien YAML dans le CSV. Retourne le nombre repris.

    N'écrase que la colonne `label`, et seulement pour les motifs présents
    dans le YAML.
    """
    df = load(project)
    if df is None:
        return 0
    brut = yaml.safe_load(Path(yaml_path).read_text(encoding="utf-8")) or {}
    repris = 0
    for cle, valeur in brut.items():
        texte = str(valeur or "").strip()
        if not texte:
            continue
        masque = df["motif_id"].astype(str) == str(cle)
        if masque.any():
            df.loc[masque, "label"] = texte
            repris += 1
    save(project, df)
    return repris
```

- [ ] **Step 4: Lancer les tests**

Run: `pytest tests/test_motif_labels.py -v`
Expected: PASS (10 tests)

- [ ] **Step 5: Supprimer l'ancien module**

```bash
git rm streamlit_app/lib/labels.py
```

- [ ] **Step 6: Commit**

```bash
git add streamlit_app/lib/motif_labels.py tests/test_motif_labels.py
git commit -m "lib: motif_labels.csv remplace le YAML par algo, colonnes libres preservees"
```

---

### Task 7: `lib/sessions.py` — metadata génériques et statuts corrects

Deux défauts : des colonnes `Stress`/`ANGII` codées en dur alors que l'Excel est à colonnes libres (`lib/sessions.py:109-110`), et un statut VAME qui teste `vame_dir()/<session>`, chemin qui n'existe dans aucun layout (`lib/sessions.py:95`).

**Files:**
- Modify: `streamlit_app/lib/sessions.py` (réécriture)
- Create: `tests/test_sessions.py`

**Interfaces:**
- Consumes: `lib.project`, `lib.vame.session_has_labels`, `paths.*`
- Produces:
  - `load_metadata(project, session_id) -> dict | None`
  - `list_sessions(project) -> pd.DataFrame` (colonnes : `session_id, vidéo, DLC, split, nettoyage, VAME`)
  - `arenes_dataframe(meta) -> pd.DataFrame`
  - `metadata_fields(meta) -> dict` — toutes les clés scalaires, dans l'ordre du fichier
  - `session_ids(project) -> list[str]`

- [ ] **Step 1: Écrire les tests qui échouent**

`tests/test_sessions.py` :

```python
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
    assert list(champs) == ["id", "group", "regime_alimentaire", "operateur"]


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
```

- [ ] **Step 2: Lancer pour vérifier l'échec**

Run: `pytest tests/test_sessions.py -v`
Expected: FAIL — `TypeError: list_sessions() takes 0 positional arguments but 1 was given`

- [ ] **Step 3: Implémenter**

Réécrire `streamlit_app/lib/sessions.py`. Points essentiels :

```python
"""Inventaire des sessions et de leur avancement — sans Streamlit.

Aucune colonne de metadata n'est connue à l'avance : `sync_from_excel.py`
recopie *toutes* les colonnes de l'Excel, y compris celles que
l'utilisateur invente. Une app qui n'afficherait que des colonnes connues
casserait cette promesse.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import yaml

from lib.project import SCRIPTS_DIR
from lib.vame import session_has_labels

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
from paths import cleaned_h5_path, dlc_output_dir, raw_dir  # noqa: E402

# Clés de metadata qui ne sont pas des facteurs expérimentaux.
_NON_FACTEURS = {"source_video", "arenes", "camera", "video"}


def session_ids(project: Path) -> list[str]:
    rd = raw_dir(Path(project))
    if not rd.is_dir():
        return []
    return sorted(d.name for d in rd.iterdir()
                  if d.is_dir() and not d.name.startswith("."))


def load_metadata(project: Path, session_id: str) -> dict | None:
    p = raw_dir(Path(project)) / session_id / "metadata.yaml"
    if not p.is_file():
        return None
    try:
        return yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except Exception:
        return None


def metadata_fields(meta: dict | None) -> dict:
    """Clés scalaires de la metadata, dans l'ordre du fichier.

    Exclut les chemins et les structures imbriquées : ce qui reste est
    exactement ce qui peut servir d'axe de comparaison dans analyze_vame.
    """
    if not meta:
        return {}
    return {
        k: v for k, v in meta.items()
        if k not in _NON_FACTEURS and not isinstance(v, (dict, list))
    }


def _dlc_ok(project: Path, session_id: str) -> bool:
    """Un dossier vide ne compte pas : il faut un .h5 produit."""
    d = dlc_output_dir(Path(project)) / session_id
    return d.is_dir() and any(d.glob("*.h5"))


def _split_ok(project: Path, session_id: str) -> bool:
    d = dlc_output_dir(Path(project)) / session_id
    return d.is_dir() and any(d.glob(f"{session_id}_A*.h5"))


def list_sessions(project: Path) -> pd.DataFrame:
    project = Path(project)
    colonnes = ["session_id", "vidéo", "DLC", "split", "nettoyage", "VAME"]
    lignes: list[dict] = []
    for session_id in session_ids(project):
        meta = load_metadata(project, session_id) or {}
        source = meta.get("source_video")
        lignes.append({
            "session_id": session_id,
            "vidéo": "OK" if source and Path(source).exists() else "manque",
            "DLC": "OK" if _dlc_ok(project, session_id) else "—",
            "split": "OK" if _split_ok(project, session_id) else "—",
            "nettoyage": "OK" if cleaned_h5_path(project, session_id).exists() else "—",
            "VAME": "OK" if session_has_labels(project, session_id) else "—",
        })
    return pd.DataFrame(lignes, columns=colonnes)


def arenes_dataframe(meta: dict | None) -> pd.DataFrame:
    """Toutes les clés présentes dans les arènes, sans en présupposer aucune."""
    arenes = (meta or {}).get("arenes") or []
    if not arenes:
        return pd.DataFrame()
    colonnes: list[str] = []
    for arene in arenes:
        for cle in arene:
            if cle not in colonnes:
                colonnes.append(cle)
    lignes = []
    for arene in arenes:
        ligne = {}
        for cle in colonnes:
            valeur = arene.get(cle)
            if cle == "coords":
                ligne[cle] = str(valeur) if valeur else "(à définir)"
            else:
                ligne[cle] = "" if valeur is None else valeur
        lignes.append(ligne)
    return pd.DataFrame(lignes, columns=colonnes)
```

- [ ] **Step 4: Lancer les tests**

Run: `pytest tests/test_sessions.py -v`
Expected: PASS (8 tests)

- [ ] **Step 5: Commit**

```bash
git add streamlit_app/lib/sessions.py tests/test_sessions.py
git commit -m "lib: sessions.py sans colonnes en dur, statut VAME sur le vrai artefact"
```

---

### Task 8: `lib/config.py` — couche Streamlit mince

`lib/config.py` fait aujourd'hui trois métiers. Il n'en garde qu'un : brancher le projet courant du `session_state` sur `lib/project.py`, plus héberger le vocabulaire suggéré.

**Files:**
- Modify: `streamlit_app/lib/config.py` (réécriture)
- Create: `tests/test_ethogram.py`

**Interfaces:**
- Consumes: `lib.project`, `lib.motif_labels.categories`
- Produces:
  - `current_project() -> Path | None`, `current_project_name() -> str | None`
  - `set_current_project(path) -> None`, `require_project() -> Path`
  - `VOCABULAIRE_SUGGERE: dict[str, list[str]]`
  - ré-export `categories()`

- [ ] **Step 1: Écrire le test de cohérence de l'ETHOGRAM**

Deux référentiels divergeaient : 9 catégories dans `lib/config.py:149-185`, 8 dans `run_vame.py:640-644`. `tests/test_ethogram.py` :

```python
from lib import config as C
from lib import motif_labels as ML


def test_categories_viennent_des_scripts():
    """`category` est une liste fermée, groupée par les analyses."""
    assert C.categories() == ML.categories()
    assert "Arena-specific" not in C.categories()   # n'existait que côté app
    assert "Catch-all" not in C.categories()


def test_vocabulaire_suggere_est_libre():
    """Le vocabulaire aide à remplir `label`, qui lui est libre."""
    voc = C.VOCABULAIRE_SUGGERE
    assert "grooming face" in voc["Grooming"]
    assert "thigmotaxis" in voc["Arena-specific"]


def test_vocabulaire_et_categories_sont_deux_choses():
    """Aucun test d'égalité entre les deux : ce sont des rôles distincts."""
    assert set(C.VOCABULAIRE_SUGGERE) != set(C.categories())
```

- [ ] **Step 2: Lancer pour vérifier l'échec**

Run: `pytest tests/test_ethogram.py -v`
Expected: FAIL — `AttributeError: module 'lib.config' has no attribute 'categories'`

- [ ] **Step 3: Implémenter**

Réécrire `streamlit_app/lib/config.py` :

```python
"""Branchement du projet courant sur `lib/project.py`, et vocabulaire.

Seul module de `lib/` autorisé à importer Streamlit : il lit le projet
courant dans le `session_state`. Toute la logique testable vit dans
`lib/project.py`.
"""
from __future__ import annotations

from pathlib import Path

import streamlit as st

from lib.motif_labels import categories  # noqa: F401  (ré-export)
from lib.project import (  # noqa: F401  (ré-exports pour les vues)
    SCRIPTS_DIR,
    arena_coords,
    dlc_config_path,
    list_dlc_models,
    list_projects,
    load_prefs,
    models_root,
    project_kind,
    projects_root,
    px_per_cm,
    read_pipeline_config,
    save_prefs,
)

_CLE = "current_project_path"


def current_project() -> Path | None:
    valeur = st.session_state.get(_CLE)
    return Path(valeur) if valeur else None


def current_project_name() -> str | None:
    projet = current_project()
    return projet.name if projet else None


def set_current_project(path: Path | str | None) -> None:
    if path is None:
        st.session_state.pop(_CLE, None)
        return
    st.session_state[_CLE] = str(Path(path))
    prefs = load_prefs()
    prefs["last_project"] = str(Path(path))
    save_prefs(prefs)


def require_project() -> Path:
    """À appeler en tête de toute vue qui a besoin d'un projet."""
    projet = current_project()
    if projet is None:
        st.warning("Ouvre un projet depuis la page **Projet**.")
        st.stop()
    return projet


# Exemples pour aider à remplir le champ `label`, qui est libre. À ne pas
# confondre avec `categories()` : liste fermée écrite dans `category` et
# utilisée par les analyses pour grouper.
VOCABULAIRE_SUGGERE: dict[str, list[str]] = {
    "Locomotion": [
        "locomotion", "slow locomotion", "fast locomotion", "running",
        "pivoting", "turning", "walking", "trotting", "darting", "circling",
    ],
    "Stationary": [
        "immobility", "freezing", "resting", "crouching",
        "alert immobility", "rest immobility", "vigilance posture", "pause",
    ],
    "Vertical exploration": [
        "rearing supported", "rearing unsupported", "stretch-attend posture",
        "SAP", "half-rear", "elongated stretch",
    ],
    "Sniffing": [
        "sniffing wall", "sniffing floor", "sniffing air", "sniffing (general)",
    ],
    "Grooming": [
        "grooming face", "grooming body", "grooming tail",
        "grooming genital", "scratching", "paw licking",
    ],
    "Exploration": [
        "exploration", "exploration (active)", "exploration (slow)",
        "novelty investigation", "approach", "inspection",
        "head scanning", "wall-following", "nose-poking",
    ],
    "Arena-specific": [
        "thigmotaxis", "center exploration", "corner",
        "transition wall→center", "transition center→wall",
    ],
    "Specific behaviors": [
        "jumping", "digging", "wall climbing",
        "body shake", "arched back", "hunched posture",
    ],
    "Catch-all": [
        "transition", "ambiguous", "artifact", "immobility (imputed)",
    ],
}
```

Les fonctions `data_root()`, `raw_dir()`, `cropped_dir()`, `dlc_output_dir()`, `vame_dir()`, `cleaned_h5_path()`, `results_dir()`, `pipeline_config_path()` sont supprimées : les vues appellent désormais `paths.<fn>(projet)` avec le projet explicite. Les alias figés `DATA_ROOT`, `RAW_DIR`, `CROPPED_DIR`, `DLC_OUTPUT_DIR`, `VAME_DIR` (`lib/config.py:96-100`) et `create_project()` (`lib/config.py:125-131`) disparaissent aussi. `DEFAULT_VAME_PROJECTS_ROOT`, `vame_projects_root()` et `CONFIG_POINTER` également.

- [ ] **Step 4: Lancer toute la suite**

Run: `pytest tests/ -v`
Expected: PASS. `tests/test_ethogram.py` passe et rien d'autre n'a régressé.

- [ ] **Step 5: Commit**

```bash
git add streamlit_app/lib/config.py tests/test_ethogram.py
git commit -m "lib: config.py reduit au projet courant, ETHOGRAM aligne sur run_vame"
```

---

# Phase 2 — Parcours minimal vérifiable

À la fin de cette phase on mène un projet de sa création jusqu'à l'inférence DLC. C'est ce qui valide le socle sur de vrais jobs.

---

### Task 9: `views/_job.py` — composant de suivi partagé

Présent sur toutes les pages qui lancent quelque chose. L'écrire une fois évite huit copies divergentes.

**Files:**
- Create: `streamlit_app/views/_job.py`

**Interfaces:**
- Consumes: `lib.runner`
- Produces:
  - `lancer(projet, cmd, *, cle: str) -> None` — bouton + gestion de `JobBusy`
  - `panneau(projet) -> None` — état du job courant, log en direct, annulation
  - `bouton_lancer(projet, label, cmd, *, cle, type="primary", disabled=False, help=None) -> None`

- [ ] **Step 1: Écrire le composant**

`streamlit_app/views/_job.py` :

```python
"""Affichage du job en cours — partagé par toutes les pages d'étape."""
from __future__ import annotations

from pathlib import Path

import streamlit as st

from lib import runner

_ETATS = {
    "running":     ("⏳", "En cours"),
    "succeeded":   ("✅", "Terminé"),
    "failed":      ("❌", "Échec"),
    "cancelled":   ("⛔", "Annulé"),
    "interrupted": ("⚠️", "Interrompu (l'app a été arrêtée pendant le job)"),
}


def bouton_lancer(projet: Path, label: str, cmd, *, cle: str,
                  type: str = "primary", disabled: bool = False,
                  help: str | None = None) -> None:
    occupe = runner.is_running(projet)
    if st.button(label, key=cle, type=type, disabled=disabled or occupe,
                 help=help or ("Un autre job tourne déjà." if occupe else None)):
        try:
            runner.start(projet, cmd)
            st.rerun()
        except runner.JobBusy as e:
            st.error(str(e))


@st.fragment(run_every=2.0)
def _log_en_direct(projet: Path, job_id: str, lignes: int) -> None:
    """Rafraîchit le log toutes les 2 s sans recharger la page entière.

    `st.fragment(run_every=...)` demande Streamlit >= 1.33.
    """
    st.code(runner.read_log(projet, job_id, tail=lignes) or "(pas encore de sortie)",
            language="text")
    job = runner.current(projet)
    if job and job.state != "running":
        # Le job vient de finir : on sort du fragment pour rafraîchir la page,
        # afin que les sections « artefacts produits » apparaissent.
        st.rerun()


def panneau(projet: Path, *, lignes: int = 30) -> None:
    job = runner.current(projet)
    if job is None:
        return
    icone, texte = _ETATS.get(job.state, ("•", job.state))

    entete, action = st.columns([5, 1])
    with entete:
        st.markdown(f"**{icone} {job.label}** — {texte}")
        st.caption(f"`{job.script}` (env `{job.env}`) · démarré {job.started_at}")
    with action:
        if job.state == "running":
            if st.button("Annuler", key=f"annuler_{job.job_id}"):
                runner.cancel(projet, job.job_id)
                st.rerun()

    if job.state == "running":
        _log_en_direct(projet, job.job_id, lignes)
    else:
        st.code(runner.read_log(projet, job.job_id, tail=lignes) or "(pas de sortie)",
                language="text")

    with st.expander("Log complet", expanded=False):
        st.code(runner.read_log(projet, job.job_id) or "(vide)", language="text")

    if job.state == "failed":
        st.error(
            f"Code de retour {job.returncode}. La cause est dans les dernières "
            "lignes du log ci-dessus."
        )


def historique(projet: Path, limite: int = 10) -> None:
    jobs = runner.history(projet, limit=limite)
    if not jobs:
        return
    with st.expander(f"Jobs précédents ({len(jobs)})", expanded=False):
        for job in jobs:
            icone, texte = _ETATS.get(job.state, ("•", job.state))
            st.markdown(f"{icone} **{job.label}** — {texte} · {job.started_at}")
            with st.expander("log", expanded=False):
                st.code(runner.read_log(projet, job.job_id, tail=200) or "(vide)",
                        language="text")
```

- [ ] **Step 2: Relever le plancher de version Streamlit**

`st.fragment(run_every=...)` demande Streamlit 1.33. Remplacer `streamlit>=1.30` par `streamlit>=1.33` dans `environment-pipeline.yml` (section `pip:`) et `requirements-pipeline.txt`, puis :

```bash
conda activate ethoflow && pip install "streamlit>=1.33"
```

- [ ] **Step 3: Vérifier manuellement**

Run: `conda activate ethoflow && streamlit run streamlit_app/app.py`
Expected: l'app démarre sans erreur d'import. Le composant n'est pas encore appelé, on vérifie seulement que l'import passe.

- [ ] **Step 4: Commit**

```bash
git add streamlit_app/views/_job.py environment-pipeline.yml requirements-pipeline.txt
git commit -m "views: composant de suivi de job avec log en direct et annulation"
```

---

### Task 10: Page Projet

Remplace `views/dashboard.py`. Trois responsabilités : ouvrir/créer un projet, désigner le modèle DLC, montrer l'avancement.

Le point important : `lib/config.create_project()` créait les dossiers par `mkdir`, sans `pipeline_config.yaml`, sans `--kind` et sans Excel starter. On appelle le vrai script.

**Files:**
- Modify: `streamlit_app/views/dashboard.py` → renommer en `streamlit_app/views/projet.py`
- Delete: `streamlit_app/views/dashboard.py`

**Interfaces:**
- Consumes: `lib.pipeline.create_project`, `lib.pipeline.diagnose_dlc_model`, `lib.project.*`, `lib.sessions.list_sessions`, `views._job`
- Produces: `render() -> None`

- [ ] **Step 1: Section « Ouvrir un projet »**

Reprendre le sélecteur et la suppression avec confirmation de `views/dashboard.py:65-119` (ils fonctionnent). Deux changements : `list_projects(projects_root())` prend maintenant la racine en argument, et la sélection appelle `config.set_current_project()` pour que le dernier projet soit mémorisé entre les sessions.

- [ ] **Step 2: Section « Créer un projet »**

Remplacer l'appel à `lib.config.create_project` par le script :

```python
def _section_creation() -> None:
    st.subheader("Créer un projet")
    nom = st.text_input("Nom du projet", placeholder="ex : bottomview-MCC-2026-06")
    kind = st.radio(
        "Nombre d'animaux par vidéo",
        ["single", "multi"],
        format_func=lambda k: {
            "single": "1 animal par vidéo (1 vidéo = 1 session)",
            "multi": "N animaux dans N arènes séparées (1 vidéo = N sessions)",
        }[k],
        help="Choisis selon le nombre d'animaux, pas selon l'angle caméra. "
             "'multi' active le split par arène et écrit des coordonnées par défaut.",
    )
    modeles = list_dlc_models(models_root())
    choix = st.selectbox(
        "Modèle DLC (optionnel)",
        options=["(choisir plus tard)"] + [str(m / "config.yaml") for m in modeles],
        help="Le modèle reste où il est, il n'est jamais copié dans le projet. "
             "Tu peux le désigner plus tard.",
    )
    cible = projects_root() / nom.strip().replace(" ", "-") if nom.strip() else None
    if cible and cible.exists():
        st.error(f"`{cible}` existe déjà.")
        return
    if cible:
        cmd = PL.create_project(
            cible, kind=kind,
            dlc_config=None if choix.startswith("(") else choix,
        )
        _job.bouton_lancer(cible.parent, "Créer le projet", cmd, cle="btn_creer")
```

Le job tourne dans `cible.parent` : le projet n'existe pas encore, donc son dossier `.ethoflow/jobs/` non plus. Après succès, `set_current_project(cible)`.

- [ ] **Step 3: Section « Modèle DLC »**

C'est la seule porte d'entrée pour les modèles : ils sont produits hors de l'app (Parcours B du README) et jamais entraînés ici.

```python
def _section_modele_dlc(projet: Path) -> None:
    st.subheader("Modèle DLC")
    st.caption(
        "Le modèle vit hors du projet et n'est jamais copié. Un même modèle "
        "sert à autant de projets que tu veux. Pour en entraîner un nouveau, "
        "voir le Parcours B du README — ça se fait au terminal."
    )
    actuel = dlc_config_path(projet)
    if actuel:
        existe = Path(actuel).is_file()
        (st.success if existe else st.error)(
            f"`{actuel}`" + ("" if existe else " — introuvable, modèle déplacé ?")
        )
    else:
        st.warning(
            "Aucun modèle configuré. `run_dlc_inference --mode custom` en a besoin."
        )

    modeles = list_dlc_models(models_root())
    options = [str(m / "config.yaml") for m in modeles]
    choisi = st.selectbox("Modèles trouvés", options=options) if options else None
    libre = st.text_input("…ou un chemin de config.yaml", value="")
    chemin = libre.strip() or choisi
    if chemin:
        _job.bouton_lancer(
            projet, "Utiliser ce modèle",
            PL.create_project(projet, kind=project_kind(projet),
                              dlc_config=chemin, force=True),
            cle="btn_modele",
        )

    col1, col2 = st.columns(2)
    with col1:
        _job.bouton_lancer(projet, "Diagnostiquer", PL.diagnose_dlc_model(projet),
                           cle="btn_diag", type="secondary",
                           disabled=not actuel,
                           help="Répond à l'erreur « Could not find a shuffle… ».")
    with col2:
        _job.bouton_lancer(projet, "Réparer", PL.diagnose_dlc_model(projet, fix=True),
                           cle="btn_fix", type="secondary", disabled=not actuel)
```

- [ ] **Step 4: Section « Avancement »**

Reprendre `_section_sessions` de `views/dashboard.py:147-172` en l'adaptant aux nouvelles colonnes de `lib.sessions.list_sessions(projet)` : `vidéo`, `DLC`, `split`, `nettoyage`, `VAME`. La colonne `split` n'est affichée que si `project_kind(projet) == "multi"`.

- [ ] **Step 5: Vérifier manuellement**

Run: `conda activate ethoflow && streamlit run streamlit_app/app.py`

Vérifier dans l'ordre :
1. Créer un projet `test-plan` en `single` → le job passe au vert, `configs/pipeline_config.yaml` et `test-plan_sessions.xlsx` existent sur disque.
2. Désigner un modèle DLC → `dlc_project_config` apparaît dans le YAML.
3. Cliquer Diagnostiquer → le log affiche le rapport de `diagnose_dlc_model.py`.

- [ ] **Step 6: Commit**

```bash
git add streamlit_app/views/projet.py
git rm streamlit_app/views/dashboard.py
git commit -m "views: page Projet appelle create_project.py et importe un modele DLC"
```

---

### Task 11: Page Données

Étapes 2 et 3 du README : l'Excel starter, puis le sync.

**Files:**
- Modify: `streamlit_app/views/sync_excel.py` → renommer en `streamlit_app/views/donnees.py`

**Interfaces:**
- Consumes: `lib.pipeline.sync_from_excel`, `lib.sessions`, `views._job`
- Produces: `render() -> None`

- [ ] **Step 1: Section Excel**

`create_project.py` a déjà généré `<projet>/<nom>_sessions.xlsx`. La page le localise, propose son téléchargement, et accepte le dépôt d'une version remplie qui écrase l'original après confirmation.

Afficher aussi les colonnes attendues, reprises du README §Étape 2 : `id` obligatoire (nom du fichier vidéo sans extension, clé unique de session), `mouse_id` et `group` recommandées, toutes les autres libres et recopiées telles quelles dans les metadata.

- [ ] **Step 2: Section Sync avec aperçu**

Deux boutons distincts, l'aperçu d'abord :

```python
cmd_apercu = PL.sync_from_excel(projet, videos_dir=videos_dir, excel=excel,
                                video_ext=ext, dry_run=True)
cmd_reel = PL.sync_from_excel(projet, videos_dir=videos_dir, excel=excel,
                              video_ext=ext, overwrite=overwrite)
```

`--dry-run` montre ce qui serait écrit sans rien toucher : c'est le moment de repérer un `id` qui ne correspond à aucun fichier vidéo. Le bouton réel reste actif en permanence — imposer un aperçu préalable serait pénible en usage répété.

- [ ] **Step 3: Section Sessions synchronisées**

Tableau `lib.sessions.list_sessions(projet)`, avec le nombre de sessions et le nombre de vidéos localisées. Si des vidéos manquent, renvoyer explicitement vers la page Vidéos & calibration où le re-pointage est possible (Task 19).

- [ ] **Step 4: Retirer le code mort**

`views/sync_excel.py` fait 579 lignes et contient une implémentation du parsing Excel côté app, doublon de `sync_from_excel.py`. La supprimer : le script est la seule autorité sur la lecture de l'Excel. Garder uniquement le formulaire et l'affichage.

- [ ] **Step 5: Vérifier manuellement**

Run: `streamlit run streamlit_app/app.py`
Expected: sur le projet `test-plan`, l'aperçu liste les sessions qui seraient créées ; le sync réel produit les `data/raw/<session>/metadata.yaml` et le tableau se remplit.

- [ ] **Step 6: Commit**

```bash
git add streamlit_app/views/donnees.py
git rm streamlit_app/views/sync_excel.py
git commit -m "views: page Donnees (Excel + sync avec apercu), parsing delegue au script"
```

---

### Task 12: Page Pose (DLC)

Étape 5. Extraite de `views/run_pipeline.py:54-106`, avec les corrections : `--project-dir`, `--no-prompt`, et un mode par défaut cohérent avec le `kind` du projet.

**Files:**
- Create: `streamlit_app/views/pose.py`

**Interfaces:**
- Consumes: `lib.pipeline.run_dlc_inference`, `lib.sessions`, `lib.project.project_kind`, `views._job`
- Produces: `render() -> None`

- [ ] **Step 1: Sélecteur de sessions réutilisable**

Extraire un helper partagé, utilisé aussi par les pages Nettoyage et Visualisations :

```python
def selecteur_sessions(projet: Path, *, cle: str) -> tuple[list[str], bool]:
    """Retourne (sessions choisies, drapeau --all)."""
    df = sessions.list_sessions(projet)
    tout = st.checkbox("Toutes les sessions (`--all`)", value=True, key=f"{cle}_all")
    choisies = st.multiselect(
        "Sessions", options=list(df["session_id"]), disabled=tout, key=f"{cle}_sel",
    )
    return choisies, tout
```

Le placer dans `views/_widgets.py`.

- [ ] **Step 2: Formulaire d'inférence**

Le mode par défaut suit le projet : `custom` si un modèle DLC est configuré, `superanimal` sinon en `multi`, `single-animal` si des vidéos croppées existent. Afficher les trois avec l'explication du README §Étape 5.

Avertir quand `--mode custom` est choisi sans `dlc_project_config` : le script demandera le modèle à l'invite, et avec `--no-prompt` il échouera. Renvoyer vers la section Modèle DLC de la page Projet.

`--video-adapt-batch-size` n'est affiché que si `--video-adapt` est coché, et son aide reprend le README : 2 sur GPU 16 Go, 4-8 sur 24 Go.

- [ ] **Step 3: Panneau de job et artefacts**

`_job.panneau(projet)` puis, après succès, la liste des `.h5` et `_labeled.mp4` produits dans `data/dlc-output/<session>/`, avec un renvoi vers la page Vidéos & calibration pour le QC visuel.

- [ ] **Step 4: Vérifier manuellement**

Run: `streamlit run streamlit_app/app.py`
Expected: le bouton construit la bonne commande. Sur une machine sans env `dlc`, le job passe en `failed` avec un message conda explicite dans le log — c'est le comportement voulu : un échec franc et lisible, pas un figement.

- [ ] **Step 5: Commit**

```bash
git add streamlit_app/views/pose.py streamlit_app/views/_widgets.py
git commit -m "views: page Pose (DLC) avec mode par defaut deduit du projet"
```

---

### Task 13: Navigation

`app.py` référence des vues supprimées ou renommées. On recâble, et on ajoute la garde « pas de projet ouvert ».

**Files:**
- Modify: `streamlit_app/app.py:22-84`
- Modify: `streamlit_app/views/about.py`
- Delete: `streamlit_app/views/run_pipeline.py`, `streamlit_app/views/session_details.py`

**Interfaces:**
- Consumes: toutes les vues
- Produces: la structure de navigation de la spec §8

- [ ] **Step 1: Recâbler les pages**

```python
PAGES_PROJET = [
    {"nom": "Projet",                "icone": "folder-open",       "module": projet},
]
PAGES_ETAPES = [
    {"nom": "Données",               "icone": "database",          "module": donnees},
    {"nom": "Vidéos & calibration",  "icone": "video",             "module": videos},
    {"nom": "Pose (DLC)",            "icone": "scan-line",         "module": pose},
    {"nom": "Nettoyage",             "icone": "brush-cleaning",    "module": nettoyage},
    {"nom": "VAME",                  "icone": "waypoints",         "module": vame_page},
    {"nom": "Motifs",                "icone": "tags",              "module": motifs},
    {"nom": "Analyses",              "icone": "chart-column",      "module": analyses},
    {"nom": "Visualisations",        "icone": "clapperboard",      "module": visualisations},
]
PAGES_SYSTEME = [
    {"nom": "Configuration",         "icone": "settings",          "module": configuration},
    {"nom": "À propos",              "icone": "info",              "module": about},
]
```

`PAGES_ETAPES` n'est visible que si un projet est ouvert. Le CSS de la sidebar (`app.py:86-230`) est conservé tel quel, seule la liste des pages change.

- [ ] **Step 2: Restaurer le dernier projet au démarrage**

```python
if "current_project_path" not in st.session_state:
    dernier = load_prefs().get("last_project")
    if dernier and Path(dernier).is_dir():
        st.session_state["current_project_path"] = dernier
```

Rouvrir l'app sur le projet en cours plutôt que sur un écran vide.

- [ ] **Step 3: Supprimer les vues remplacées**

`views/run_pipeline.py` est éclaté entre Pose, Nettoyage, VAME et Analyses. `views/session_details.py` est absorbé par la page Vidéos & calibration (Task 19).

- [ ] **Step 4: Mettre à jour « À propos »**

Ajouter une section qui pointe vers le README, avec une phrase explicite : l'entraînement d'un modèle DLC (Parcours B) se fait au terminal via `scripts/dlc_model-training/`, l'app ne fait qu'importer un modèle existant.

- [ ] **Step 5: Vérifier manuellement**

Run: `streamlit run streamlit_app/app.py`

Vérifier :
1. Sans projet ouvert, seules Projet, Configuration et À propos sont visibles.
2. Après ouverture d'un projet, les 8 pages d'étape apparaissent.
3. Arrêter l'app, la relancer : le projet est toujours ouvert.

- [ ] **Step 6: Commit**

```bash
git add streamlit_app/app.py streamlit_app/views/about.py
git rm streamlit_app/views/run_pipeline.py streamlit_app/views/session_details.py
git commit -m "app: navigation sur les 9 etapes, dernier projet restaure au demarrage"
```

---

# Phase 3 — Le reste du parcours

---

### Task 14: Page Nettoyage

Étape 6b. Aucune des trois actions actuellement câblées (`filter_keypoints`, `fill_nan_h5`, `trim_empty_arena`) n'est le vrai script de l'étape.

**Files:**
- Create: `streamlit_app/views/nettoyage.py`

**Interfaces:**
- Consumes: `lib.pipeline.prepare_vame_input`, `lib.pipeline.assign_arenas`, `lib.pipeline.inspect_session`, `lib.project.px_per_cm`, `views._job`, `views._widgets`
- Produces: `render() -> None`

- [ ] **Step 1: Section split par arène (multi uniquement)**

Affichée seulement si `project_kind(projet) == "multi"`. `assign_arenas` doit tourner avant le nettoyage sur la voie A (DLC multi-animal puis split). Si le projet est `single`, afficher une ligne explicative plutôt que rien : le lecteur doit comprendre pourquoi la section est absente.

- [ ] **Step 2: Section nettoyage des poses**

Les quatre passes du README §Étape 6b, avec les explications reprises telles quelles :

1. filtre médian temporel (`--window-length`, 5 frames)
2. cutoff de likelihood (`--likelihood-threshold`, 0.70, recommandation Tony/LIN)
3. détection de vitesse aberrante (`--max-speed`, 5 m/s) — **nécessite `px_per_cm`**
4. détection de points collants (`--no-sticky-detection` pour désactiver)

Point à traiter explicitement : si `px_per_cm(projet)` est `None`, la passe 3 est silencieusement désactivée par le script. C'est la passe que Tony privilégie, et son absence ne se voit nulle part.

```python
echelle = px_per_cm(projet)
if echelle is None:
    st.warning(
        "**Passe 3 (vitesse aberrante) désactivée** — `px_per_cm` n'est pas "
        "calibré pour ce projet. C'est la passe la plus efficace selon "
        "l'équipe VAME/LIN : elle attrape les points *confiants mais faux*, "
        "que le seuil de likelihood laisse passer. Calibre l'échelle dans "
        "**Vidéos & calibration → Échelle px/cm**, ou saisis la valeur "
        "ci-dessous pour cette exécution."
    )
    saisie = st.number_input("px/cm (pour cette exécution seulement)",
                             min_value=0.0, value=0.0, step=0.5)
    echelle = saisie if saisie > 0 else None
else:
    st.caption(f"Échelle : {echelle} px/cm — passe 3 active.")
```

`echelle` est ensuite passée en `px_per_cm=` à `PL.prepare_vame_input`. Un `st.caption` discret ne conviendrait pas : une passe silencieusement inactive est exactement le genre de dégradation qu'on ne remarque qu'après coup.

`--qc-bodypart` par défaut `tail_base`, avec l'explication du README : c'est le point le plus stable, un saut sur sa trajectoire est forcément une erreur de tracking.

- [ ] **Step 3: Section QC**

Après le job, afficher le résumé produit (% de frames utilisables, frames réparées) et la galerie des graphes `data/dlc-output/_qc_trajectories/`. Rappeler le seuil du README : au-delà de 10-15 % de frames réparées, le problème est le modèle DLC, pas le post-traitement — retourner au Parcours B plutôt que compenser.

Un bouton `inspect_session` pour le contrôle qualité chiffré.

- [ ] **Step 4: Repli « Outils avancés »**

Dans un `st.expander` fermé : `filter_keypoints`, `fill_nan_h5`, `trim_empty_arena`. Ces trois scripts ne sont pas projet-aware — leur passer les chemins explicitement, dérivés du projet courant. Les présenter comme des dépannages hors parcours principal, pas comme des étapes.

- [ ] **Step 5: Vérifier manuellement**

Run: `streamlit run streamlit_app/app.py`
Expected: sans `px_per_cm` configuré, l'avertissement sur la passe 3 est visible sans avoir à chercher.

- [ ] **Step 6: Commit**

```bash
git add streamlit_app/views/nettoyage.py
git commit -m "views: page Nettoyage sur prepare_vame_input_custom et ses 4 passes"
```

---

### Task 15: Page VAME

Étape 7, présentée en stepper : on voit ce qui est fait et ce qui vient.

**Files:**
- Create: `streamlit_app/views/vame_page.py`

**Interfaces:**
- Consumes: `lib.vame.stage_status`, `lib.vame.n_clusters`, `lib.pipeline.vame_stage`, `views._job`
- Produces: `render() -> None`

- [ ] **Step 1: Stepper d'état**

`lib.vame.stage_status(projet)` donne l'avancement. Chaque étape affiche son état (fait / à faire), sa durée attendue et son bouton. Les étapes dont le prérequis manque sont grisées avec l'explication.

| Étape | Durée annoncée |
|---|---|
| setup | quelques secondes |
| align | quelques minutes |
| trainset | quelques minutes |
| train | 3 à 8 h sur GPU |
| evaluate | quelques minutes |
| segment | quelques minutes |

- [ ] **Step 2: Formulaire de setup**

`--project-name`, `--pose-confidence` (0.6 par défaut), `--copy-videos` / `--no-copy-videos`, `--force`. Les `--input-dir` et `--cropped-dir` par défaut viennent du projet ; ne les exposer que dans un repli.

- [ ] **Step 3: Segment et n_clusters**

`--n-clusters` écrit la valeur dans `data/vame/config.yaml` puis segmente. Afficher la valeur courante lue par `lib.vame.n_clusters(projet)` et l'avertissement du README : chaque valeur crée son propre dossier de résultats, rien n'est écrasé, **mais `motif_labels.csv` est unique par projet** — changer `n_clusters` après avoir annoté demande de sauvegarder le CSV d'abord.

- [ ] **Step 4: Avertissement sur `train`**

`train` dure des heures. Le job survit à la fermeture de l'onglet (l'état est sur disque), et le dire explicitement à côté du bouton évite qu'on garde un onglet ouvert une nuit par précaution.

- [ ] **Step 5: Vérifier manuellement**

Run: `streamlit run streamlit_app/app.py`
Expected: sur un projet sans VAME, seul `setup` est actif ; les autres étapes sont grisées avec leur raison.

- [ ] **Step 6: Commit**

```bash
git add streamlit_app/views/vame_page.py
git commit -m "views: page VAME en stepper avec etat lu sur disque"
```

---

### Task 16: Page Motifs

Étape 8. Remplace `views/label_motifs.py`, qui écrit un YAML que rien ne lit.

**Files:**
- Modify: `streamlit_app/views/label_motifs.py` → renommer en `streamlit_app/views/motifs.py`

**Interfaces:**
- Consumes: `lib.motif_labels`, `lib.vame`, `lib.config.VOCABULAIRE_SUGGERE`, `lib.pipeline.vame_stage`, `views._job`
- Produces: `render() -> None`

- [ ] **Step 1: Génération du CSV**

Si `motif_labels.csv` est absent, deux boutons : `motif-videos` (clips + CSV, long) et `motif-labels` (CSV seul, rapide, si les clips existent déjà). Ne rien afficher d'autre tant que le fichier n'existe pas — l'app ne fabrique pas ce fichier elle-même.

- [ ] **Step 2: Reprise d'un ancien YAML**

Si `lib.motif_labels.legacy_yaml_files(projet)` retourne quelque chose, proposer la reprise. Écrire uniquement après confirmation explicite, et annoncer le nombre de labels repris.

- [ ] **Step 3: Vue par motif**

Reprendre la structure de `views/label_motifs.py:90-162`, avec ces changements :

- Le clip vient de `lib.motif_labels.video_path(projet, ligne)` — la colonne `video` du CSV, déjà résolue par le script. Supprimer `find_any_motif_video()` et ses trois globs successifs.
- Deux champs au lieu d'un : `label` en texte libre, `category` en `st.selectbox` sur `categories()` (liste fermée, groupée par les analyses).
- Le vocabulaire de `VOCABULAIRE_SUGGERE` est présenté comme **exemples pour `label`**, dans un repli, avec un intitulé qui le dit : « Exemples de labels ». Ne pas le mélanger avec le sélecteur de catégorie.
- Les motifs sont triés par `usage_pct` décroissant, conformément au conseil du README : commencer par le haut, les motifs sous 1 % ne pèsent presque rien.
- Rappeler qu'un motif ininterprétable se met en `category = artifact`, ce qui l'exclut des stats.

- [ ] **Step 4: Vue tableau**

`st.data_editor` sur le DataFrame, avec `label` et `category` éditables (le reste en lecture seule) et `category` en `SelectboxColumn` sur `categories()`. À la sauvegarde, passer par `lib.motif_labels.save` pour préserver les colonnes ajoutées à la main.

- [ ] **Step 5: Vérifier manuellement**

Run: `streamlit run streamlit_app/app.py`
Expected: éditer un label écrit bien dans `data/vame/motif_labels.csv` avec `;` et le BOM. Vérifier avec `head -2 <projet>/data/vame/motif_labels.csv`.

- [ ] **Step 6: Commit**

```bash
git add streamlit_app/views/motifs.py
git rm streamlit_app/views/label_motifs.py
git commit -m "views: page Motifs ecrit motif_labels.csv, avec reprise des anciens YAML"
```

---

### Task 17: Page Analyses

Étape 9. La partie où une interface apporte le plus : `--list-columns` sort une liste texte qu'il faut relire et retaper.

**Files:**
- Create: `streamlit_app/views/analyses.py`
- Modify: `streamlit_app/views/results.py` → fusionné dans `analyses.py`, puis supprimé
- Create: `tests/test_list_columns.py`

**Interfaces:**
- Consumes: `lib.pipeline.analyze_vame`, `lib.vame.analysis_dir`, `views._job`
- Produces: `render() -> None`, et dans `lib/analysis.py` : `parse_list_columns(texte) -> list[dict]`

- [ ] **Step 1: Écrire le test du parseur**

La sortie de `--list-columns` doit devenir des sélecteurs. Le parsing est de la logique, donc dans `lib/` et testé. `tests/test_list_columns.py` :

```python
from lib.analysis import parse_list_columns

SORTIE = """
Colonnes exploitables comme axe de comparaison (3) :

  captopril                2 groupes : Captopril (8 sessions), Control (8 sessions)
  condition                2 groupes : MCCiECKO (8 sessions), MCCf/f (8 sessions)
  cage                     4 groupes : C0 (4 sessions), C1 (4 sessions), C2 (4 sessions), C3 (4 sessions)
"""


def test_extraction_des_colonnes():
    cols = parse_list_columns(SORTIE)
    assert [c["nom"] for c in cols] == ["captopril", "condition", "cage"]
    assert cols[0]["n_groupes"] == 2
    assert cols[2]["n_groupes"] == 4


def test_resume_des_groupes_conserve():
    cols = parse_list_columns(SORTIE)
    assert "Captopril (8 sessions)" in cols[0]["groupes"]


def test_sortie_vide():
    assert parse_list_columns("") == []
    assert parse_list_columns("Aucune colonne exploitable.") == []


def test_sortie_avec_bruit_conda():
    """`conda run` préfixe parfois des lignes qui ne sont pas du script."""
    bruit = "WARNING: overwriting environment variables\n" + SORTIE
    assert len(parse_list_columns(bruit)) == 3
```

- [ ] **Step 2: Lancer pour vérifier l'échec**

Run: `pytest tests/test_list_columns.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'lib.analysis'`

- [ ] **Step 3: Implémenter le parseur**

`streamlit_app/lib/analysis.py` :

```python
"""Lecture des sorties d'analyze_vame destinées à alimenter l'interface."""
from __future__ import annotations

import re

# "  captopril                2 groupes : Captopril (8 sessions), Control (8 sessions)"
_LIGNE = re.compile(
    r"^\s{2,}(?P<nom>[A-Za-z_][\w]*)\s{2,}(?P<n>\d+)\s+groupes?\s*:\s*(?P<groupes>.+)$"
)


def parse_list_columns(texte: str) -> list[dict]:
    """Transforme la sortie de `analyze_vame --list-columns` en données.

    Retourne une liste de `{nom, n_groupes, groupes}`. Les lignes que
    `conda run` ajoute au flux sont ignorées : seules celles qui matchent
    la forme attendue sont retenues.
    """
    resultats: list[dict] = []
    for ligne in (texte or "").splitlines():
        m = _LIGNE.match(ligne.rstrip())
        if m:
            resultats.append({
                "nom": m.group("nom"),
                "n_groupes": int(m.group("n")),
                "groupes": m.group("groupes").strip(),
            })
    return resultats
```

- [ ] **Step 4: Lancer les tests**

Run: `pytest tests/test_list_columns.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Onglet « Lancer une analyse »**

1. Bouton **Découvrir les axes** → `analyze_vame(projet, list_columns=True)`. Le job est court ; à la fin on relit son log et on le parse :

```python
def _axes_disponibles(projet: Path) -> list[dict]:
    """Axes découverts, relus depuis le dernier job --list-columns."""
    for job in runner.history(projet, limit=30):
        if job.script == "analyze_vame.py" and "--list-columns" in job.argv:
            if job.state == "succeeded":
                return parse_list_columns(runner.read_log(projet, job.job_id))
            return []
    return []
```

Relire le log plutôt que mémoriser dans `session_state` : le résultat survit ainsi au rafraîchissement du navigateur, comme les jobs eux-mêmes.

2. Cases à cocher pour `--group-by`, chacune annotée du nombre de groupes et des effectifs. Reprendre l'avertissement du README : en dessous de 3 sessions par groupe le motif est ignoré, un axe à 6 groupes sur 16 sessions donne des p-values inexploitables.
3. Paires pour `--cross`, avec le nom composite affiché (`condition_x_captopril`).
4. `--extended` avec `--extended-by` limité aux axes découverts et aux composites construits.
5. `--labels` pré-rempli sur `motif_labels.csv` s'il existe, avec un rappel : sans lui les figures affichent `motif_0`, `motif_1`…

- [ ] **Step 6: Onglet « Résultats »**

Lire `lib.vame.analysis_dir(projet)` avec les **vrais** noms de fichiers du README §Étape 9 — `views/results.py:116-136` cherche des noms qui n'existent plus :

- `heatmap_usage_by_<colonne>.png`
- `mean_by_*.png`, `boxplots_top_by_*.png`, `boxplots_by_category_by_*.png`
- `bout_duration_by_*.png`, `thigmotaxis_by_*.png`, `temporal_by_motif_*.png`
- CSV : `motif_usage.csv`, `motif_usage_long.csv`, `stats_by_motif_*.csv`, `usage_by_category.csv`

Grouper par axe de comparaison plutôt que par type de fichier : c'est ainsi qu'on les lit. Chaque CSV avec aperçu et bouton de téléchargement.

Ne pas coder en dur la liste des figures : parcourir le dossier et grouper par motif de nom. `analyze_vame.py` peut en produire de nouvelles sans que l'app soit à modifier.

- [ ] **Step 7: Vérifier manuellement**

Run: `streamlit run streamlit_app/app.py`
Expected: sur un projet analysé, les figures apparaissent groupées par axe. Sur un projet sans `analysis/`, un message renvoie vers le bouton de lancement.

- [ ] **Step 8: Commit**

```bash
git add streamlit_app/views/analyses.py streamlit_app/lib/analysis.py tests/test_list_columns.py
git rm streamlit_app/views/results.py
git commit -m "views: page Analyses, --list-columns devient des selecteurs"
```

---

# Phase 4 — Vidéo et calibration

---

### Task 18: `lib/video.py` — sonde et extraction de frames

**Files:**
- Create: `streamlit_app/lib/video.py`
- Create: `tests/test_video.py`

**Interfaces:**
- Consumes: `cv2`, `lib.sessions`
- Produces:
  - `@dataclass VideoInfo(path, exists, fps, n_frames, width, height, duration_s)`
  - `probe(path) -> VideoInfo`
  - `grab_frame(path, index=0) -> np.ndarray | None`
  - `frame_png_bytes(path, index=0, max_width=None) -> bytes | None`
  - `draw_arenas(frame, coords) -> np.ndarray`
  - `find_relinks(project, videos_dir, ext="mp4") -> list[tuple[str, Path]]`
  - `apply_relinks(project, relinks) -> int`

- [ ] **Step 1: Écrire les tests qui échouent**

`tests/test_video.py`. On fabrique une vraie petite vidéo avec OpenCV : tester une sonde vidéo sur un fichier factice ne prouverait rien.

```python
import numpy as np
import pytest
import yaml

from lib import video as V


@pytest.fixture
def petite_video(tmp_path):
    cv2 = pytest.importorskip("cv2")
    chemin = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(
        str(chemin), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (64, 48))
    for i in range(20):
        frame = np.full((48, 64, 3), i * 10, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    if not chemin.exists() or chemin.stat().st_size == 0:
        pytest.skip("encodeur mp4v indisponible")
    return chemin


def test_probe(petite_video):
    info = V.probe(petite_video)
    assert info.exists
    assert info.width == 64 and info.height == 48
    assert abs(info.fps - 10.0) < 0.5
    assert info.n_frames == 20
    assert abs(info.duration_s - 2.0) < 0.3


def test_probe_fichier_absent(tmp_path):
    info = V.probe(tmp_path / "nexiste-pas.mp4")
    assert not info.exists
    assert info.fps is None and info.n_frames is None


def test_grab_frame(petite_video):
    frame = V.grab_frame(petite_video, index=5)
    assert frame is not None and frame.shape == (48, 64, 3)


def test_grab_frame_hors_bornes(petite_video):
    assert V.grab_frame(petite_video, index=9999) is None


def test_frame_png_bytes(petite_video):
    data = V.frame_png_bytes(petite_video, index=0)
    assert data.startswith(b"\x89PNG")


def test_draw_arenas_ne_modifie_pas_loriginal():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    sortie = V.draw_arenas(frame, {"A1": [10, 10, 30, 30]})
    assert frame.sum() == 0            # l'original est intact
    assert sortie.sum() > 0            # le rectangle est dessiné


def test_relink_retrouve_les_videos_deplacees(project, session_factory, tmp_path):
    """Cas du README : metadata avec chemins Windows sur machine Linux."""
    session_factory("BV-970", video=False)
    (project / "data" / "raw" / "BV-970" / "metadata.yaml").write_text(
        yaml.safe_dump({"id": "BV-970",
                        "source_video": r"D:\ancien\chemin\BV-970.mp4"}))
    nouveau = tmp_path / "videos"
    nouveau.mkdir()
    (nouveau / "BV-970.mp4").write_bytes(b"\x00")

    relinks = V.find_relinks(project, nouveau)
    assert relinks == [("BV-970", nouveau / "BV-970.mp4")]
    assert V.apply_relinks(project, relinks) == 1

    meta = yaml.safe_load(
        (project / "data" / "raw" / "BV-970" / "metadata.yaml").read_text())
    assert meta["source_video"] == str(nouveau / "BV-970.mp4")


def test_relink_ignore_les_sessions_deja_ok(project, session_factory, tmp_path):
    session_factory("BV-971")           # vidéo présente et valide
    assert V.find_relinks(project, tmp_path) == []
```

- [ ] **Step 2: Lancer pour vérifier l'échec**

Run: `pytest tests/test_video.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'lib.video'`

- [ ] **Step 3: Implémenter**

`streamlit_app/lib/video.py`. Points d'attention :

- `probe` retourne un `VideoInfo` avec des champs `None` plutôt que de lever : une vidéo manquante est un cas courant, pas une exception.
- Toujours `cap.release()`, y compris en cas d'erreur : sinon les descripteurs fuient à chaque rerun Streamlit.
- `draw_arenas` travaille sur `frame.copy()`.
- `find_relinks` cherche `<videos_dir>/<id>.<ext>` où `id` vient de `metadata["id"]` puis du nom du dossier de session, et ne retourne que les sessions dont la `source_video` actuelle est absente.
- `apply_relinks` réécrit `source_video` dans le `metadata.yaml` sans toucher aux autres clés, avec `yaml.safe_dump(..., sort_keys=False, allow_unicode=True)` pour préserver l'ordre.

- [ ] **Step 4: Lancer les tests**

Run: `pytest tests/test_video.py -v`
Expected: PASS (8 tests)

- [ ] **Step 5: Commit**

```bash
git add streamlit_app/lib/video.py tests/test_video.py
git commit -m "lib: video.py (sonde, extraction de frames, re-pointage des videos)"
```

---

### Task 19: Page Vidéos — navigateur et re-pointage

**Files:**
- Create: `streamlit_app/views/videos.py`

**Interfaces:**
- Consumes: `lib.video`, `lib.sessions`, `views._job`
- Produces: `render() -> None` (onglets : Sessions, Calibration arènes, Échelle px/cm)

- [ ] **Step 1: Onglet Sessions**

Pour la session choisie : vignette de la première frame, lecteur `st.video`, et un tableau comparant **déclaré** (metadata) et **réel** (sonde OpenCV) pour fps, dimensions, nombre de frames, durée.

Signaler visiblement un écart de fps : il fausse toutes les conversions frames → secondes en aval, et se corrige mieux ici qu'après une inférence.

Sous la vidéo, les métadonnées via `lib.sessions.metadata_fields` — toutes les clés, aucune présupposée — et le tableau des arènes via `arenes_dataframe` pour les projets `multi`.

- [ ] **Step 2: Section re-pointage**

Si des sessions ont une `source_video` absente, afficher un champ « dossier des vidéos », lister ce que `find_relinks` retrouve, et appliquer après confirmation. Ce cas est documenté au Troubleshooting du README et se produit à chaque changement de machine ou de disque.

- [ ] **Step 3: Section crop (multi uniquement)**

`crop_arenes` avec `--all` / `--all-new` / sélection. Rappeler le choix du README §Étape 4 : voie A (DLC multi-animal puis split, plus rapide) ou voie B (crop puis DLC single-animal, sortie plus propre, indispensable pour labelliser des frames).

- [ ] **Step 4: Vérifier manuellement**

Run: `streamlit run streamlit_app/app.py`
Expected: la vignette et les caractéristiques réelles s'affichent. En modifiant une `source_video` à la main vers un chemin bidon, le re-pointage la retrouve.

- [ ] **Step 5: Commit**

```bash
git add streamlit_app/views/videos.py
git commit -m "views: navigateur video avec sonde reelle et re-pointage des sources"
```

---

### Task 20: Calibration au clic

**Files:**
- Modify: `streamlit_app/views/videos.py` (onglets Calibration)
- Modify: `environment-pipeline.yml`, `requirements-pipeline.txt`

**Interfaces:**
- Consumes: `streamlit_image_coordinates`, `calibrate_arenes.save_coords_default`, `calibrate_scale.write_scale`
- Produces: deux onglets dans `views/videos.py`

- [ ] **Step 1: Ajouter la dépendance**

`streamlit-image-coordinates>=0.1.9` dans `environment-pipeline.yml` (section `pip:`) et `requirements-pipeline.txt`.

```bash
conda activate ethoflow && pip install "streamlit-image-coordinates>=0.1.9"
```

- [ ] **Step 2: Onglet Calibration arènes**

Choix de la session et de l'index de frame, extraction via `lib.video.grab_frame`, affichage cliquable. Deux clics définissent une arène (coins opposés), l'aperçu redessine tous les rectangles nommés `A1`…`A4` via `lib.video.draw_arenas` après chaque clic.

Après les clics, quatre `st.number_input` (x, y, w, h) par arène permettent l'ajustement au pixel — recliquer parfaitement est pénible, corriger de 3 px ne doit pas l'être.

L'écriture passe par le script :

```python
import sys
sys.path.insert(0, str(SCRIPTS_DIR))
from calibrate_arenes import save_coords_default

save_coords_default(projet, coords)   # dict[str, [x, y, w, h]]
```

Ne **pas** réécrire la sérialisation YAML dans l'app : deux implémentations du même format divergeraient au premier changement. `save_coords_default` fait des `print` — sans conséquence, ils partent dans la console du serveur.

- [ ] **Step 3: Onglet Échelle px/cm**

Deux clics sur les extrémités d'une distance connue, saisie de la distance en cm, calcul `px_per_cm = distance_px / known_cm`, aperçu du segment tracé sur la frame et de la valeur obtenue (`1 px = X mm`).

Un champ de saisie directe pour une valeur déjà connue (équivalent de `--set`). L'écriture passe par `calibrate_scale.write_scale(projet, px_per_cm)`.

Rappeler le conseil du README : photographier une règle plutôt que mesurer l'arène, la distorsion de lentille faussant d'autant plus la mesure que l'objet est grand.

Une source d'image supplémentaire : dépôt d'une photo de règle, en plus des frames de vidéo.

- [ ] **Step 4: Vérifier manuellement**

Run: `streamlit run streamlit_app/app.py`

Vérifier :
1. Quatre arènes cliquées → `default_arenes_coords` dans `configs/pipeline_config.yaml` au format `{A1: [x, y, w, h], …}`.
2. Deux clics + 10 cm → `px_per_cm` écrit dans le même fichier.
3. `python scripts/crop_arenes.py --project-dir <projet> --all` consomme bien ces coordonnées.

Le point 3 est la vraie validation : l'app et le CLI doivent lire le même fichier de la même façon.

- [ ] **Step 5: Commit**

```bash
git add streamlit_app/views/videos.py environment-pipeline.yml requirements-pipeline.txt
git commit -m "views: calibration arenes et echelle au clic, ecriture deleguee aux scripts"
```

---

### Task 21: Galerie QC sur la page Nettoyage

**Files:**
- Modify: `streamlit_app/views/nettoyage.py`

- [ ] **Step 1: Galerie des graphes de trajectoire**

Les fichiers sont `data/dlc-output/_qc_trajectories/<session>_<keypoint>.png`. Extraire les keypoints disponibles depuis les noms de fichiers, proposer un sélecteur, afficher en grille.

- [ ] **Step 2: Régénérer sur un autre keypoint**

Champ `--qc-bodypart` + bouton relançant `prepare_vame_input` avec les mêmes seuils sur les mêmes sessions. Le nom de fichier contient le keypoint, donc les graphes coexistent sans s'écraser — le dire, sinon on croit écraser le précédent.

- [ ] **Step 3: Comparaison brut / nettoyé**

Pour la session choisie, afficher côte à côte le `_labeled.mp4` de DLC et le graphe QC correspondant. Rappeler le critère de Tony : tracer la trajectoire sur toute la vidéo ne doit montrer aucun saut anormal, sans avoir jeté de points.

- [ ] **Step 4: Vérifier manuellement**

Run: `streamlit run streamlit_app/app.py`
Expected: après un nettoyage, les graphes apparaissent et le sélecteur de keypoint fonctionne.

- [ ] **Step 5: Commit**

```bash
git add streamlit_app/views/nettoyage.py
git commit -m "views: galerie QC des trajectoires avec selecteur de keypoint"
```

---

### Task 22: Page Visualisations

**Files:**
- Create: `streamlit_app/views/visualisations.py`

**Interfaces:**
- Consumes: `lib.pipeline.motif_gif`, `lib.pipeline.behavior_structure_gif`, `lib.pipeline.community_dendrogram`, `views._job`
- Produces: `render() -> None`

- [ ] **Step 1: Trois onglets**

- **Bande de motifs** (`motif_gif`) : session, début, durée, format mp4/gif.
- **Manifold** (`behavior_structure_gif`) : session, projection umap/pca, `--pool-all-sessions`, `--with-video`, durée, format. Signaler que `--pool-all-sessions` est long et met en garde contre le blocage documenté au Troubleshooting du README.
- **Dendrogramme** (`community_dendrogram`) : algo, `--group`, `--linkage`.

Le champ `--labels` est pré-rempli sur `motif_labels.csv` s'il existe : sans lui les figures affichent `motif_0`, `motif_1`…

- [ ] **Step 2: Affichage des rendus**

Après le job, afficher le fichier produit (`st.video` ou `st.image` selon l'extension) avec un bouton de téléchargement. Lister les rendus précédents trouvés dans le dossier de sortie, triés du plus récent au plus ancien.

- [ ] **Step 3: Vérifier manuellement**

Run: `streamlit run streamlit_app/app.py`
Expected: sur un projet segmenté, `motif_gif` produit un mp4 qui s'affiche dans la page.

- [ ] **Step 4: Commit**

```bash
git add streamlit_app/views/visualisations.py
git commit -m "views: page Visualisations (motif_gif, manifold, dendrogramme)"
```

---

### Task 23: Page Configuration et nettoyage final

**Files:**
- Modify: `streamlit_app/views/configuration.py`
- Create: `tests/test_pas_de_code_mort.py`

**Interfaces:**
- Consumes: `lib.project.save_prefs`, `lib.pipeline.SCRIPT_ENVS`
- Produces: `render() -> None`

- [ ] **Step 1: Section racines**

Champs pour `projects_root` et `models_root`, persistés via `save_prefs`. Afficher les valeurs par défaut venant de `scripts/interactive.py` et signaler quand la racine configurée n'existe pas.

- [ ] **Step 2: Section vérification des environnements**

Le risque identifié dans la spec §15 : `environment-vame.yml` ne déclare que `vame-py`, et matplotlib / scipy / umap / sklearn arrivent en dépendances transitives. Si l'une manque, `analyze_vame` échoue à l'import — après plusieurs minutes d'attente, ou après un `train` de plusieurs heures.

Un bouton **Vérifier les environnements** qui, pour chacun des trois, lance une sonde d'import et affiche le résultat :

| Env | Sonde |
|---|---|
| `ethoflow` | `import pandas, yaml, cv2, openpyxl` |
| `dlc` | `import deeplabcut, torch; print(torch.cuda.is_available())` |
| `vame` | `import vame, matplotlib, scipy, umap, sklearn` |

Chaque sonde tourne via `conda run -n <env> python -c "…"` avec un timeout de 60 s. Afficher pour l'env `dlc` si CUDA est disponible : sans GPU, l'inférence prend des heures et il vaut mieux le savoir avant de lancer.

- [ ] **Step 3: Test anti-régression sur le code mort**

`tests/test_pas_de_code_mort.py` :

```python
"""Garde-fous contre le retour des modèles périmés.

Chacun de ces symboles correspond à un bug corrigé. Les revoir apparaître
signifierait qu'on a réintroduit l'ancien modèle de données.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
APP = ROOT / "streamlit_app"

INTERDITS = {
    "vame_projects_root": "le layout VAME est plat dans le projet",
    "discover_projects": "il n'y a qu'un projet VAME, rien à découvrir",
    "DEFAULT_VAME_PROJECTS_ROOT": "racine VAME externe supprimée",
    ".vame_config_path": "l'app ne lit plus le pointeur de config VAME",
    "motif_labels_": "les labels sont dans motif_labels.csv, pas un YAML par algo",
    "--no-capture-output": "ce flag vide stdout au lieu de le capturer",
    "ANGII": "l'Excel est à colonnes libres",
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
```

- [ ] **Step 4: Lancer toute la suite**

Run: `conda activate ethoflow && pytest tests/ -v`
Expected: PASS. Si `test_aucun_symbole_perime` échoue, il nomme le fichier et la raison — corriger avant de continuer.

- [ ] **Step 5: Vérifier manuellement le parcours complet**

Run: `streamlit run streamlit_app/app.py`

Parcourir les 9 pages dans l'ordre sur un projet réel, et vérifier :
1. Chaque page se charge sans exception.
2. Chaque bouton produit un job qui apparaît dans le panneau.
3. Aucun job ne reste bloqué en `running` sans avancer — ce serait le retour du bug d'invite.
4. Le log avance en direct pendant un job long.

- [ ] **Step 6: Commit**

```bash
git add streamlit_app/views/configuration.py tests/test_pas_de_code_mort.py
git commit -m "views: page Configuration avec sondes d'environnement + gardes anti-regression"
```

---

## Récapitulatif

| Phase | Tasks | Livrable |
|---|---|---|
| 1 — Socle | 1-8 | `lib/` testé, sans Streamlit, sur le modèle de données actuel |
| 2 — Parcours minimal | 9-13 | Création de projet → Excel → sync → inférence DLC |
| 3 — Reste du parcours | 14-17 | Nettoyage, VAME, Motifs, Analyses |
| 4 — Vidéo | 18-23 | Navigateur vidéo, QC, calibration au clic, visualisations |

Chaque phase laisse le dépôt dans un état cohérent : la suite de tests passe et l'app démarre.
