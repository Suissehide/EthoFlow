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
for extra in (ROOT / "streamlit_app", ROOT / "scripts",
              ROOT / "scripts" / "dlc_model-training"):
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
