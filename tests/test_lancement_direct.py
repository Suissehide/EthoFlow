"""Lancement des jobs sans `conda run`.

`conda run` retient toute la sortie du script jusqu'à sa fin : sur une
inférence DLC de plusieurs heures, l'app n'affichait ni log ni
progression. Le Python de l'env est lancé directement quand on le trouve.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "streamlit_app"))
from lib import pipeline as PL  # noqa: E402


def _faux_conda(tmp_path: Path, env: str) -> Path:
    base = tmp_path / "miniforge3"
    if os.name == "nt":
        (base / "Scripts").mkdir(parents=True)
        (base / "Scripts" / "conda.exe").write_text("")
        py = base / "envs" / env / "python.exe"
        conda = base / "Scripts" / "conda.exe"
    else:
        (base / "bin").mkdir(parents=True)
        (base / "bin" / "conda").write_text("")
        py = base / "envs" / env / "bin" / "python"
        conda = base / "bin" / "conda"
    py.parent.mkdir(parents=True, exist_ok=True)
    py.write_text("")
    return conda


def test_python_de_l_env_lance_directement(tmp_path, monkeypatch):
    conda = _faux_conda(tmp_path, "dlc")
    monkeypatch.setenv("CONDA_EXE", str(conda))
    cmd = PL.Command(env="dlc", script="run_dlc_inference.py", args=["--all"])
    argv, variables = PL.lancement_direct(cmd)
    assert "envs" in argv[0] and Path(argv[0]).name.startswith("python")
    assert argv[1].endswith("run_dlc_inference.py") and argv[2:] == ["--all"]
    assert variables["CONDA_DEFAULT_ENV"] == "dlc"
    # Les dossiers de l'env passent en tête du PATH, comme `conda activate`.
    assert Path("envs", "dlc").as_posix() in Path(variables["PATH"].split(os.pathsep)[0]).as_posix()


def test_env_introuvable_retombe_sur_conda_run(tmp_path, monkeypatch):
    monkeypatch.setenv("CONDA_EXE", str(tmp_path / "nulle-part" / "bin" / "conda"))
    monkeypatch.delenv("CONDA_PREFIX", raising=False)
    monkeypatch.delenv("CONDA_PREFIX_1", raising=False)
    cmd = PL.Command(env="dlc", script="run_dlc_inference.py")
    assert PL.lancement_direct(cmd) is None
