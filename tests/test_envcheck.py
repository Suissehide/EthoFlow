"""Tests de `lib.envcheck` — sans conda ni GPU réels : `subprocess.run` monkeypatché.

Les vérifications avec les VRAIS environnements conda (`ethoflow`/`dlc`/
`vame`) sont faites manuellement, hors suite pytest (voir le rapport de la
Task 23) — elles prennent jusqu'à 60 s par environnement et dépendent d'une
installation locale que la CI n'a pas.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for extra in (ROOT / "streamlit_app",):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from lib import envcheck as EC  # noqa: E402


class _FakeCompleted:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def test_probe_env_argv_conda_run(monkeypatch):
    """La commande construite suit le patron `conda run -n <env> python -c <code>`."""
    captured = {}

    def fake_run(argv, **kwargs):
        captured["argv"] = argv
        return _FakeCompleted(returncode=0, stdout="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    EC.probe_env("ethoflow")
    argv = captured["argv"]
    assert argv[:3] == ["conda", "run", "-n"]
    assert argv[3] == "ethoflow"
    assert argv[4:6] == ["python", "-c"]
    assert argv[6] == EC.PROBES["ethoflow"]


def test_probe_env_ok(monkeypatch):
    monkeypatch.setattr(
        subprocess, "run",
        lambda argv, **kw: _FakeCompleted(returncode=0, stdout=""),
    )
    res = EC.probe_env("ethoflow")
    assert res.ok is True
    assert res.env == "ethoflow"
    assert res.cuda is None


def test_probe_env_echec_import(monkeypatch):
    monkeypatch.setattr(
        subprocess, "run",
        lambda argv, **kw: _FakeCompleted(
            returncode=1, stdout="", stderr="ModuleNotFoundError: No module named 'umap'"
        ),
    )
    res = EC.probe_env("vame")
    assert res.ok is False
    assert "umap" in res.output


def test_probe_env_timeout(monkeypatch):
    def fake_run(argv, **kw):
        raise subprocess.TimeoutExpired(cmd=argv, timeout=kw.get("timeout", 60))

    monkeypatch.setattr(subprocess, "run", fake_run)
    res = EC.probe_env("dlc", timeout=5)
    assert res.ok is False
    assert "5" in res.output


def test_probe_env_conda_absent(monkeypatch):
    def fake_run(argv, **kw):
        raise FileNotFoundError("conda")

    monkeypatch.setattr(subprocess, "run", fake_run)
    res = EC.probe_env("ethoflow")
    assert res.ok is False
    assert "conda" in res.output.lower()


def test_probe_dlc_cuda_disponible(monkeypatch):
    monkeypatch.setattr(
        subprocess, "run",
        lambda argv, **kw: _FakeCompleted(returncode=0, stdout="CUDA_DISPONIBLE=True\n"),
    )
    res = EC.probe_env("dlc")
    assert res.ok is True
    assert res.cuda is True


def test_probe_dlc_cuda_indisponible(monkeypatch):
    monkeypatch.setattr(
        subprocess, "run",
        lambda argv, **kw: _FakeCompleted(returncode=0, stdout="CUDA_DISPONIBLE=False\n"),
    )
    res = EC.probe_env("dlc")
    assert res.ok is True
    assert res.cuda is False


def test_parse_cuda_line_absente():
    assert EC.parse_cuda_line("rien à voir ici") is None


def test_probe_all_couvre_les_trois_envs(monkeypatch):
    monkeypatch.setattr(
        subprocess, "run",
        lambda argv, **kw: _FakeCompleted(returncode=0, stdout=""),
    )
    resultats = EC.probe_all()
    assert set(resultats) == {"ethoflow", "dlc", "vame"}
    assert all(r.ok for r in resultats.values())
