"""Sondes d'import pour les trois environnements conda du pipeline.

Le risque (spec §15) : `environment-vame.yml` ne déclare que `vame-py`,
matplotlib / scipy / umap / scikit-learn arrivent en dépendances
transitives. Si l'une manque, `analyze_vame.py` ou
`behavior_structure_gif.py` échouent à l'import — potentiellement après un
`run_vame.py train` de plusieurs heures. Une sonde d'import de quelques
secondes, lancée depuis la page Configuration, permet de le savoir avant
de lancer un job long plutôt qu'après.

Module sans Streamlit (comme tout `lib/`, sauf `config.py`) : testable en
monkeypatchant `subprocess.run`, sans conda ni GPU réels.
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass

# Une sonde par env : le code Python exact exécuté via
# `conda run -n <env> python -c "<code>"`. Pour `dlc`, une ligne de sortie
# supplémentaire signale explicitement la disponibilité de CUDA — sans
# GPU, l'inférence prend des heures et il vaut mieux le savoir avant de
# lancer (voir `parse_cuda_line`).
PROBES: dict[str, str] = {
    "ethoflow": "import pandas, yaml, cv2, openpyxl",
    "dlc": "import deeplabcut, torch; print('CUDA_DISPONIBLE=' + str(torch.cuda.is_available()))",
    "vame": "import vame, matplotlib, scipy, umap, sklearn",
}

TIMEOUT_DEFAUT = 60


@dataclass(frozen=True)
class ProbeResult:
    env: str
    ok: bool
    output: str
    cuda: bool | None = None  # seulement renseigné pour l'env `dlc`


def parse_cuda_line(stdout: str) -> bool | None:
    """Extrait `CUDA_DISPONIBLE=True/False` de la sortie de la sonde `dlc`.

    `None` si la ligne est absente (sonde en échec avant le print, ou sonde
    d'un autre env).
    """
    for ligne in stdout.splitlines():
        ligne = ligne.strip()
        if ligne.startswith("CUDA_DISPONIBLE="):
            return ligne.split("=", 1)[1].strip() == "True"
    return None


def probe_env(env: str, *, timeout: int = TIMEOUT_DEFAUT) -> ProbeResult:
    """Lance la sonde d'import de `env` via `conda run` et interprète le résultat.

    `env` doit être une clé de `PROBES` (`KeyError` volontaire sinon, comme
    `SCRIPT_ENVS` dans `lib.pipeline` : se tromper de nom doit planter fort,
    pas silencieusement).
    """
    code = PROBES[env]
    argv = ["conda", "run", "-n", env, "python", "-c", code]
    try:
        result = subprocess.run(
            argv, capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return ProbeResult(
            env=env, ok=False,
            output=f"Délai dépassé ({timeout}s) — l'environnement `{env}` "
                   "n'a pas répondu à temps.",
        )
    except FileNotFoundError:
        return ProbeResult(
            env=env, ok=False,
            output="`conda` introuvable dans le PATH de l'app.",
        )

    sortie = ((result.stdout or "") + (result.stderr or "")).strip()
    ok = result.returncode == 0
    cuda = parse_cuda_line(result.stdout or "") if env == "dlc" and ok else None
    return ProbeResult(env=env, ok=ok, output=sortie, cuda=cuda)


def probe_all(*, timeout: int = TIMEOUT_DEFAUT) -> dict[str, ProbeResult]:
    """Sonde les trois environnements dans l'ordre `ethoflow`, `dlc`, `vame`."""
    return {env: probe_env(env, timeout=timeout) for env in PROBES}
