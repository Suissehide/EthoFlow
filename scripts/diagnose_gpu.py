"""Diagnostique pourquoi torch ne voit pas la GPU.

Répond à : « `torch.cuda.is_available()` renvoie False, et maintenant ? »
Il y a quatre causes possibles et elles ne se corrigent pas pareil :

  1. torch installé en build CPU (`2.5.1+cpu`) — le cas le plus fréquent
     sur Windows, `pip install torch` sert la build CPU par défaut.
  2. torch CUDA mais aucun driver NVIDIA utilisable.
  3. torch CUDA + driver présent, mais GPU trop récente pour la build
     (RTX 50xx / Blackwell = sm_120, absent des builds stables).
  4. Pas de GPU NVIDIA du tout sur la machine.

Le script identifie laquelle et donne la commande exacte à lancer.

C'est LE problème qu'on rencontre après chaque `conda env create -f
environment-dlc.yml` : le fichier d'env installe `torch` depuis PyPI, et
PyPI sert la build CPU sur Windows. Recréer l'env repose donc le problème
à chaque fois — d'où `--fix`, qui réinstalle sans rien avoir à recopier.

Usage :
    conda activate dlc
    python scripts/diagnose_gpu.py          # diagnostic seul
    python scripts/diagnose_gpu.py --fix    # diagnostic + réinstallation

Sans GPU, le pipeline tourne quand même : DLC et VAME fonctionnent sur CPU,
juste 10 à 50× plus lentement. Le script le rappelle en fin de diagnostic.
"""
from __future__ import annotations

import argparse
import platform
import re
import shutil
import subprocess
import sys

OK = "  ✓"
KO = "  ✗"
WARN = "  ⚠"

# Architectures GPU et la version minimale de CUDA qui les supporte.
# Sert à repérer le cas « build CUDA correcte mais trop ancienne pour ma
# carte » : la GPU est vue par le driver mais pas par torch.
_ARCH_MIN_CUDA = [
    (r"RTX\s*50\d\d", "Blackwell (sm_120)", "12.8"),
    (r"RTX\s*40\d\d", "Ada Lovelace (sm_89)", "11.8"),
    (r"H100|H200", "Hopper (sm_90)", "11.8"),
    (r"A100|A\d{4}", "Ampere (sm_80)", "11.1"),
    (r"RTX\s*30\d\d", "Ampere (sm_86)", "11.1"),
]


def run(cmd: list[str]) -> str | None:
    """Exécute une commande, retourne stdout ou None si indisponible."""
    if shutil.which(cmd[0]) is None:
        return None
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    except (subprocess.TimeoutExpired, OSError):
        return None
    return r.stdout if r.returncode == 0 else None


def nvidia_smi() -> tuple[str | None, str | None, list[str]]:
    """(version du driver, version CUDA max du driver, noms des GPU)."""
    out = run(["nvidia-smi",
                "--query-gpu=name,driver_version",
                "--format=csv,noheader"])
    if out is None:
        return None, None, []
    names, driver = [], None
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if parts and parts[0]:
            names.append(parts[0])
        if len(parts) > 1:
            driver = parts[1]
    cuda_max = None
    header = run(["nvidia-smi"])
    if header:
        m = re.search(r"CUDA Version:\s*([\d.]+)", header)
        if m:
            cuda_max = m.group(1)
    return driver, cuda_max, names


def arch_of(gpu_name: str) -> tuple[str, str] | None:
    """(architecture, CUDA minimale) pour un nom de GPU, si reconnu."""
    for pattern, arch, min_cuda in _ARCH_MIN_CUDA:
        if re.search(pattern, gpu_name, re.IGNORECASE):
            return arch, min_cuda
    return None


# Index de wheels réellement publiés par PyTorch. On ne peut recommander
# que ceux-là : `--index-url .../whl/cu111` n'existe pas et échouerait.
_WHEEL_INDEXES = ["cu118", "cu121", "cu124", "cu126", "cu128"]


def pick_wheel_index(min_cuda: str, driver_cuda_max: str | None
                      ) -> tuple[str, bool]:
    """Choisit (index de wheel, faut-il la nightly ?).

    Contraintes : au moins `min_cuda` pour que l'architecture de la carte
    soit compilée dans la build, au plus la version que le driver expose.
    À qualité égale on prend cu124, la build stable la plus éprouvée.
    """
    def num(tag: str) -> float:
        return float(tag[2:4] + "." + tag[4:])

    lo = float(min_cuda)
    hi = float(driver_cuda_max) if driver_cuda_max else 99.0
    candidates = [t for t in _WHEEL_INDEXES if num(t) >= lo and num(t) <= hi]
    if not candidates:
        # Carte trop récente pour toute build stable (Blackwell aujourd'hui)
        return "cu128", True
    if "cu124" in candidates:
        return "cu124", False
    # cu128 n'existe pour l'instant qu'en nightly
    choice = candidates[-1]
    return choice, choice == "cu128"


def cuda_install_cmd(tag: str, nightly: bool) -> str:
    """Commande d'installation torch pour un index de wheel donné."""
    if nightly:
        return (
            "pip uninstall torch torchvision torchaudio -y\n"
            "  pip install --pre torch torchvision torchaudio \\\n"
            f"      --index-url https://download.pytorch.org/whl/nightly/{tag}"
        )
    return (
        "pip uninstall torch torchvision torchaudio -y\n"
        "  pip install torch torchvision torchaudio \\\n"
        f"      --index-url https://download.pytorch.org/whl/{tag}"
    )


def other_conda_envs() -> list[tuple[str, str]]:
    """[(nom d'env, chemin de son python)] pour les autres envs conda."""
    out = []
    raw = run(["conda", "env", "list"])
    if not raw:
        return out
    here = str(sys.executable).lower()
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.replace("*", " ").split()
        if len(parts) < 2:
            continue
        name, path = parts[0], parts[-1]
        exe = (f"{path}\\python.exe" if platform.system() == "Windows"
               else f"{path}/bin/python")
        if exe.lower() == here or not shutil.os.path.exists(exe):
            continue
        out.append((name, exe))
    return out


def scan_other_envs() -> list[tuple[str, str, bool]]:
    """Cherche un env conda voisin où torch voit déjà la GPU.

    Si `vame` fonctionne pendant que `dlc` échoue, la build installée dans
    `vame` est la preuve vivante de ce qui marche sur cette machine — plus
    fiable que n'importe quelle déduction depuis le nom de la carte.
    """
    snippet = (
        "import torch,sys;"
        "sys.stdout.write(torch.__version__+'|'+str(torch.cuda.is_available()))"
    )
    found = []
    for name, exe in other_conda_envs():
        out = run([exe, "-c", snippet])
        if not out or "|" not in out:
            continue
        version, avail = out.strip().split("|", 1)
        found.append((name, version, avail == "True"))
    return found


def apply_fix(tag: str, nightly: bool) -> int:
    """Désinstalle torch et réinstalle depuis le bon index. Renvoie un code.

    On passe par `sys.executable -m pip` et non `pip` : ça garantit qu'on
    installe dans l'environnement courant, même si un autre pip traîne
    plus haut dans le PATH (piège classique avec conda sur Windows).
    """
    index = (f"https://download.pytorch.org/whl/nightly/{tag}" if nightly
             else f"https://download.pytorch.org/whl/{tag}")
    steps = [
        [sys.executable, "-m", "pip", "uninstall", "-y",
         "torch", "torchvision", "torchaudio"],
        [sys.executable, "-m", "pip", "install"]
        + (["--pre"] if nightly else [])
        + ["torch", "torchvision", "torchaudio", "--index-url", index],
    ]
    for cmd in steps:
        print(f"\n$ {' '.join(cmd[2:])}\n")
        r = subprocess.run(cmd)
        if r.returncode != 0 and "uninstall" not in cmd:
            print(f"\n❌ Échec (code {r.returncode}).", file=sys.stderr)
            return r.returncode
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--fix", action="store_true",
                        help="Réinstalle torch depuis l'index CUDA adapté "
                             "à ta carte, dans l'environnement courant.")
    args = parser.parse_args()

    print("=" * 66)
    print("Diagnostic GPU / CUDA")
    print("=" * 66)
    print(f"  Python  : {sys.version.split()[0]}  ({sys.executable})")
    print(f"  Système : {platform.system()} {platform.release()}")
    print()

    # ---- 1) Le driver et le matériel ----
    driver, cuda_max, gpus = nvidia_smi()
    if driver is None:
        print(f"{KO} `nvidia-smi` introuvable ou en échec")
        print("     → soit la machine n'a pas de GPU NVIDIA,")
        print("       soit le driver n'est pas installé.")
        print("     Vérifie dans le Gestionnaire de périphériques → "
              "Cartes graphiques.")
        has_gpu = False
    else:
        has_gpu = True
        print(f"{OK} Driver NVIDIA {driver}"
              + (f" (supporte CUDA ≤ {cuda_max})" if cuda_max else ""))
        for g in gpus:
            info = arch_of(g)
            suffix = f"  [{info[0]}, CUDA ≥ {info[1]}]" if info else ""
            print(f"       · {g}{suffix}")

    # ---- 2) torch ----
    print()
    try:
        import torch
    except ImportError:
        print(f"{KO} torch n'est pas installé dans cet environnement")
        print("     → conda activate dlc, puis relance ce script.")
        sys.exit(1)

    version = torch.__version__
    built_cuda = getattr(torch.version, "cuda", None)
    is_cpu_build = version.endswith("+cpu") or built_cuda is None
    print(f"   torch {version}")

    if is_cpu_build:
        print(f"{KO} build CPU — torch a été installé sans support CUDA")
        print("     C'est le piège n°1 sur Windows : `pip install torch`")
        print("     sert la build CPU par défaut.")
    else:
        print(f"{OK} build CUDA {built_cuda}")

    available = torch.cuda.is_available()
    if available:
        print(f"{OK} torch.cuda.is_available() = True")
        try:
            name = torch.cuda.get_device_name(0)
            print(f"       GPU visible : {name}")
            x = torch.randn(512, 512, device="cuda")
            _ = (x @ x).sum().item()
            print(f"{OK} calcul GPU réel effectué — tout est fonctionnel")
        except Exception as e:  # noqa: BLE001
            print(f"{KO} la GPU est déclarée mais le calcul échoue : {e}")
            available = False
    else:
        print(f"{KO} torch.cuda.is_available() = False")

    # ---- 3) Verdict + commande à lancer ----
    print()
    print("=" * 66)
    if available:
        print("✅ Rien à faire, l'entraînement utilisera la GPU.")
        print("=" * 66)
        sys.exit(0)

    if not has_gpu:
        print("Verdict : aucune GPU NVIDIA détectée sur cette machine.")
        print()
        print("Le pipeline fonctionne quand même sur CPU, mais compte")
        print("10 à 50× plus lent : une inférence DLC de 20 min de vidéo")
        print("passe de ~2 min à ~1 h, et l'entraînement VAME de 3-8 h")
        print("à plusieurs jours. Pour l'inférence sur quelques vidéos")
        print("c'est tenable ; pour entraîner un modèle, trouve une")
        print("machine avec GPU.")
        print("=" * 66)
        sys.exit(1)

    # GPU présente mais pas utilisée : reste à savoir quelle build viser.
    # CUDA minimale exigée par l'architecture de la carte. Inconnue → 11.8,
    # le plancher raisonnable pour tout ce qui tourne aujourd'hui.
    min_cuda = "11.8"
    arch_name = None
    for g in gpus:
        info = arch_of(g)
        if info:
            arch_name, min_cuda = info
            break

    if is_cpu_build:
        print("Verdict : GPU présente, mais torch est en build CPU.")
    elif built_cuda and float(built_cuda) < float(min_cuda):
        print(f"Verdict : la build torch (CUDA {built_cuda}) est trop "
              f"ancienne pour cette carte.")
        if arch_name:
            print(f"          {arch_name} exige CUDA ≥ {min_cuda}.")
    else:
        print(f"Verdict : torch est compilé pour CUDA {built_cuda} et la "
              f"carte devrait convenir,")
        print("          mais elle reste invisible. Causes fréquentes :")
        print("           · driver NVIDIA trop ancien pour cette build")
        print("           · CUDA_VISIBLE_DEVICES posé à vide ou à -1")
        print("           · GPU accaparée par un autre processus / une "
              "session RDP")
        print("          Réinstaller torch sur la bonne version CUDA reste")
        print("          la première chose à essayer :")

    tag, nightly = pick_wheel_index(min_cuda, cuda_max)

    # Un env voisin qui fonctionne vaut mieux qu'une déduction : si `vame`
    # voit la GPU, sa build est prouvée sur CETTE machine.
    working = [(n, v) for n, v, ok in scan_other_envs() if ok]
    if working:
        print()
        print("   Autres environnements conda où torch voit déjà la GPU :")
        for name, version in working:
            print(f"       · {name} : torch {version}")
        m = re.search(r"\+(cu\d+)", working[0][1])
        if m:
            tag = m.group(1)
            nightly = "dev" in working[0][1] or tag == "cu128"
            print(f"   → on vise la même build ({tag}"
                  f"{', nightly' if nightly else ''}).")

    print()

    if args.fix:
        print(f"--fix : réinstallation depuis {tag}"
              f"{' (nightly)' if nightly else ''}")
        print(f"        environnement ciblé : {sys.executable}")
        print("=" * 66)
        rc = apply_fix(tag, nightly)
        if rc == 0:
            print()
            print("=" * 66)
            print("Réinstallation terminée. Relance le diagnostic :")
            print("  python scripts/diagnose_gpu.py")
            print("=" * 66)
        sys.exit(rc)

    print("Dans l'environnement où tu as le problème :")
    print()
    print(f"  {cuda_install_cmd(tag, nightly)}")
    if nightly:
        print()
        print("  (nightly : aucune build stable ne couvre encore cette "
              "architecture)")
    print()
    print("Ou laisse le script le faire :")
    print("  python scripts/diagnose_gpu.py --fix")
    print("=" * 66)
    sys.exit(1)


if __name__ == "__main__":
    main()
