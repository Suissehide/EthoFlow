"""Diagnostique un projet DLC qui refuse de servir à l'inférence.

Répond à la question : « pourquoi DLC me sort *Could not find a shuffle
with trainingset fraction X and index Y* alors que mon modèle a l'air
entraîné ? »

Vérifie dans l'ordre :

  1. le config.yaml existe et est lisible
  2. `project_path` pointe vers le dossier réel (le piège n°1 : un
     projet déplacé garde son ancien chemin en dur)
  3. des snapshots existent (le modèle a bien été entraîné)
  4. le shuffle demandé par `iteration` + `TrainingFraction` correspond
     à un dossier réellement présent
  5. les données de training-datasets/ sont là

Usage :
    # Interactif — menu des modèles trouvés sous D:/EthoFlow/models
    python scripts/diagnose_dlc_model.py

    # Sur un modèle précis
    python scripts/diagnose_dlc_model.py --model-dir D:/EthoFlow/models/souris-bottomview-Leo-2026-06-05

    # Depuis un projet EthoFlow (lit dlc_project_config)
    python scripts/diagnose_dlc_model.py --project-dir D:/EthoFlow/projects/mon-projet

    # Répare ce qui est réparable sans rien demander
    python scripts/diagnose_dlc_model.py --model-dir <...> --fix
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from interactive import (  # noqa: E402
    DEFAULT_MODELS_ROOT,
    add_no_prompt_arg,
    confirm,
    prompt,
    prompt_existing_path,
)
from paths import dlc_model_dir, pipeline_config_path  # noqa: E402


OK = "  ✓"
KO = "  ✗"
WARN = "  ⚠"


def find_shuffles(dlc_dir: Path) -> list[tuple[int | None, float, int, Path]]:
    """Liste les shuffles présents : (iteration, fraction, index, dossier)."""
    out = []
    for models_root in ("dlc-models-pytorch", "dlc-models"):
        root = dlc_dir / models_root
        if not root.exists():
            continue
        for d in root.rglob("*-trainset*shuffle*"):
            if not d.is_dir():
                continue
            m = re.search(r"-trainset(\d+)shuffle(\d+)$", d.name)
            if not m:
                continue
            frac = int(m.group(1)) / 100
            idx = int(m.group(2))
            it = None
            for parent in d.parents:
                mi = re.fullmatch(r"iteration-(\d+)", parent.name)
                if mi:
                    it = int(mi.group(1))
                    break
            out.append((it, frac, idx, d))
    return sorted(out, key=lambda x: (x[0] if x[0] is not None else -1, x[1], x[2]))


def diagnose(dlc_dir: Path, fix: bool, no_prompt: bool) -> bool:
    """Renvoie True si le modèle est utilisable à la fin du diagnostic."""
    print("=" * 66)
    print(f"Diagnostic : {dlc_dir}")
    print("=" * 66)
    print()

    cfg_path = dlc_dir / "config.yaml"
    if not cfg_path.exists():
        print(f"{KO} config.yaml absent — ce n'est pas un projet DLC.")
        return False
    print(f"{OK} config.yaml présent")

    with open(cfg_path) as f:
        cfg = yaml.safe_load(f) or {}

    # ---- 1) project_path ----
    declared = cfg.get("project_path")
    path_ok = True
    if declared is None:
        print(f"{WARN} `project_path` absent du config.yaml")
    elif Path(declared).resolve() != dlc_dir.resolve():
        path_ok = False
        print(f"{KO} `project_path` obsolète — LE PROJET A ÉTÉ DÉPLACÉ")
        print(f"       déclaré : {declared}")
        print(f"       réel    : {dlc_dir}")
        print(f"       → DLC cherche dlc-models-pytorch/ à l'ancien endroit,")
        print(f"         ne trouve rien, et sort l'erreur « shuffle ».")
        do_fix = fix or (not no_prompt and confirm(
            "       Corriger maintenant ?", default="y"))
        if do_fix:
            text = cfg_path.read_text(encoding="utf-8")
            old, new = str(declared), str(dlc_dir)
            text = text.replace(old, new)
            of, nf = old.replace("\\", "/"), new.replace("\\", "/")
            if of != old:
                text = text.replace(of, nf)
            cfg_path.write_text(text, encoding="utf-8")
            print(f"{OK} corrigé")
            path_ok = True
            cfg = yaml.safe_load(cfg_path.read_text()) or {}
    else:
        print(f"{OK} `project_path` cohérent")

    # ---- 2) snapshots ----
    snapshots = []
    for models_root in ("dlc-models-pytorch", "dlc-models"):
        root = dlc_dir / models_root
        if root.exists():
            snapshots += list(root.rglob("snapshot-*.pt"))
            snapshots += list(root.rglob("snapshot-*.index"))
    if not snapshots:
        print(f"{KO} aucun snapshot — le modèle n'a jamais été entraîné")
        labeled = dlc_dir / "labeled-data"
        n_lab = 0
        if labeled.exists():
            n_lab = sum(1 for d in labeled.iterdir()
                        if d.is_dir() and list(d.glob("CollectedData_*.h5")))
        if n_lab == 0:
            print("       → labellise dans la GUI, puis lance 02_train.py")
        else:
            print(f"       → {n_lab} vidéo(s) labellisée(s), lance 02_train.py")
        return False
    print(f"{OK} {len(snapshots)} snapshot(s) trouvé(s)")
    for s in sorted(snapshots)[-3:]:
        print(f"       · {s.name}")

    # ---- 3) shuffle demandé vs présents ----
    iteration = cfg.get("iteration", 0)
    fractions = cfg.get("TrainingFraction") or [0.95]
    shuffles = find_shuffles(dlc_dir)
    print()
    print(f"   config.yaml demande : iteration={iteration}, "
          f"TrainingFraction={fractions}")
    print(f"   shuffles présents :")
    for it, frac, idx, d in shuffles:
        print(f"       · iteration-{it}, fraction {frac}, shuffle {idx}")

    wanted_frac = float(fractions[0])
    match = [s for s in shuffles
             if s[0] == iteration and abs(s[1] - wanted_frac) < 1e-6]
    if not match:
        print(f"{KO} aucun shuffle ne correspond à ce que demande le config")
        if shuffles:
            it, frac, idx, _ = shuffles[-1]
            print(f"       → mets dans {cfg_path.name} :")
            print(f"           iteration: {it}")
            print(f"           TrainingFraction:")
            print(f"           - {frac}")
            do_fix = fix or (not no_prompt and confirm(
                "       Appliquer ces valeurs ?", default="y"))
            if do_fix:
                cfg["iteration"] = it
                cfg["TrainingFraction"] = [frac]
                with open(cfg_path, "w", encoding="utf-8") as f:
                    yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
                print(f"{OK} corrigé")
                match = [shuffles[-1]]
        if not match:
            return False
    else:
        print(f"{OK} le shuffle demandé existe")

    # ---- 4) contenu du shuffle ----
    shuffle_dir = match[0][3]
    train_dir = shuffle_dir / "train"
    if not train_dir.exists():
        print(f"{KO} dossier train/ absent dans {shuffle_dir.name}")
        return False
    snaps_here = list(train_dir.glob("snapshot-*.pt")) + \
        list(train_dir.glob("snapshot-*.index"))
    if not snaps_here:
        print(f"{KO} aucun snapshot dans {shuffle_dir.name}/train/")
        return False
    print(f"{OK} {len(snaps_here)} snapshot(s) dans le shuffle ciblé")

    # ---- 5) training-datasets ----
    td = dlc_dir / "training-datasets"
    if not td.exists() or not any(td.rglob("*.pickle")) and not any(td.rglob("*.mat")):
        print(f"{WARN} training-datasets/ vide ou incomplet")
        print(f"       (sans impact pour l'inférence, mais bloquera un")
        print(f"        ré-entraînement — relance create_training_dataset)")

    print()
    print("=" * 66)
    print("✅ Le modèle est utilisable pour l'inférence.")
    print("=" * 66)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model-dir", type=Path, default=None,
                        help="Dossier du projet DLC à diagnostiquer.")
    parser.add_argument("--project-dir", type=Path, default=None,
                        help="Projet EthoFlow — lit dlc_project_config "
                             "depuis son pipeline_config.yaml.")
    parser.add_argument("--fix", action="store_true",
                        help="Applique les corrections sans demander.")
    add_no_prompt_arg(parser)
    args = parser.parse_args()

    dlc_dir = args.model_dir

    if dlc_dir is None and args.project_dir is not None:
        cfg_path = pipeline_config_path(args.project_dir)
        if cfg_path.exists():
            cfg = yaml.safe_load(cfg_path.read_text()) or {}
            dlc_cfg = cfg.get("dlc_project_config")
            if dlc_cfg:
                dlc_dir = dlc_model_dir(dlc_cfg)
                print(f"ℹ  Modèle lu depuis {cfg_path.name} : {dlc_dir}\n")

    if dlc_dir is None:
        models = []
        if DEFAULT_MODELS_ROOT.exists():
            models = sorted(d for d in DEFAULT_MODELS_ROOT.iterdir()
                            if d.is_dir() and (d / "config.yaml").exists())
        if models and not args.no_prompt:
            print(f"Modèles DLC trouvés dans {DEFAULT_MODELS_ROOT} :")
            for i, m in enumerate(models, start=1):
                print(f"  {i}. {m.name}")
            print(f"  {len(models) + 1}. (autre chemin)")
            choice = prompt("Modèle à diagnostiquer", default="1")
            if choice.isdigit() and 1 <= int(choice) <= len(models):
                dlc_dir = models[int(choice) - 1]
        if dlc_dir is None:
            if args.no_prompt:
                print("❌ --model-dir ou --project-dir requis en mode "
                      "--no-prompt.", file=sys.stderr)
                sys.exit(1)
            dlc_dir = prompt_existing_path("Dossier du projet DLC",
                                            must_exist=True)

    # --model-dir accepte aussi le config.yaml : on remonte au dossier.
    ok = diagnose(dlc_model_dir(dlc_dir).resolve(), args.fix, args.no_prompt)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
