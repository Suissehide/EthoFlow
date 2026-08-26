"""Helper : point d'entrée pour charger un `_config.py` custom.

Utilisé par tous les scripts numérotés (01 → 06) pour supporter le flag
`--config-dir <dossier>` de manière uniforme, sans dupliquer la logique.

Sans le flag, le dossier est **demandé à l'invite** — menu numéroté des
dossiers de config trouvés sous la racine des modèles, exactement comme
`--project-dir` et `--dlc-config` ailleurs dans EthoFlow. Il n'y a plus de
repli silencieux sur le `_config.py` template du repo : il porte la vidéo
pilote et le nom de projet de quelqu'un d'autre, et un script qui part
dessus sans le dire fabrique un projet DLC au mauvais endroit.

Pattern d'utilisation dans un script numéroté :

    import argparse
    from pathlib import Path
    import sys

    # 1) Insère le dossier du script dans sys.path pour trouver ce helper
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _load_config import add_config_dir_arg, load_config  # noqa: E402

    parser = argparse.ArgumentParser(...)
    add_config_dir_arg(parser)
    # ... autres args ...
    args = parser.parse_args()

    # 2) Résout le dossier de config (flag ou invite) et l'insère en tête
    #    de sys.path avant d'importer _config
    config_dir = load_config(args)
    from _config import PROJECT_DIR, CONFIG  # noqa: E402
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# `interactive` vit dans scripts/, un cran au-dessus.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from interactive import (  # noqa: E402
    DEFAULT_MODELS_ROOT,
    add_no_prompt_arg,
    prompt_dlc_config_dir,
)


def add_config_dir_arg(parser: argparse.ArgumentParser) -> None:
    """Ajoute --config-dir (+ --no-prompt, son pendant non-interactif)."""
    parser.add_argument(
        "--config-dir", type=Path, default=None,
        help="Dossier contenant ton _config.py custom (produit par "
             "00_init_training_config.py). Demandé à l'invite si absent.",
    )
    add_no_prompt_arg(parser)


def load_config(args_or_dir) -> Path:
    """Résout le dossier de config et l'insère en tête de `sys.path`.

    Pour que `from _config import ...` cible la bonne copie. Accepte un
    Namespace argparse (lit `args.config_dir`, et y réécrit la valeur
    résolue — `01_setup_project` en a besoin pour savoir où merger le
    projet DLC) ou directement un Path.

    Renvoie le dossier retenu.
    """
    args = args_or_dir if isinstance(args_or_dir, argparse.Namespace) else None
    cd = args.config_dir if args is not None else args_or_dir

    if cd is None:
        no_prompt = bool(getattr(args, "no_prompt", False)) if args else False
        cd = prompt_dlc_config_dir(DEFAULT_MODELS_ROOT, no_prompt=no_prompt)

    cd = Path(cd).resolve()
    if not (cd / "_config.py").exists():
        print(f"❌ _config.py introuvable dans {cd}\n"
              f"   Lance d'abord : python scripts/dlc_model-training/"
              f"00_init_training_config.py", file=sys.stderr)
        sys.exit(1)
    sys.path.insert(0, str(cd))
    if args is not None:
        args.config_dir = cd

    # Le dossier du template reste en FIN de sys.path : un _config.py
    # partiel qui ne définirait pas toutes les constantes attendues y
    # trouve les défauts, sans jamais primer sur le config de l'user.
    sys.path.append(str(Path(__file__).resolve().parent))
    return cd
