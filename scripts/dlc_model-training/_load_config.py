"""Helper : point d'entrée pour charger un `_config.py` custom.

Utilisé par tous les scripts numérotés (01 → 06) pour supporter le flag
`--config-dir <dossier>` de manière uniforme, sans dupliquer la logique.

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

    # 2) Insère éventuellement le --config-dir user avant d'importer _config
    load_config(args)
    from _config import PROJECT_DIR, CONFIG  # noqa: E402
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def add_config_dir_arg(parser: argparse.ArgumentParser) -> None:
    """Ajoute --config-dir au parser."""
    parser.add_argument(
        "--config-dir", type=Path, default=None,
        help="Dossier contenant ton _config.py custom (produit par "
             "00_init_training_config.py). Sans ce flag, utilise le "
             "template dans scripts/dlc_model-training/.",
    )


def load_config(args_or_dir) -> None:
    """Insère le dossier de config au top de sys.path pour que
    `from _config import ...` cible la bonne copie.

    Accepte soit un Namespace argparse (lit args.config_dir), soit
    directement un Path/None.
    """
    if isinstance(args_or_dir, argparse.Namespace):
        cd = args_or_dir.config_dir
    else:
        cd = args_or_dir

    if cd is not None:
        cd = Path(cd).resolve()
        if not (cd / "_config.py").exists():
            print(f"❌ _config.py introuvable dans {cd}\n"
                  f"   Lance d'abord : python scripts/dlc_model-training/"
                  f"00_init_training_config.py", file=sys.stderr)
            sys.exit(1)
        sys.path.insert(0, str(cd))
    # Toujours ajouter aussi le dossier du template comme fallback : au
    # cas où l'utilisateur donne un _config.py partiel qui référencerait
    # implicitement les défauts du template (peu probable, mais safe).
    sys.path.append(str(Path(__file__).resolve().parent))
