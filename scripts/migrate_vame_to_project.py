"""Migre un projet VAME de l'ancien emplacement legacy global vers
le dossier projet-aware `<project>/data/vame/<name>/`.

Pourquoi : avant la refactor projet-aware, run_vame.py écrivait tous
ses projets VAME dans un dossier global partagé entre projets EthoFlow
(`<repo_parent>/vame-projects/`). Avec plusieurs projets EthoFlow
(topview + bottomview, par exemple), deux projets peuvent porter le
même nom de projet VAME et entrent en collision. La migration scope
chaque projet VAME à son projet EthoFlow.

Ce que fait le script :
    1. Déplace le dossier complet du projet VAME vers
       `<ethoflow_project>/data/vame/<name>/`.
    2. Réécrit toutes les occurrences de l'ancien path absolu dans le
       `config.yaml` du projet VAME (project_path, paths des sessions,
       etc) — le YAML est sérialisé tel quel, on fait un str.replace
       global qui couvre tous les champs sans risque de casser le
       schéma.
    3. Écrit `<ethoflow_project>/.vame_config_path` pointant vers le
       nouveau `config.yaml`. Supprime le pointer legacy global
       `<repo>/.vame_config_path` s'il pointait vers le projet migré.

Sécurité :
    - --dry-run : affiche l'opération sans rien déplacer/écrire.
    - Idempotent : si la cible existe déjà OU si la source est déjà à
      la bonne place, le script s'arrête proprement.
    - Le déplacement est atomique au niveau filesystem (shutil.move)
      tant que source et cible sont sur le même volume. Sinon il copie
      puis supprime, ce qui reste safe.

Usage :
    # Migre le projet VAME bottomview-MCC-2026-06 vers son projet EthoFlow
    python scripts/migrate_vame_to_project.py \\
        --project-dir D:\\ethoflow\\projects\\bottomview-MCC-2026-06 \\
        --vame-project-name bottomview-MCC-2026-06

    # Avec un chemin source non-standard (legacy != <repo_parent>/vame-projects)
    python scripts/migrate_vame_to_project.py \\
        --project-dir D:\\ethoflow\\projects\\foo \\
        --vame-project-name foo-vame \\
        --legacy-vame-projects-dir D:\\autre\\dossier\\vame-projects

    # Dry-run pour voir le diff de paths sans toucher au disque
    python scripts/migrate_vame_to_project.py \\
        --project-dir <...> --vame-project-name <...> --dry-run
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import (  # noqa: E402
    REPO_ROOT,
    add_project_dir_arg,
    resolve_project,
    vame_config_pointer,
    vame_dir,
)


def default_legacy_dir() -> Path:
    """Emplacement legacy par défaut où vivaient les projets VAME."""
    return REPO_ROOT.parent / "vame-projects"


def rewrite_config_paths(
    config_path: Path,
    old_prefix: str,
    new_prefix: str,
    dry_run: bool,
) -> int:
    """Remplace toutes les occurrences textuelles de old_prefix par new_prefix
    dans le config.yaml. Retourne le nombre de remplacements faits.

    Volontairement low-level (find/replace texte plutôt que parsing YAML +
    réécriture) : VAME's config.yaml a plein de chemins dans des
    sous-structures (session_names, …) et le find/replace les capture tous
    sans dépendre du schéma. Comme les paths absolus sont uniques et
    contiennent le séparateur de répertoire, il n'y a pas de risque de
    collision avec un autre champ.
    """
    text = config_path.read_text(encoding="utf-8")
    # On normalise les slashes pour matcher autant le format Windows que Unix.
    # VAME peut écrire indifféremment "D:\\foo\\bar" ou "D:/foo/bar".
    variants = {old_prefix, old_prefix.replace("\\", "/"), old_prefix.replace("/", "\\")}
    n_total = 0
    for variant in variants:
        if variant in text:
            new_variant = (
                new_prefix.replace("\\", "/") if "/" in variant
                else new_prefix.replace("/", "\\")
            )
            n = text.count(variant)
            text = text.replace(variant, new_variant)
            n_total += n
            print(f"    {n:>4d} × {variant!r} → {new_variant!r}")

    if n_total and not dry_run:
        config_path.write_text(text, encoding="utf-8")
    return n_total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    add_project_dir_arg(parser, required=True)
    parser.add_argument(
        "--vame-project-name", required=True,
        help="Nom du dossier du projet VAME à migrer "
             "(ex: 'bottomview-MCC-2026-06')",
    )
    parser.add_argument(
        "--legacy-vame-projects-dir", type=Path, default=None,
        help="Dossier legacy contenant <vame-project-name>/ "
             f"(défaut: {default_legacy_dir()})",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="N'effectue aucune écriture/déplacement, affiche juste les opérations.",
    )
    args = parser.parse_args()

    project = resolve_project(args)
    legacy_root = args.legacy_vame_projects_dir or default_legacy_dir()
    src = legacy_root / args.vame_project_name
    dst_root = vame_dir(project)
    dst = dst_root / args.vame_project_name

    print(f"Projet EthoFlow  : {project}")
    print(f"Source legacy    : {src}")
    print(f"Destination      : {dst}")
    if args.dry_run:
        print("\n[DRY RUN] aucune écriture ne sera faite.\n")

    # Sanity checks
    if src.resolve() == dst.resolve():
        print("\n✅ Source == destination : projet déjà migré (rien à faire).")
        return
    if not src.exists():
        print(f"\n❌ Source absente : {src}", file=sys.stderr)
        sys.exit(1)
    if dst.exists():
        print(f"\n❌ Destination déjà occupée : {dst}\n"
              "   Si tu veux refaire la migration, supprime d'abord la "
              "destination, ou bouge-la manuellement.",
              file=sys.stderr)
        sys.exit(1)

    # 1) Move le dossier
    if args.dry_run:
        print(f"  [dry] move {src} → {dst}")
    else:
        print(f"\n📦 Déplacement {src} → {dst}")
        dst_root.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
        print("  ✓ déplacement OK")

    # 2) Réécriture du config.yaml
    config_path = dst / "config.yaml" if not args.dry_run else src / "config.yaml"
    print(f"\n📝 Réécriture des paths dans {config_path.relative_to(REPO_ROOT.parent) if config_path.is_relative_to(REPO_ROOT.parent) else config_path}")
    if not config_path.exists():
        print(f"  ⚠️  config.yaml introuvable, skip (à vérifier manuellement)")
    else:
        n = rewrite_config_paths(
            config_path,
            str(src),
            str(dst),
            dry_run=args.dry_run,
        )
        if n == 0:
            print("    Aucun chemin absolu legacy détecté (déjà propre ?)")
        else:
            print(f"    Total : {n} remplacement(s)")

    # 3) Pointer .vame_config_path : nouveau (projet) + cleanup legacy
    new_pointer = vame_config_pointer(project)
    legacy_pointer = REPO_ROOT / ".vame_config_path"
    new_pointer_content = str(dst / "config.yaml")

    if args.dry_run:
        print(f"\n[dry] write pointer : {new_pointer} → {new_pointer_content}")
    else:
        new_pointer.parent.mkdir(parents=True, exist_ok=True)
        new_pointer.write_text(new_pointer_content)
        print(f"\n✓ pointer écrit : {new_pointer}")

    if legacy_pointer.exists():
        legacy_content = legacy_pointer.read_text().strip()
        # Le pointer legacy pointait-il vers le projet qu'on vient de migrer ?
        if str(src) in legacy_content or args.vame_project_name in legacy_content:
            if args.dry_run:
                print(f"[dry] delete legacy pointer : {legacy_pointer}")
            else:
                legacy_pointer.unlink()
                print(f"✓ pointer legacy supprimé : {legacy_pointer}")
        else:
            print(f"ℹ️  pointer legacy {legacy_pointer} pointe vers un autre projet "
                  f"({legacy_content}), conservé tel quel.")

    print(
        "\n✅ Migration terminée.\n\n"
        "Tu peux maintenant relancer les commandes VAME normalement :\n"
        f"  python scripts/run_vame.py --project-dir {project} info\n"
        f"  python scripts/run_vame.py --project-dir {project} motif-videos\n"
    )


if __name__ == "__main__":
    main()
