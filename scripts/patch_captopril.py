"""Ajoute le champ `captopril` aux metadata.yaml sans re-syncer l'Excel.

Si tu as déjà généré tes metadata.yaml via sync_from_excel.py et
que tu veux juste ajouter (ou corriger) le champ `captopril` sans tout
regénérer, ce script te permet de patcher en place.

Trois modes d'attribution (mutuellement exclusifs) :

    1) --by-birth-year 2024=oui 2026=non
       Lit l'année du birth_date de chaque metadata et applique le mapping.
       Utile si les birth_dates sont correctes.

    2) --by-mouse-ids <id1> <id2> ... --value oui
       Force la valeur pour la liste de mouse_ids fournie. À combiner avec
       plusieurs invocations si tu veux différentes valeurs pour différents
       groupes.

    3) --from-excel <path> [--sheet Sessions]
       Relit l'Excel maître et applique la colonne `captopril` par mouse_id.
       Comme sync_from_excel mais sans écraser le reste de la
       metadata. Pratique si tu as juste ajouté la colonne captopril dans
       l'Excel et ne veux pas repartir de zéro.

Usage :
    # Mode birth-year
    python scripts/patch_captopril.py \\
        --project-dir D:/ethoflow/projects/bottomview-MCC-2026-06 \\
        --by-birth-year 2024=oui 2026=non

    # Mode liste explicite (utile si les birth_dates sont fausses)
    python scripts/patch_captopril.py \\
        --project-dir D:/ethoflow/projects/bottomview-MCC-2026-06 \\
        --by-mouse-ids 970 971 972 973 --value oui

    # Mode Excel : relit la colonne captopril depuis l'Excel maître
    python scripts/patch_captopril.py \\
        --project-dir D:/ethoflow/projects/bottomview-MCC-2026-06 \\
        --from-excel D:/ethoflow/projects/bottomview-MCC-2026-06/bottomview_sessions.xlsx

Après patching, relance :
    python scripts/analyze_vame.py --project-dir <...> --extended
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import add_project_dir_arg, raw_dir, resolve_project  # noqa: E402


def parse_year_mapping(items: list[str]) -> dict[str, str]:
    """Parse '2024=oui 2026=non' → {'2024': 'oui', '2026': 'non'}."""
    out = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Attendu format YEAR=VALUE, reçu : {item}")
        year, val = item.split("=", 1)
        out[year.strip()] = val.strip()
    return out


def year_of(birth_date) -> str | None:
    """Extrait l'année d'un birth_date (str 'YYYY-MM-DD' ou datetime)."""
    if birth_date is None:
        return None
    s = str(birth_date).strip()
    if len(s) < 4:
        return None
    return s[:4] if s[:4].isdigit() else None


def load_excel_captopril(excel_path: Path, sheet: str) -> dict[int, str]:
    """Lit la colonne captopril de l'Excel maître, retourne {mouse_id: val}."""
    try:
        import pandas as pd
    except ImportError:
        print("❌ pandas requis pour --from-excel", file=sys.stderr)
        sys.exit(1)
    df = pd.read_excel(excel_path, sheet_name=sheet, header=0)
    df.columns = [str(c).strip() for c in df.columns]
    if "mouse_id" not in df.columns:
        raise ValueError(f"'mouse_id' absent des colonnes {list(df.columns)}")
    if "captopril" not in df.columns:
        raise ValueError(
            f"'captopril' absent des colonnes {list(df.columns)}. "
            f"Ajoute la colonne 'captopril' dans l'Excel avant de patcher."
        )
    df = df.dropna(subset=["mouse_id"]).copy()
    df["mouse_id"] = df["mouse_id"].astype(int)
    out = {}
    for _, row in df.iterrows():
        val = row.get("captopril")
        if val is None:
            continue
        s = str(val).strip()
        if s and s.lower() != "nan":
            out[int(row["mouse_id"])] = s
    return out


def apply_patch(
    project: Path,
    year_map: dict[str, str] | None,
    mouse_ids: set[int] | None,
    value: str | None,
    excel_map: dict[int, str] | None,
    dry_run: bool,
) -> None:
    raw = raw_dir(project)
    if not raw.exists():
        print(f"❌ raw/ introuvable dans {project}", file=sys.stderr)
        sys.exit(1)

    n_patched = 0
    n_skipped = 0
    n_no_rule = 0
    for session_dir in sorted(raw.iterdir()):
        meta_path = session_dir / "metadata.yaml"
        if not meta_path.exists():
            continue
        with open(meta_path) as f:
            meta = yaml.safe_load(f) or {}

        mouse_id = meta.get("mouse_id")
        session_id = meta.get("session_id") or session_dir.name

        # Détermine la valeur à appliquer selon le mode
        new_val = None
        if excel_map is not None and mouse_id is not None:
            new_val = excel_map.get(int(mouse_id))
        elif year_map is not None:
            y = year_of(meta.get("birth_date"))
            if y is not None:
                new_val = year_map.get(y)
        elif mouse_ids is not None:
            if mouse_id is not None and int(mouse_id) in mouse_ids:
                new_val = value

        if new_val is None:
            n_no_rule += 1
            print(f"  · {session_id}: aucune règle applicable (mouse_id={mouse_id}, "
                  f"birth_date={meta.get('birth_date')})")
            continue

        old_val = meta.get("captopril")
        if old_val == new_val:
            print(f"  = {session_id}: déjà à '{new_val}'")
            n_skipped += 1
            continue

        if dry_run:
            print(f"  [dry] {session_id}: captopril '{old_val}' → '{new_val}'")
        else:
            meta["captopril"] = new_val
            with open(meta_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(meta, f, allow_unicode=True, sort_keys=False)
            print(f"  ✓ {session_id}: captopril = '{new_val}' "
                  f"({'ajouté' if old_val is None else f'était {old_val!r}'})")
        n_patched += 1

    print()
    verb = "à patcher" if dry_run else "patchés"
    print(f"✅ {n_patched} metadata {verb}")
    if n_skipped:
        print(f"   {n_skipped} déjà à la bonne valeur, non touchés")
    if n_no_rule:
        print(f"⚠  {n_no_rule} session(s) sans règle applicable (voir plus haut)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    add_project_dir_arg(parser, required=True)

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--by-birth-year", nargs="+", metavar="YEAR=VAL",
        help="Mapping par année de birth_date, ex : 2024=oui 2026=non",
    )
    group.add_argument(
        "--by-mouse-ids", nargs="+", type=int, metavar="ID",
        help="Liste explicite de mouse_ids à patcher (à combiner avec --value)",
    )
    group.add_argument(
        "--from-excel", type=Path,
        help="Relit la colonne captopril depuis un Excel maître",
    )

    parser.add_argument(
        "--value", type=str, default=None,
        help="Valeur à appliquer avec --by-mouse-ids (ex : oui / non)",
    )
    parser.add_argument(
        "--sheet", default="Sessions",
        help="Feuille de l'Excel à lire avec --from-excel (défaut : Sessions)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Affiche les changements sans écrire",
    )
    args = parser.parse_args()

    project = resolve_project(args)
    print(f"Projet : {project}\n")

    year_map = None
    mouse_ids = None
    excel_map = None

    if args.by_birth_year:
        year_map = parse_year_mapping(args.by_birth_year)
        print(f"Mode : birth-year → {year_map}\n")
    elif args.by_mouse_ids:
        if args.value is None:
            print("❌ --by-mouse-ids requiert --value <oui|non|...>", file=sys.stderr)
            sys.exit(1)
        mouse_ids = set(args.by_mouse_ids)
        print(f"Mode : mouse-ids explicites → {sorted(mouse_ids)} = '{args.value}'\n")
    elif args.from_excel:
        if not args.from_excel.exists():
            print(f"❌ Excel introuvable : {args.from_excel}", file=sys.stderr)
            sys.exit(1)
        excel_map = load_excel_captopril(args.from_excel, args.sheet)
        print(f"Mode : Excel → {len(excel_map)} valeurs chargées de "
              f"{args.from_excel.name}\n")

    apply_patch(project, year_map, mouse_ids, args.value, excel_map, args.dry_run)


if __name__ == "__main__":
    main()
