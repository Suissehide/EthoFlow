"""Sync sessions bottom-view depuis un Excel maître + un dossier de vidéos.

Schéma de l'Excel maître attendu (produit par combine_bottomview_excels.py
ou édité à la main) :

    Feuille 'Sessions', en-tête à la ligne 1 :
        id | mouse_id | sex | group | cage | tail_label | birth_date |
        animal_id | line | origin |
        genotype_mcc | genotype_cdh5_cre | genotype_col1_egfp

    **Une ligne = une vidéo = une session.**

    `id` (recommandé) = nom du fichier vidéo sans extension
    (`970` → `970.mp4`). C'est la clé unique de la session et le nom du
    dossier créé dans data/raw/ (préfixé `BV-`).

    `mouse_id` identifie l'ANIMAL. Il peut se répéter sur plusieurs
    lignes si la même souris est filmée à plusieurs timepoints — dans
    ce cas `id` diffère (`970-M1`, `970-M2`) et on obtient deux
    sessions distinctes pour le même animal.

    Rétrocompat : si la colonne `id` est absente, `mouse_id` sert
    d'identifiant de session (comportement historique 1 vidéo/souris).

    Pas de splitting d'arène ici (contrairement au schéma multi-animal
    où N souris cohabitent dans N cadrans d'une même vidéo).

Workflow type avec deux journées d'acquisition :

    python scripts/sync_from_excel_single.py \\
        --project-dir D:/ethoflow/projects/bottomview-MCC-2026-06 \\
        --excel D:/ethoflow/projects/bottomview-MCC-2026-06/bottomview_sessions.xlsx \\
        --videos-dir E:/data/bottom_view/08062026 \\
        --date 2026-06-08

    python scripts/sync_from_excel_single.py \\
        --project-dir D:/ethoflow/projects/bottomview-MCC-2026-06 \\
        --excel D:/ethoflow/projects/bottomview-MCC-2026-06/bottomview_sessions.xlsx \\
        --videos-dir E:/data/bottom_view/11062026 \\
        --date 2026-06-11

Le filtre se fait par présence du fichier vidéo dans --videos-dir :
seules les souris dont le `<mouse_id>.mp4` est trouvé donnent lieu à une
session. Tu peux donc passer le même Excel maître pour les deux batches.

Schéma de metadata produit :

    session_id: BV-970
    project: BottomView
    mouse_id: 970
    sex: F
    group: MCCf/f
    cage: CD329
    tail_label: 1
    birth_date: 2024-10-15
    date_recorded: 2026-06-08
    animal_id: 54310               # si dispo (batch 11/06 surtout)
    line: MCC*Cdh5-cre
    genotype_cdh5_cre: cre+        # si dispo
    source_video: E:/data/bottom_view/08062026/970.mp4
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

try:
    import pandas as pd
except ImportError:
    print("❌ pandas requis. Installe : pip install pandas openpyxl", file=sys.stderr)
    sys.exit(1)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import add_project_dir_arg, raw_dir, resolve_project  # noqa: E402


def parse_master_excel(excel_path: Path) -> pd.DataFrame:
    """Lit la feuille 'Sessions' de l'Excel maître.

    Format attendu : en-tête à la ligne 1, **une ligne par vidéo**.

    Colonne `id` (recommandée) = nom du fichier vidéo sans extension.
    C'est la clé unique de la session : elle permet d'avoir plusieurs
    enregistrements de la même souris (design longitudinal) — `mouse_id`
    se répète alors sur plusieurs lignes, `id` reste unique
    (ex. `970-M1`, `970-M2`).

    Rétrocompat : si `id` est absent, on retombe sur `mouse_id` comme
    identifiant de session (comportement historique, 1 vidéo par souris).

    Les autres colonnes sont propagées dans la metadata si présentes et
    non-nulles.
    """
    df = pd.read_excel(excel_path, sheet_name="Sessions", header=0)
    df.columns = [str(c).strip() for c in df.columns]

    has_id = "id" in df.columns
    if not has_id and "mouse_id" not in df.columns:
        raise ValueError(
            f"L'Excel {excel_path} doit avoir une colonne 'id' (nom du "
            f"fichier vidéo) OU 'mouse_id' dans la feuille 'Sessions'. "
            f"Colonnes trouvées : {list(df.columns)}"
        )

    key_col = "id" if has_id else "mouse_id"
    df = df.dropna(subset=[key_col]).copy()

    # Normalise `id` en str (peut être numérique dans Excel : 970 → "970")
    if has_id:
        df["id"] = df["id"].apply(
            lambda v: str(int(v)) if isinstance(v, float) and v.is_integer()
            else str(v).strip()
        )
        # Détecte les doublons d'id — ça écraserait des sessions
        dups = df["id"][df["id"].duplicated()].unique()
        if len(dups):
            raise ValueError(
                f"Colonne 'id' avec doublons dans {excel_path} : "
                f"{list(dups)}. Chaque ligne doit avoir un 'id' unique "
                f"(= un fichier vidéo distinct)."
            )
    else:
        print("ℹ  Pas de colonne 'id' — utilisation de 'mouse_id' comme "
              "identifiant de session (mode historique, 1 vidéo/souris).",
              file=sys.stderr)

    if "mouse_id" in df.columns:
        df["mouse_id"] = df["mouse_id"].apply(
            lambda v: int(v) if pd.notna(v) else None
        )
    return df


def session_key(row: pd.Series) -> str:
    """Identifiant de session = `id` si dispo, sinon `mouse_id`."""
    if "id" in row.index and pd.notna(row["id"]):
        return str(row["id"]).strip()
    return str(int(row["mouse_id"]))


# Champs Excel → champs metadata.yaml (les non listés sont ignorés)
META_FIELDS_OPTIONAL = [
    ("sex", "sex"),
    ("group", "group"),
    ("cage", "cage"),
    ("tail_label", "tail_label"),
    ("birth_date", "birth_date"),
    ("animal_id", "animal_id"),
    ("line", "line"),
    ("origin", "origin"),
    ("genotype_mcc", "genotype_mcc"),
    ("genotype_cdh5_cre", "genotype_cdh5_cre"),
    ("genotype_col1_egfp", "genotype_col1_egfp"),
    # Traitement pharmaco (bottomview MCC-2026-06 : oui = 2024 cohort, non = 2026)
    ("captopril", "captopril"),
    ("notes", "notes"),
]


def _coerce(val):
    """Cast intelligent pour les valeurs YAML : int si possible, str sinon."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    if isinstance(val, (int, float)) and not pd.isna(val):
        # Convertit les floats entiers en int (Excel lit souvent en float)
        if isinstance(val, float) and val.is_integer():
            return int(val)
        return val
    s = str(val).strip()
    return s if s else None


def build_metadata(
    row: pd.Series,
    video_path: Path,
    date_recorded: str | None,
) -> dict:
    """Construit le dict metadata pour une session (= une vidéo)."""
    key = session_key(row)
    session_id = f"BV-{key}"

    meta = {
        "session_id": session_id,
        "project": "BottomView",
        "source_video": str(video_path),
    }
    # mouse_id = l'ANIMAL (peut se répéter entre sessions longitudinales)
    if "mouse_id" in row.index and pd.notna(row["mouse_id"]):
        meta["mouse_id"] = int(row["mouse_id"])
    if date_recorded:
        meta["date_recorded"] = date_recorded

    # Champs optionnels : seulement si présents dans le DF ET non vides
    for excel_col, meta_key in META_FIELDS_OPTIONAL:
        if excel_col not in row.index:
            continue
        val = _coerce(row[excel_col])
        if val is not None:
            meta[meta_key] = val
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    add_project_dir_arg(parser, required=True)
    parser.add_argument(
        "--excel", required=True, type=Path,
        help="Fichier Excel décrivant les souris de ce batch",
    )
    parser.add_argument(
        "--videos-dir", required=True, type=Path,
        help="Dossier contenant les .mp4 nommés <Numéromarquage>.mp4",
    )
    parser.add_argument(
        "--date", type=str, default=None,
        help="Date d'enregistrement (YYYY-MM-DD) injectée dans metadata. Optionnel.",
    )
    parser.add_argument(
        "--video-ext", default="mp4",
        help="Extension à matcher (défaut: mp4)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Affiche les sessions à créer sans rien écrire",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Écrase les metadata.yaml déjà existants (sinon skip)",
    )
    args = parser.parse_args()

    if not args.excel.exists():
        print(f"❌ Excel introuvable : {args.excel}", file=sys.stderr)
        sys.exit(1)
    if not args.videos_dir.exists():
        print(f"❌ Dossier vidéos introuvable : {args.videos_dir}", file=sys.stderr)
        sys.exit(1)

    project = resolve_project(args)
    raw = raw_dir(project)
    print(f"Projet     : {project}")
    print(f"Excel      : {args.excel}")
    print(f"Vidéos     : {args.videos_dir}")
    print(f"Sortie     : {raw}")
    if args.date:
        print(f"Date       : {args.date}")
    print()

    df = parse_master_excel(args.excel)
    print(f"Lignes Excel valides : {len(df)}\n")

    n_written = 0
    n_skipped_existing = 0
    n_no_video = 0
    for _, row in df.iterrows():
        key = session_key(row)
        video_path = args.videos_dir / f"{key}.{args.video_ext}"
        session_id = f"BV-{key}"

        if not video_path.exists():
            print(f"  ✗ {session_id}: vidéo absente ({video_path.name})")
            n_no_video += 1
            continue

        session_dir = raw / session_id
        metadata_path = session_dir / "metadata.yaml"

        if metadata_path.exists() and not args.overwrite:
            print(f"  · {session_id}: metadata déjà existante, skip")
            n_skipped_existing += 1
            continue

        meta = build_metadata(row, video_path, args.date)

        if args.dry_run:
            print(f"  [dry] {session_id}  group={meta.get('group')} → {video_path.name}")
        else:
            session_dir.mkdir(parents=True, exist_ok=True)
            with open(metadata_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(meta, f, allow_unicode=True, sort_keys=False)
            print(f"  ✓ {session_id}  group={meta.get('group')} → {video_path.name}")
        n_written += 1

    print()
    verb = "à écrire" if args.dry_run else "écrits"
    print(f"✅ {n_written} metadata {verb}")
    if n_skipped_existing:
        print(f"   {n_skipped_existing} session(s) skip (déjà existante — relance avec --overwrite si tu veux écraser)")
    if n_no_video:
        print(f"⚠  {n_no_video} ligne(s) Excel sans vidéo correspondante dans {args.videos_dir}")

    if not args.dry_run and n_written > 0:
        print(
            "\nÉtape suivante :\n"
            f"  python scripts/run_dlc_inference.py --project-dir {project} "
            f"--all --mode custom"
        )


if __name__ == "__main__":
    main()
