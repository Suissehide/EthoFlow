"""Génère les metadata.yaml des sessions à partir de l'Excel maître.

**Un seul script pour les deux schémas** — le format est détecté
automatiquement depuis les feuilles présentes dans le classeur :

  - Feuille `Sessions`                     → schéma 1 animal / vidéo
  - Feuilles `Trials_Videos` + `Arena_Mapping` → schéma N animaux / vidéo

Tu n'as donc pas à savoir quel « type » de projet tu as : remplis
l'Excel généré par `create_project.py` et lance ce script.

---------------------------------------------------------------------
Schéma 1 animal / vidéo — feuille `Sessions`
---------------------------------------------------------------------

    id | mouse_id | group | sex | cage | tail_label | birth_date |
    line | origin | genotype_* | captopril | notes

Une ligne = une vidéo = une session. Seule `id` est obligatoire ;
`mouse_id` et `group` sont recommandés. Toute autre colonne présente
est recopiée telle quelle dans le metadata.yaml — tu peux ajouter les
tiennes sans toucher au code.

`id` = nom du fichier vidéo sans extension (`970` → `970.mp4`). C'est
la clé unique de la session et le nom du dossier `data/raw/BV-<id>/`.

`mouse_id` = l'ANIMAL. Peut se répéter sur plusieurs lignes si la même
souris est filmée à plusieurs timepoints (`970-M1`, `970-M2`) — on
obtient alors deux sessions distinctes pour le même animal.

Rétrocompat : sans colonne `id`, `mouse_id` sert de clé de session.

---------------------------------------------------------------------
Schéma N animaux / vidéo — feuilles Subjects + Trials_Videos + Arena_Mapping
---------------------------------------------------------------------

    Subjects        : MouseID | Baseline group (M1) | ANGII group (M2) |
                      Stress (CUS?) | Notes
    Trials_Videos   : TrialCode | Timepoint | Date | VideoNo |
                      Original file name | FPS | Width | Height | Notes
    Arena_Mapping   : TrialCode | Arena | MouseID | Notes

Une ligne de `Trials_Videos` = une vidéo = N sessions (une par arène).

---------------------------------------------------------------------
Usage
---------------------------------------------------------------------

    # Interactif — le script demande ce qui manque
    python scripts/sync_from_excel.py

    # Tout en arguments
    python scripts/sync_from_excel.py \\
        --project-dir D:/EthoFlow/projects/bottomview-MCC-2026-06 \\
        --excel D:/EthoFlow/projects/bottomview-MCC-2026-06/..._sessions.xlsx \\
        --videos-dir E:/data/bottom_view/08062026

    # Non-interactif (CI) — échoue si un argument manque
    python scripts/sync_from_excel.py --project-dir <...> --excel <...> \\
        --videos-dir <...> --no-prompt

Répète la commande pour chaque batch d'acquisition (`--videos-dir`
change, l'Excel reste le même). `--overwrite` pour re-générer une
metadata déjà existante.

"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

try:
    import pandas as pd
except ImportError:
    print("❌ pandas requis. Installe : pip install pandas openpyxl",
          file=sys.stderr)
    sys.exit(1)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import raw_dir  # noqa: E402
from interactive import (  # noqa: E402
    add_no_prompt_arg,
    prompt,
    prompt_existing_path,
    resolve_or_prompt_project,
)


# ======================================================================
# Détection du schéma
# ======================================================================

def detect_schema(excel_path: Path) -> str:
    """Renvoie 'single' ou 'multi' selon les feuilles présentes."""
    xl = pd.ExcelFile(excel_path)
    sheets = set(xl.sheet_names)
    if {"Trials_Videos", "Arena_Mapping"}.issubset(sheets):
        return "multi"
    if "Sessions" in sheets:
        return "single"
    raise ValueError(
        f"Impossible de détecter le schéma de {excel_path}.\n"
        f"Feuilles trouvées : {sorted(sheets)}\n"
        f"Attendu : une feuille 'Sessions' (1 animal/vidéo) OU les "
        f"feuilles 'Trials_Videos' + 'Arena_Mapping' (N animaux/vidéo)."
    )


def _coerce(val):
    """Cast pour YAML : int si possible, str sinon, None si vide/NaN."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    if isinstance(val, (int, float)) and not pd.isna(val):
        if isinstance(val, float) and val.is_integer():
            return int(val)
        return val
    s = str(val).strip()
    return s if s else None


# ======================================================================
# Schéma 1 animal / vidéo
# ======================================================================

# Colonnes du template recopiées telles quelles dans metadata.yaml.
# Note : toute colonne PRÉSENTE dans l'Excel et non listée ici est aussi
# recopiée (cf. build_metadata_single) — cette liste ne sert qu'à fixer
# l'ordre des champs connus. L'utilisateur peut ajouter ses propres
# colonnes sans toucher au code.
META_FIELDS_SINGLE = [
    "group", "sex", "cage", "tail_label", "birth_date",
    "line", "origin", "genotype_mcc", "genotype_cdh5_cre",
    "genotype_col1_egfp", "captopril", "notes",
]

# Colonnes à ne jamais recopier telles quelles (déjà traitées à part)
_RESERVED_COLS = {"id", "mouse_id"}


def parse_single(excel_path: Path) -> pd.DataFrame:
    """Lit et valide la feuille `Sessions`."""
    df = pd.read_excel(excel_path, sheet_name="Sessions", header=0)
    df.columns = [str(c).strip() for c in df.columns]

    has_id = "id" in df.columns
    if not has_id and "mouse_id" not in df.columns:
        raise ValueError(
            f"La feuille 'Sessions' doit avoir une colonne 'id' (nom du "
            f"fichier vidéo) OU 'mouse_id'. Colonnes : {list(df.columns)}"
        )

    key_col = "id" if has_id else "mouse_id"
    df = df.dropna(subset=[key_col]).copy()

    if has_id:
        df["id"] = df["id"].apply(
            lambda v: str(int(v)) if isinstance(v, float) and v.is_integer()
            else str(v).strip()
        )
        dups = df["id"][df["id"].duplicated()].unique()
        if len(dups):
            raise ValueError(
                f"Colonne 'id' avec doublons : {list(dups)}. Chaque ligne "
                f"doit avoir un 'id' unique (= un fichier vidéo distinct)."
            )
    else:
        print("ℹ  Pas de colonne 'id' — 'mouse_id' sert d'identifiant de "
              "session (mode historique, 1 vidéo/souris).")

    if "mouse_id" in df.columns:
        df["mouse_id"] = df["mouse_id"].apply(
            lambda v: int(v) if pd.notna(v) else None
        )
    return df


def session_key_single(row: pd.Series) -> str:
    """Clé de session = `id` si dispo, sinon `mouse_id`."""
    if "id" in row.index and pd.notna(row["id"]):
        return str(row["id"]).strip()
    return str(int(row["mouse_id"]))


def build_metadata_single(row: pd.Series, video_path: Path) -> dict:
    """Metadata pour une session 1-animal.

    Toutes les colonnes de l'Excel sont recopiées telles quelles (y
    compris `date` si tu en ajoutes une) — cf. la boucle générique en
    fin de fonction.
    """
    key = session_key_single(row)
    meta = {
        "session_id": f"BV-{key}",
        "project": "BottomView",
        "source_video": str(video_path),
    }
    if "mouse_id" in row.index and pd.notna(row["mouse_id"]):
        meta["mouse_id"] = int(row["mouse_id"])

    # 1) Champs connus d'abord (ordre stable et lisible dans le YAML)
    for col in META_FIELDS_SINGLE:
        if col in row.index:
            val = _coerce(row[col])
            if val is not None:
                meta[col] = val

    # 2) Puis TOUTE autre colonne présente dans l'Excel — l'utilisateur
    #    peut ajouter ses propres colonnes sans modifier le code, elles
    #    deviennent utilisables comme variables de groupement.
    for col in row.index:
        if col in _RESERVED_COLS or col in META_FIELDS_SINGLE:
            continue
        val = _coerce(row[col])
        if val is not None:
            meta[str(col)] = val
    return meta


def sync_single(excel: Path, videos_dir: Path, raw: Path,
                 video_ext: str, overwrite: bool, dry_run: bool) -> tuple[int, int, int]:
    df = parse_single(excel)
    print(f"Lignes Excel valides : {len(df)}\n")

    n_written = n_skipped = n_no_video = 0
    for _, row in df.iterrows():
        key = session_key_single(row)
        video_path = videos_dir / f"{key}.{video_ext}"
        session_id = f"BV-{key}"

        if not video_path.exists():
            print(f"  ✗ {session_id}: vidéo absente ({video_path.name})")
            n_no_video += 1
            continue

        session_dir = raw / session_id
        metadata_path = session_dir / "metadata.yaml"
        if metadata_path.exists() and not overwrite:
            print(f"  · {session_id}: metadata déjà existante, skip")
            n_skipped += 1
            continue

        meta = build_metadata_single(row, video_path)
        if dry_run:
            print(f"  [dry] {session_id}  group={meta.get('group')} "
                  f"→ {video_path.name}")
        else:
            session_dir.mkdir(parents=True, exist_ok=True)
            with open(metadata_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(meta, f, allow_unicode=True, sort_keys=False)
            print(f"  ✓ {session_id}  group={meta.get('group')} "
                  f"→ {video_path.name}")
        n_written += 1
    return n_written, n_skipped, n_no_video


# ======================================================================
# Schéma N animaux / vidéo
# ======================================================================

def fallback_from_codebook(mouse_id: int, timepoint: str) -> dict:
    """Règle de groupe par défaut documentée dans le Codebook de l'Excel.

    MouseIDs 1-10  : CUS  à M1, CUS+ANGII  à M2 (stress = oui)
    MouseIDs 11-18 : SHAM à M1, SHAM+ANGII à M2 (stress = non)

    Fallback quand la feuille Subjects renvoie NaN — typiquement si les
    formules Excel n'ont pas été ré-évaluées (édition via openpyxl).
    """
    is_stress = mouse_id <= 10
    if timepoint == "M1":
        condition = "CUS" if is_stress else "SHAM"
    else:
        condition = "CUS+ANGII" if is_stress else "SHAM+ANGII"
    return {"condition": condition, "stress": is_stress,
            "angii": timepoint == "M2"}


def derive_arena_info(mouse_id, timepoint: str,
                       subjects: pd.DataFrame) -> dict:
    """Dérive condition / ANGII / stress pour une souris à un timepoint."""
    if pd.isna(mouse_id):
        return {"mouse_id": None, "condition": None, "angii": None,
                "stress": None}

    mid = int(mouse_id)
    row = subjects[subjects["MouseID"] == mid] if "MouseID" in subjects.columns \
        else pd.DataFrame()

    condition = stress = None
    if not row.empty:
        col = ("Baseline group (M1)" if timepoint == "M1"
               else "ANGII group (M2)")
        if col in row.columns:
            raw_val = row[col].iloc[0]
            if pd.notna(raw_val):
                condition = str(raw_val)
        if "Stress (CUS?)" in row.columns:
            s_raw = row["Stress (CUS?)"].iloc[0]
            if pd.notna(s_raw):
                stress = str(s_raw).strip().lower() in ("yes", "oui", "true", "1")

    if condition is None or stress is None:
        fb = fallback_from_codebook(mid, timepoint)
        condition = condition if condition is not None else fb["condition"]
        stress = stress if stress is not None else fb["stress"]

    return {
        "mouse_id": mid,
        "condition": condition,
        "angii": isinstance(condition, str) and "ANGII" in condition,
        "stress": stress,
    }


def _col(df: pd.DataFrame, *candidates: str) -> str | None:
    """Retourne le premier nom de colonne présent parmi `candidates`.

    Tolère les variantes de nommage entre l'Excel généré par le template
    (`TrialCode`) et l'Excel historique du labo (`TrialCode (auto)`).
    """
    for c in candidates:
        if c in df.columns:
            return c
    return None


def build_metadata_multi(trial: pd.Series, subjects: pd.DataFrame,
                          arenas: pd.DataFrame, videos_dir: Path,
                          trial_col: str, arena_trial_col: str) -> dict:
    trial_code = trial[trial_col]
    timepoint = trial.get("Timepoint")

    date_raw = trial.get("Date (YYYY-MM-DD)", trial.get("Date"))
    if isinstance(date_raw, pd.Timestamp):
        date = date_raw.strftime("%Y-%m-%d")
    else:
        date = str(date_raw).split(" ")[0] if pd.notna(date_raw) else None

    # Vidéo : "Original file name" si renseigné, sinon <TrialCode>.mp4
    fname = trial.get("Original file name")
    if pd.notna(fname) and str(fname).strip():
        candidate = videos_dir / str(fname).strip()
        if not candidate.suffix:
            candidate = candidate.with_suffix(".mp4")
    else:
        candidate = videos_dir / f"{trial_code}.mp4"
    source_video = str(candidate.resolve()) if candidate.exists() else None

    trial_arenas = arenas[arenas[arena_trial_col] == trial_code]
    if "Arena" in trial_arenas.columns:
        trial_arenas = trial_arenas.sort_values("Arena")

    arena_list = []
    for _, ar in trial_arenas.iterrows():
        info = derive_arena_info(ar.get("MouseID"), timepoint, subjects)
        arena_list.append({
            "id": f"A{int(ar['Arena'])}",
            "coords": None,  # renseigné par calibrate_arenes.py
            **info,
        })

    meta = {
        "session_id": str(trial_code),
        "project": "OF",
        "timepoint": timepoint,
        "date": date,
        "source_video": source_video,
        "arenes": arena_list,
    }
    if pd.notna(trial.get("VideoNo")):
        meta["trial_no"] = _coerce(trial["VideoNo"])
    cam = {}
    for k, col in (("fps", "FPS"), ("width", "Width"), ("height", "Height")):
        if col in trial.index and pd.notna(trial[col]):
            cam[k] = int(trial[col])
    if cam:
        meta["camera"] = cam
    notes = trial.get("Notes")
    meta["notes"] = str(notes) if pd.notna(notes) else ""
    return meta


def write_metadata_multi(meta: dict, raw: Path, dry_run: bool) -> Path:
    """Écrit la metadata en préservant les overrides locaux."""
    session_dir = raw / meta["session_id"]
    target = session_dir / "metadata.yaml"
    if dry_run:
        return target

    # Préserve les overrides manuels (trims temporels, coords calibrées)
    if target.exists():
        with open(target) as f:
            existing = yaml.safe_load(f) or {}
        for key in ("start_time_s", "end_time_s"):
            if key in existing:
                meta[key] = existing[key]
        existing_coords = {
            a["id"]: a.get("coords")
            for a in existing.get("arenes", []) if a.get("coords")
        }
        for ar in meta["arenes"]:
            if existing_coords.get(ar["id"]):
                ar["coords"] = existing_coords[ar["id"]]

    session_dir.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8") as f:
        yaml.safe_dump(meta, f, allow_unicode=True, sort_keys=False)
    return target


def sync_multi(excel: Path, videos_dir: Path, raw: Path,
                dry_run: bool) -> tuple[int, int, int]:
    xl = pd.ExcelFile(excel)
    trials = xl.parse("Trials_Videos")
    arenas = xl.parse("Arena_Mapping")
    subjects = xl.parse("Subjects") if "Subjects" in xl.sheet_names \
        else pd.DataFrame()
    for df in (trials, arenas, subjects):
        if not df.empty:
            df.columns = [str(c).strip() for c in df.columns]

    trial_col = _col(trials, "TrialCode", "TrialCode (auto)")
    arena_trial_col = _col(arenas, "TrialCode", "TrialCode (auto)")
    if trial_col is None or arena_trial_col is None:
        raise ValueError(
            "Colonne 'TrialCode' introuvable dans Trials_Videos ou "
            "Arena_Mapping."
        )
    trials = trials.dropna(subset=[trial_col])

    print(f"  → {len(subjects)} sujets, {len(trials)} trials, "
          f"{len(arenas)} arènes\n")

    n_written = n_no_video = 0
    for _, trial in trials.iterrows():
        meta = build_metadata_multi(trial, subjects, arenas, videos_dir,
                                     trial_col, arena_trial_col)
        write_metadata_multi(meta, raw, dry_run)
        status = "✓" if meta["source_video"] else "✗ vidéo manquante"
        verb = "[dry]" if dry_run else "→"
        print(f"  {verb} {meta['session_id']}  {status} "
              f"({len(meta['arenes'])} arènes)")
        if not meta["source_video"]:
            n_no_video += 1
        n_written += 1
    return n_written, 0, n_no_video


# ======================================================================
# CLI
# ======================================================================

def find_project_excel(project: Path) -> Path | None:
    """Cherche l'Excel starter à la racine du projet."""
    candidates = sorted(project.glob("*_sessions.xlsx"))
    if candidates:
        return candidates[0]
    candidates = sorted(project.glob("*.xlsx"))
    return candidates[0] if len(candidates) == 1 else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--project-dir", type=Path, default=None,
                        help="Racine du projet EthoFlow. Demandé si absent.")
    parser.add_argument("--excel", type=Path, default=None,
                        help="Excel maître. Auto-détecté à la racine du "
                             "projet si absent, sinon demandé.")
    parser.add_argument("--videos-dir", type=Path, default=None,
                        help="Dossier contenant les .mp4. Demandé si absent.")
    parser.add_argument("--video-ext", default="mp4",
                        help="Extension vidéo à matcher (défaut : mp4)")
    parser.add_argument("--schema", choices=["single", "multi"], default=None,
                        help="Force le schéma au lieu de l'auto-détecter.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Écrase les metadata.yaml existants")
    parser.add_argument("--dry-run", action="store_true",
                        help="Affiche sans rien écrire")
    add_no_prompt_arg(parser)
    args = parser.parse_args()

    # ---- Projet ----
    project = resolve_or_prompt_project(args)

    # ---- Excel : auto-détection à la racine du projet ----
    excel = args.excel
    if excel is None:
        auto = find_project_excel(project)
        if auto is not None:
            excel = auto
            print(f"ℹ  Excel auto-détecté : {excel.name}")
        else:
            excel = prompt_existing_path(
                "Chemin de l'Excel maître", must_exist=True,
                no_prompt=args.no_prompt,
            )
    if not excel.exists():
        print(f"❌ Excel introuvable : {excel}", file=sys.stderr)
        sys.exit(1)

    # ---- Schéma ----
    try:
        schema = args.schema or detect_schema(excel)
    except ValueError as e:
        print(f"❌ {e}", file=sys.stderr)
        sys.exit(1)

    # ---- Dossier vidéos ----
    videos_dir = args.videos_dir
    if videos_dir is None:
        videos_dir = prompt_existing_path(
            "Dossier contenant les vidéos (.mp4)", must_exist=True,
            no_prompt=args.no_prompt,
        )
    if not videos_dir.exists():
        print(f"❌ Dossier vidéos introuvable : {videos_dir}", file=sys.stderr)
        sys.exit(1)

    raw = raw_dir(project)
    print()
    print(f"Projet     : {project}")
    print(f"Excel      : {excel}")
    print(f"Schéma     : {schema} "
          f"({'1 animal/vidéo' if schema == 'single' else 'N animaux/vidéo'})")
    print(f"Vidéos     : {videos_dir}")
    print(f"Sortie     : {raw}")
    print()

    try:
        if schema == "single":
            n_written, n_skipped, n_no_video = sync_single(
                excel, videos_dir, raw, args.video_ext,
                args.overwrite, args.dry_run,
            )
        else:
            n_written, n_skipped, n_no_video = sync_multi(
                excel, videos_dir, raw, args.dry_run,
            )
    except ValueError as e:
        print(f"\n❌ {e}", file=sys.stderr)
        sys.exit(1)

    print()
    verb = "à écrire" if args.dry_run else "écrits"
    print(f"✅ {n_written} metadata {verb}")
    if n_skipped:
        print(f"   {n_skipped} session(s) skip (déjà existante — relance "
              f"avec --overwrite pour écraser)")
    if n_no_video:
        print(f"⚠  {n_no_video} ligne(s) sans vidéo trouvée dans {videos_dir}")

    if not args.dry_run and n_written > 0:
        print()
        print("Étape suivante :")
        if schema == "single":
            print(f"  python scripts/run_dlc_inference.py "
                  f"--project-dir {project} --all --mode custom")
        else:
            print(f"  python scripts/run_pipeline.py "
                  f"--project-dir {project} --all")


if __name__ == "__main__":
    main()
