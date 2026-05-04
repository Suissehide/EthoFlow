"""
Synchronise les sessions depuis le fichier Excel maître.

Lit le tableur (Trials_Videos + Subjects + Arena_Mapping) et génère un
metadata.yaml par session dans data/raw/<TrialCode>/.

Le metadata contient toutes les infos nécessaires au pipeline :
- identifiant de session
- timepoint, date, projet
- chemin absolu vers la vidéo source
- specs caméra
- pour chaque arène : MouseID + condition dérivée (CUS / SHAM, ANGII, stress)

Usage:
    python scripts/sync_from_excel.py
    python scripts/sync_from_excel.py --excel /chemin/vers/fichier.xlsx
    python scripts/sync_from_excel.py --videos-dir /chemin/vers/videos/
    python scripts/sync_from_excel.py --dry-run    # affiche sans écrire

Defaults:
    --excel       : ../data/OpenField_trials_C DUPLAA.xlsx (frère du repo)
    --videos-dir  : ../data/                                (.mp4 à plat)

Pré-requis :
    pip install pandas openpyxl pyyaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
DEFAULT_EXCEL = ROOT.parent / "data" / "OpenField_trials_C DUPLAA.xlsx"
DEFAULT_VIDEOS = ROOT.parent / "data"


def derive_arena_info(mouse_id, timepoint: str, subjects: pd.DataFrame) -> dict:
    """Dérive condition / ANGII / stress pour une souris à un timepoint."""
    if pd.isna(mouse_id):
        return {
            "mouse_id": None,
            "condition": None,
            "angii": None,
            "stress": None,
        }

    mid = int(mouse_id)
    row = subjects[subjects["MouseID"] == mid]
    if row.empty:
        return {"mouse_id": mid, "condition": None, "angii": None, "stress": None}

    if timepoint == "M1":
        condition = row["Baseline group (M1)"].iloc[0]
    else:
        condition = row["ANGII group (M2)"].iloc[0]

    stress_raw = row["Stress (CUS?)"].iloc[0]
    stress = (str(stress_raw).strip().lower() == "yes")
    angii = isinstance(condition, str) and "ANGII" in condition

    return {
        "mouse_id": mid,
        "condition": str(condition) if pd.notna(condition) else None,
        "angii": angii,
        "stress": stress,
    }


def build_metadata(
    trial: pd.Series,
    subjects: pd.DataFrame,
    arenas: pd.DataFrame,
    videos_dir: Path,
) -> dict:
    trial_code = trial["TrialCode (auto)"]
    timepoint = trial["Timepoint"]

    # Date — peut être un Timestamp pandas, on prend la partie YYYY-MM-DD
    date_raw = trial["Date (YYYY-MM-DD)"]
    if isinstance(date_raw, pd.Timestamp):
        date = date_raw.strftime("%Y-%m-%d")
    else:
        date = str(date_raw).split(" ")[0]

    # Localiser la vidéo (à plat dans videos_dir, nom = TrialCode + .mp4)
    video_path = videos_dir / f"{trial_code}.mp4"
    source_video = str(video_path.resolve()) if video_path.exists() else None

    # Arènes pour ce trial
    trial_arenas = arenas[arenas["TrialCode"] == trial_code].sort_values("Arena")
    arena_list = []
    for _, ar in trial_arenas.iterrows():
        info = derive_arena_info(ar.get("MouseID"), timepoint, subjects)
        arena_id = f"A{int(ar['Arena'])}"
        mouse_trial_code = ar.get("MouseTrialCode (auto)")
        arena_list.append({
            "id": arena_id,
            "coords": None,  # à définir une fois la géométrie de la grille connue
            "mouse_trial_code": (
                str(mouse_trial_code) if pd.notna(mouse_trial_code) else None
            ),
            **info,
        })

    notes_raw = trial.get("Notes")
    notes = str(notes_raw) if pd.notna(notes_raw) else ""

    return {
        "session_id": trial_code,
        "project": "OF",
        "timepoint": timepoint,
        "date": date,
        "trial_no": int(trial["VideoNo"]),
        "source_video": source_video,
        "camera": {
            "fps": int(trial["FPS"]),
            "width": int(trial["Width"]),
            "height": int(trial["Height"]),
        },
        "arenes": arena_list,
        "notes": notes,
    }


def write_metadata(metadata: dict, dry_run: bool = False) -> Path:
    session_id = metadata["session_id"]
    session_dir = DATA_RAW / session_id
    target = session_dir / "metadata.yaml"

    if dry_run:
        return target

    session_dir.mkdir(parents=True, exist_ok=True)
    with open(target, "w") as f:
        yaml.dump(
            metadata, f,
            allow_unicode=True,
            sort_keys=False,
            default_flow_style=False,
        )
    return target


def main():
    parser = argparse.ArgumentParser(description="Sync sessions depuis Excel.")
    parser.add_argument("--excel", type=Path, default=DEFAULT_EXCEL,
                        help=f"Chemin du fichier Excel (défaut: {DEFAULT_EXCEL})")
    parser.add_argument("--videos-dir", type=Path, default=DEFAULT_VIDEOS,
                        help=f"Dossier des .mp4 (défaut: {DEFAULT_VIDEOS})")
    parser.add_argument("--dry-run", action="store_true",
                        help="Affiche sans écrire de fichier")
    args = parser.parse_args()

    if not args.excel.exists():
        print(f"❌ Excel introuvable : {args.excel}", file=sys.stderr)
        sys.exit(1)

    print(f"Lecture : {args.excel}")
    xl = pd.ExcelFile(args.excel)
    required = {"Subjects", "Trials_Videos", "Arena_Mapping"}
    missing = required - set(xl.sheet_names)
    if missing:
        print(f"❌ Onglets manquants : {missing}", file=sys.stderr)
        sys.exit(1)

    subjects = xl.parse("Subjects")
    trials = xl.parse("Trials_Videos")
    arenas = xl.parse("Arena_Mapping")

    print(f"  → {len(subjects)} sujets, {len(trials)} trials, {len(arenas)} arènes")
    print(f"Vidéos cherchées dans : {args.videos_dir}")
    print()

    n_written, n_no_video = 0, 0
    for _, trial in trials.iterrows():
        meta = build_metadata(trial, subjects, arenas, args.videos_dir)
        target = write_metadata(meta, dry_run=args.dry_run)

        video_status = "✓" if meta["source_video"] else "✗ vidéo manquante"
        verb = "[dry]" if args.dry_run else "→"
        print(f"  {verb} {meta['session_id']}  {video_status}")

        if not meta["source_video"]:
            n_no_video += 1
        n_written += 1

    print()
    print(f"✅ {n_written} metadata{'s' if n_written > 1 else ''} "
          f"{'à écrire' if args.dry_run else 'écrits'} dans {DATA_RAW}")
    if n_no_video:
        print(f"⚠️  {n_no_video} session(s) sans vidéo localisée — vérifier --videos-dir")


if __name__ == "__main__":
    main()
