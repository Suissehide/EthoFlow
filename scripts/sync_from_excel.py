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
    --excel       : ../data/OpenField_trials_CDUPLAA.xlsx
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
DEFAULT_EXCEL = ROOT.parent / "data" / "OpenField_trials_CDUPLAA.xlsx"
DEFAULT_VIDEOS = ROOT.parent / "data"


def fallback_from_codebook(mouse_id: int, timepoint: str) -> dict:
    """
    Règle de groupe par défaut, documentée dans le Codebook de l'Excel :
        MouseIDs 1–10  : CUS   à M1, CUS+ANGII   à M2 (stress = oui)
        MouseIDs 11–18 : SHAM  à M1, SHAM+ANGII à M2 (stress = non)

    Utilisée comme fallback quand la feuille Subjects renvoie NaN — par exemple
    si les formules `=IF(A2<=10,"CUS","SHAM")` n'ont pas été ré-évaluées par
    Excel (cas typique : édition via openpyxl sans recalcul).
    """
    is_stress = mouse_id <= 10
    if timepoint == "M1":
        condition = "CUS" if is_stress else "SHAM"
    else:
        condition = "CUS+ANGII" if is_stress else "SHAM+ANGII"
    return {"condition": condition, "stress": is_stress, "angii": timepoint == "M2"}


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

    # Tentative 1 : lire la valeur depuis la feuille Subjects
    condition = None
    stress = None
    if not row.empty:
        col = "Baseline group (M1)" if timepoint == "M1" else "ANGII group (M2)"
        condition_raw = row[col].iloc[0]
        if pd.notna(condition_raw):
            condition = str(condition_raw)

        stress_raw = row["Stress (CUS?)"].iloc[0]
        if pd.notna(stress_raw):
            stress = str(stress_raw).strip().lower() == "yes"

    # Tentative 2 : fallback sur la règle du Codebook si une info manque
    if condition is None or stress is None:
        fb = fallback_from_codebook(mid, timepoint)
        if condition is None:
            condition = fb["condition"]
        if stress is None:
            stress = fb["stress"]

    angii = isinstance(condition, str) and "ANGII" in condition

    return {
        "mouse_id": mid,
        "condition": condition,
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

    # Préserver les overrides locaux ajoutés manuellement (start_time_s,
    # end_time_s, coords par arène, etc.) qu'on ne veut pas perdre au resync.
    if target.exists():
        with open(target) as f:
            existing = yaml.safe_load(f) or {}
        for key in ("start_time_s", "end_time_s"):
            if key in existing:
                metadata[key] = existing[key]
        # Préserver les coords par arène si elles ont été overridées localement
        existing_coords = {
            a["id"]: a.get("coords")
            for a in existing.get("arenes", [])
            if a.get("coords")
        }
        for ar in metadata["arenes"]:
            if existing_coords.get(ar["id"]):
                ar["coords"] = existing_coords[ar["id"]]

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
