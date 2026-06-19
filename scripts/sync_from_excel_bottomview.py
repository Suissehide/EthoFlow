"""Sync sessions bottom-view depuis un Excel + un dossier de vidéos.

Schéma de l'Excel bottom-view (différent de l'Excel topview) :

    Une seule feuille 'OF' avec en-tête à la ligne 5 :
        cage | Numéromarquage | Groupe | N° queue | trial | remarque

    Une ligne = une souris. Numéromarquage est aussi le nom du fichier
    vidéo (961.mp4, 970.mp4, ...). Groupe = MCCiECKO ou MCCf/f.

    Une souris = une vidéo = une session. Pas de splitting d'arène
    (contrairement à topview où 4 souris cohabitent en 4 cadrans).

Comme tu as un Excel par journée et un dossier par journée, lance le
script une fois par batch :

    python scripts/sync_from_excel_bottomview.py \\
        --project-dir D:/ethoflow/projects/bottomview-MCC-2026-06 \\
        --excel "/chemin/MCC_femelles-openfield ... 20260605.xlsx" \\
        --videos-dir E:/data/bottom_view/08062026 \\
        --date 2026-06-08

    python scripts/sync_from_excel_bottomview.py \\
        --project-dir D:/ethoflow/projects/bottomview-MCC-2026-06 \\
        --excel "/chemin/MCC_femelles-openfield ... 20260605.xlsx" \\
        --videos-dir E:/data/bottom_view/11062026 \\
        --date 2026-06-11

Note : si le même Excel décrit les deux journées (même liste de souris),
tu peux le ré-utiliser pour le 2e batch. Le filtre se fait par présence
du fichier vidéo dans --videos-dir : seules les souris dont le `.mp4`
est trouvé donnent lieu à une session.

Schéma de metadata produit (minimaliste, single-animal) :

    session_id: BV-970
    project: BottomView
    mouse_id: 970
    group: MCCf/f
    cage: CD329
    tail_label: 1
    date_recorded: 2026-06-08
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


def parse_excel_OF(excel_path: Path) -> pd.DataFrame:
    """Lit la feuille 'OF' de l'Excel bottom-view.

    L'en-tête est à la ligne 5 (skiprows=4). Colonnes : un blanc puis
    cage, Numéromarquage, Groupe, N° queue, trial, remarque.

    Returns un DataFrame avec une ligne par souris labellisée
    (Numéromarquage non vide).
    """
    df = pd.read_excel(excel_path, sheet_name="OF", skiprows=4, header=0)
    # La 1ère colonne ('cage'/index numérique) peut être un blanc. Nomme-la.
    df.columns = [str(c).strip() for c in df.columns]
    # Vire les lignes sans Numéromarquage
    df = df.dropna(subset=["Numéromarquage"]).copy()
    df["Numéromarquage"] = df["Numéromarquage"].astype(int)
    return df


def build_metadata(
    row: pd.Series,
    video_path: Path,
    date_recorded: str | None,
) -> dict:
    """Construit le dict metadata pour une souris."""
    mouse_id = int(row["Numéromarquage"])
    session_id = f"BV-{mouse_id}"
    group = row.get("Groupe")
    cage = row.get("cage")
    tail = row.get("N° queue")
    notes = row.get("remarque")

    meta = {
        "session_id": session_id,
        "project": "BottomView",
        "mouse_id": mouse_id,
        "source_video": str(video_path),
    }
    if pd.notna(group):
        meta["group"] = str(group).strip()
    if pd.notna(cage) and str(cage).strip():
        meta["cage"] = str(cage).strip()
    if pd.notna(tail):
        # Cast en int quand possible
        try:
            meta["tail_label"] = int(tail)
        except (ValueError, TypeError):
            meta["tail_label"] = str(tail).strip()
    if date_recorded:
        meta["date_recorded"] = date_recorded
    if pd.notna(notes) and str(notes).strip():
        meta["notes"] = str(notes).strip()
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

    df = parse_excel_OF(args.excel)
    print(f"Lignes Excel valides : {len(df)}\n")

    n_written = 0
    n_skipped_existing = 0
    n_no_video = 0
    for _, row in df.iterrows():
        mouse_id = int(row["Numéromarquage"])
        video_path = args.videos_dir / f"{mouse_id}.{args.video_ext}"
        session_id = f"BV-{mouse_id}"

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
