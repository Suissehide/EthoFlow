"""
Tronque les frames empty-arena au début (et optionnellement en fin) de chaque
session, en miroir sur les .h5 ET les .mp4 — pour que VAME puisse être
re-segmenté sur des données propres SANS avoir à retrainer le VAE.

Lit validity_per_session.csv (produit par `analyze_vame.py --validity-source`)
qui contient n_empty_start et n_empty_end par session_full.

Pour chaque session :
  - Si n_empty_start + n_empty_end == 0 : copie le h5 et la mp4 tels quels.
  - Sinon : drop des lignes du h5 et trim de la vidéo via opencv
    (frame-accurate). Les comptes finaux h5 vs vidéo sont vérifiés.

La structure de sortie miroite l'entrée :
  <h5-output>/<session_id>/<session_full>.h5
  <video-output>/<session_id>/<session_full>.mp4

Usage:
    python scripts/trim_empty_arena.py \\
        --validity-csv vame-projects/OF-single-enhanced-2026-05/analysis/validity_per_session.csv \\
        --h5-input data/vame-input/single-enhanced-2026-05-clean \\
        --h5-output data/vame-input/single-enhanced-2026-05-trimmed \\
        --video-input data/cropped \\
        --video-output data/cropped-trimmed

Workflow VAME complet ensuite (sans retrain) :

    1) Setup d'un nouveau projet sur les données trimées :
       python scripts/run_vame.py setup \\
           --input-dir data/vame-input/single-enhanced-2026-05-trimmed \\
           --cropped-dir data/cropped-trimmed \\
           --project-name OF-single-enhanced-trimmed-2026-05

    2) Recopie le modèle VAE entraîné depuis l'ancien projet (Windows) :
       xcopy /E /Y /I "vame-projects\\OF-single-enhanced-2026-05\\model" ^
                       "vame-projects\\OF-single-enhanced-trimmed-2026-05\\model"

    3) align → segment → motif-videos → community (pas de retrain nécessaire) :
       python scripts/run_vame.py align
       python scripts/run_vame.py segment
       python scripts/run_vame.py motif-videos
       python scripts/run_vame.py community

    4) Analyse (sans --validity-source ni --mask-empty cette fois,
       puisque les données sont déjà propres) :
       python scripts/analyze_vame.py
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import cv2
import pandas as pd
import tables  # noqa: F401  — pytables requis pour pd.read_hdf format='table'


def trim_h5(input_path: Path, output_path: Path,
            n_skip_start: int, n_skip_end: int) -> tuple[int, int]:
    """Drop les n_skip_start premières et n_skip_end dernières lignes du h5.

    Retourne (n_avant, n_après) en nombre de lignes.
    """
    df = pd.read_hdf(input_path)
    n = len(df)
    end = n - n_skip_end if n_skip_end > 0 else n
    df_trimmed = df.iloc[n_skip_start:end].copy()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # key='df_with_missing' est la convention DLC attendue par VAME.
    df_trimmed.to_hdf(output_path, key="df_with_missing",
                      mode="w", format="table")
    return n, len(df_trimmed)


def trim_video(input_path: Path, output_path: Path,
               n_skip_start: int, n_skip_end: int) -> tuple[int, int]:
    """Trim frame-accurate avec opencv.

    On lit séquentiellement les frames du début, on ignore les n_skip_start
    premières, on écrit les suivantes jusqu'à n_total - n_skip_end. Évite
    les approximations du seeking ffmpeg.

    Retourne (n_total_origine, n_après_trim).
    """
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Impossible d'ouvrir {input_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n_keep = n_total - n_skip_start - n_skip_end
    if n_keep <= 0:
        cap.release()
        raise RuntimeError(
            f"{input_path} : trim trop large "
            f"({n_skip_start}+{n_skip_end} sur {n_total} frames)"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Impossible d'ouvrir en écriture {output_path}")

    # Skip les n_skip_start premières (grab sans decode pour aller plus vite)
    for _ in range(n_skip_start):
        if not cap.grab():
            cap.release()
            writer.release()
            raise RuntimeError(
                f"Lecture interrompue prématurément dans {input_path}"
            )

    written = 0
    for _ in range(n_keep):
        ok, frame = cap.read()
        if not ok:
            break
        writer.write(frame)
        written += 1

    cap.release()
    writer.release()
    return n_total, written


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tronque h5 + vidéos selon validity_per_session.csv"
    )
    parser.add_argument("--validity-csv", type=Path, required=True,
                        help="CSV produit par analyze_vame.py --validity-source.")
    parser.add_argument("--h5-input", type=Path, required=True,
                        help="Dossier d'entrée des .h5 (ex : "
                             "data/vame-input/single-enhanced-2026-05-clean).")
    parser.add_argument("--h5-output", type=Path, required=True,
                        help="Dossier de sortie des .h5 trimés.")
    parser.add_argument("--video-input", type=Path, required=True,
                        help="Dossier d'entrée des .mp4 (ex : data/cropped).")
    parser.add_argument("--video-output", type=Path, required=True,
                        help="Dossier de sortie des .mp4 trimés.")
    parser.add_argument("--dry-run", action="store_true",
                        help="N'écrit rien, affiche juste ce qui serait fait.")
    args = parser.parse_args()

    df_v = pd.read_csv(args.validity_csv)
    needed = {"session_full", "n_empty_start", "n_empty_end"}
    missing = needed - set(df_v.columns)
    if missing:
        print(f"❌ Colonnes manquantes dans {args.validity_csv}: {missing}",
              file=sys.stderr)
        sys.exit(1)
    validity = df_v.set_index("session_full")[
        ["n_empty_start", "n_empty_end"]
    ].to_dict("index")

    h5_files = sorted(args.h5_input.rglob("*.h5"))
    if not h5_files:
        print(f"❌ Aucun .h5 trouvé dans {args.h5_input}", file=sys.stderr)
        sys.exit(1)
    print(f"Trouvé {len(h5_files)} fichier(s) h5 dans {args.h5_input}")
    print(f"  → sortie h5    : {args.h5_output}")
    print(f"  → sortie vidéo : {args.video_output}")
    if args.dry_run:
        print("  [DRY RUN — rien ne sera écrit]")
    print()

    n_trimmed = n_copied = n_skipped = 0
    for h5 in h5_files:
        session_full = h5.stem            # "OF-M1-20251010-V01_A1"
        session_id = h5.parent.name       # "OF-M1-20251010-V01"
        v = validity.get(session_full)
        if v is None:
            n_skip_start = n_skip_end = 0
        else:
            n_skip_start = int(v["n_empty_start"])
            n_skip_end = int(v["n_empty_end"])

        h5_out = args.h5_output / session_id / h5.name
        video_in = args.video_input / session_id / f"{session_full}.mp4"
        video_out = args.video_output / session_id / f"{session_full}.mp4"

        if not video_in.exists():
            print(f"  ⚠️  {session_full} : vidéo introuvable {video_in} — skip")
            n_skipped += 1
            continue

        if n_skip_start + n_skip_end == 0:
            print(f"  · {session_full} : copie (pas de trim)")
            if not args.dry_run:
                h5_out.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(h5, h5_out)
                video_out.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(video_in, video_out)
            n_copied += 1
        else:
            print(f"  ✂  {session_full} : skip start={n_skip_start}, "
                  f"end={n_skip_end}")
            if not args.dry_run:
                n_before_h5, n_after_h5 = trim_h5(
                    h5, h5_out, n_skip_start, n_skip_end
                )
                n_before_v, n_after_v = trim_video(
                    video_in, video_out, n_skip_start, n_skip_end
                )
                print(f"      h5    : {n_before_h5} → {n_after_h5} frames")
                print(f"      vidéo : {n_before_v} → {n_after_v} frames")
                if n_after_h5 != n_after_v:
                    print(f"      ⚠️  mismatch h5 ({n_after_h5}) vs vidéo "
                          f"({n_after_v}) — vérifier manuellement !")
            n_trimmed += 1

    summary = f"{n_trimmed} trimées, {n_copied} copiées tel quel"
    if n_skipped:
        summary += f", {n_skipped} skip"
    print(f"\n✅ Terminé : {summary}.")
    print(f"   Sorties : {args.h5_output} + {args.video_output}")
    if args.dry_run:
        print("   (DRY RUN — rien n'a été écrit, relance sans --dry-run)")


if __name__ == "__main__":
    main()
