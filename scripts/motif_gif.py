"""Génère une vidéo (ou GIF) annotée avec les motifs comportementaux VAME.

Reproduit le style de la démo VAME du README GitHub : vidéo source de la souris
avec, superposé en bas, un bandeau color-coded qui indique le motif à chaque
frame. À gauche du bandeau, le nom du motif courant.

Approche :
    - Lit le fichier `label_<session>.npy` produit par `run_vame.py segment`
      (un entier par frame).
    - Assigne une couleur unique par motif (palette matplotlib.tab20).
    - Pour chaque frame de la vidéo :
        - dessine le bandeau (petit rectangle plein en bas, largeur = frame,
          hauteur ~40 px) avec la couleur du motif courant
        - marque la position temporelle par un curseur vertical qui balaye le
          bandeau au fil du temps
        - overlay le nom du motif en haut à gauche (utile pour interprétation
          en revisionnage lent)
    - Écrit le résultat en .mp4 (recommandé, meilleure compression) ou .gif
      (partage web direct — mais taille ~10x plus grosse).

Le GIF peut atteindre plusieurs Go pour 20 min de vidéo. Recommandation : ne
générer qu'un extrait (60-120 s max) pour les GIF ; garder le .mp4 complet
pour l'archivage.

Pré-requis :
    - conda activate vame  (ou ethoflow) — a besoin de cv2, numpy, pandas
    - la segmentation VAME a été lancée (label_<session>.npy existe)

Usage :
    # Vidéo complète (20 min → mp4 de ~200 Mo)
    python scripts/motif_gif.py \\
        --project-dir D:/ethoflow/projects/bottomview-MCC-2026-06 \\
        --session BV-970

    # Extrait de 90 s en GIF pour partage
    python scripts/motif_gif.py \\
        --project-dir <...> \\
        --session BV-970 \\
        --start 120 --duration 90 --output-format gif

    # Avec labels descriptifs depuis motif_labels.csv
    python scripts/motif_gif.py \\
        --project-dir <...> --session BV-970 \\
        --labels D:/ethoflow/projects/.../data/vame/motif_labels.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from interactive import prompt_session  # noqa: E402
from paths import add_project_dir_arg, raw_dir, resolve_project, vame_dir  # noqa: E402


# Palette 20 couleurs (matplotlib tab20 en RGB uint8, ordre stable)
TAB20 = [
    (31, 119, 180), (255, 127, 14), (44, 160, 44), (214, 39, 40),
    (148, 103, 189), (140, 86, 75), (227, 119, 194), (127, 127, 127),
    (188, 189, 34), (23, 190, 207), (174, 199, 232), (255, 187, 120),
    (152, 223, 138), (255, 152, 150), (197, 176, 213), (196, 156, 148),
    (247, 182, 210), (199, 199, 199), (219, 219, 141), (158, 218, 229),
]


def find_label_file(vame_project: Path, session: str,
                     algo: str = "hmm") -> Path | None:
    """Trouve label_<session>.npy. Convention VAME :
       <vame>/results/<session>/<model>/<algo>-<n>/<n>_<algo>_label_<session>.npy
    """
    results = vame_project / "results" / session
    if not results.exists():
        return None
    for algo_dir in results.rglob(f"{algo}-*"):
        for f in algo_dir.glob(f"*_{algo}_label_{session}.npy"):
            return f
    return None


def load_labels_dict(labels_path: Path | None) -> dict[int, str]:
    """Load motif_labels (CSV ou YAML) → {int: nom court}. Retourne {} sinon."""
    if labels_path is None or not labels_path.exists():
        return {}
    import pandas as pd
    if labels_path.suffix.lower() in (".yaml", ".yml"):
        import yaml
        with open(labels_path) as f:
            raw = yaml.safe_load(f) or {}
        return {int(k): str(v) for k, v in raw.items()}
    # CSV : auto-détecte séparateur + fallback label→category si label vide
    with open(labels_path, encoding="utf-8") as f:
        first = f.readline()
    sep = ";" if first.count(";") > first.count(",") else ","
    df = pd.read_csv(labels_path, sep=sep, dtype=str, keep_default_na=False)
    df.columns = [c.strip() for c in df.columns]
    out = {}
    for _, row in df.iterrows():
        try:
            mid = int(row.get("motif_id", "").strip())
        except (TypeError, ValueError):
            continue
        label = row.get("label", "").strip() or row.get("category", "").strip()
        if label:
            out[mid] = label
    return out


def get_source_video(project_ethoflow: Path, session: str) -> Path | None:
    import yaml
    meta_path = raw_dir(project_ethoflow) / session / "metadata.yaml"
    if not meta_path.exists():
        return None
    with open(meta_path) as f:
        meta = yaml.safe_load(f) or {}
    src = meta.get("source_video")
    if not src:
        return None
    return Path(src)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    add_project_dir_arg(parser, required=True)
    parser.add_argument("--session", default=None,
                        help="Session ID (ex: BV-970)")
    parser.add_argument("--algo", default="hmm", choices=["hmm", "kmeans"])
    parser.add_argument("--labels", type=Path, default=None,
                        help="motif_labels.csv (auto: <vame>/motif_labels.csv)")
    parser.add_argument("--start", type=float, default=0.0,
                        help="Début en secondes (défaut 0)")
    parser.add_argument("--duration", type=float, default=None,
                        help="Durée en secondes (défaut : toute la vidéo)")
    parser.add_argument("--output-format", choices=["mp4", "gif"], default="mp4",
                        help="mp4 pour archivage / analyse, gif pour partage web "
                             "(garde --duration < 120 s pour un gif raisonnable)")
    parser.add_argument("--strip-height", type=int, default=40,
                        help="Hauteur du bandeau color-coded en pixels (défaut 40)")
    parser.add_argument("--font-scale", type=float, default=0.7,
                        help="Taille du texte OSD (défaut 0.7)")
    args = parser.parse_args()

    try:
        import cv2
    except ImportError:
        print("❌ opencv-python requis. pip install opencv-python", file=sys.stderr)
        sys.exit(1)

    project = resolve_project(args)
    args.session = prompt_session(
        project, args.session,
        no_prompt=getattr(args, "no_prompt", False),
        title="Quelle session pour le GIF ?",
    )
    vame_proj = vame_dir(project)
    if not (vame_proj / "config.yaml").exists():
        print(f"❌ Projet VAME introuvable : {vame_proj}", file=sys.stderr)
        sys.exit(1)

    label_file = find_label_file(vame_proj, args.session, args.algo)
    if label_file is None:
        print(f"❌ label_{args.session}.npy introuvable dans "
              f"{vame_proj / 'results' / args.session}", file=sys.stderr)
        sys.exit(1)
    labels_per_frame = np.load(label_file).astype(int)

    src_video = get_source_video(project, args.session)
    if src_video is None or not src_video.exists():
        print(f"❌ Vidéo source introuvable pour {args.session}", file=sys.stderr)
        sys.exit(1)

    # Load motif label names
    if args.labels is None:
        default = vame_proj / "motif_labels.csv"
        if default.exists():
            args.labels = default
    label_names = load_labels_dict(args.labels)

    # Open video
    cap = cv2.VideoCapture(str(src_video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    start_frame = int(args.start * fps)
    end_frame = (int((args.start + args.duration) * fps)
                 if args.duration else total_frames)
    end_frame = min(end_frame, total_frames, len(labels_per_frame))
    n_out = end_frame - start_frame
    if n_out <= 0:
        print("❌ Intervalle vide.", file=sys.stderr)
        sys.exit(1)

    # Prepare output
    strip_h = args.strip_height
    out_H = H + strip_h
    out_dir = vame_proj / "analysis" / "motif_gifs"
    out_dir.mkdir(parents=True, exist_ok=True)

    suffix = f"_{args.start:.0f}s_{args.duration:.0f}s" if args.duration else ""
    out_stem = f"{args.session}_annotated{suffix}"

    if args.output_format == "mp4":
        out_path = out_dir / f"{out_stem}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (W, out_H))
    else:
        # Pour le GIF, on écrit d'abord en mp4 temporaire puis on convertit
        out_path = out_dir / f"{out_stem}.gif"
        tmp_mp4 = out_dir / f"{out_stem}_tmp.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(tmp_mp4), fourcc, fps, (W, out_H))

    print(f"Session       : {args.session}")
    print(f"Vidéo source  : {src_video.name}")
    print(f"Labels        : {label_file.name}")
    print(f"Intervalle    : {args.start:.0f} s → {args.start + n_out/fps:.0f} s "
          f"({n_out} frames)")
    print(f"Motifs uniques dans l'extrait : "
          f"{sorted(np.unique(labels_per_frame[start_frame:end_frame]).tolist())}")
    print(f"Sortie        : {out_path}\n")

    # Pré-calcule les positions du curseur dans le bandeau pour n_out frames
    # (mapping linéaire frame_i → x_pixel dans le bandeau)
    cursor_positions = np.linspace(0, W, n_out).astype(int)

    # Pré-rend le bandeau une seule fois (constant sur toute la vidéo)
    strip_template = np.zeros((strip_h, W, 3), dtype=np.uint8)
    labels_extract = labels_per_frame[start_frame:end_frame]
    for i, motif in enumerate(labels_extract):
        color = TAB20[int(motif) % len(TAB20)]
        # BGR pour cv2
        strip_template[:, cursor_positions[i]:cursor_positions[i] + 2] = color[::-1]

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    for i in range(n_out):
        ret, frame = cap.read()
        if not ret:
            break
        current_motif = int(labels_extract[i])
        color_bgr = TAB20[current_motif % len(TAB20)][::-1]

        # Assemble : image + bandeau
        canvas = np.zeros((out_H, W, 3), dtype=np.uint8)
        canvas[:H] = frame
        canvas[H:] = strip_template

        # Curseur vertical (position temporelle courante)
        cv2.line(canvas, (cursor_positions[i], H), (cursor_positions[i], out_H),
                 (255, 255, 255), 2)

        # Texte OSD : nom du motif
        label_txt = label_names.get(current_motif, f"motif_{current_motif}")
        # Rectangle noir semi-transparent sous le texte pour lisibilité
        cv2.rectangle(canvas, (5, 5), (450, 40), (0, 0, 0), -1)
        cv2.putText(canvas, f"[{current_motif}] {label_txt}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, args.font_scale,
                    color_bgr, 2, cv2.LINE_AA)

        writer.write(canvas)

        if i % 500 == 0 and i > 0:
            print(f"  {i}/{n_out} ({100*i/n_out:.0f}%)")

    cap.release()
    writer.release()

    if args.output_format == "gif":
        print("\nConversion mp4 → gif via ffmpeg...")
        import subprocess
        # Réduit la taille du gif : downscale à ~640 px + 15 fps
        palette = out_dir / f"{out_stem}_palette.png"
        subprocess.run([
            "ffmpeg", "-y", "-i", str(tmp_mp4),
            "-vf", "fps=15,scale=640:-1:flags=lanczos,palettegen",
            str(palette),
        ], check=True)
        subprocess.run([
            "ffmpeg", "-y", "-i", str(tmp_mp4), "-i", str(palette),
            "-filter_complex", "fps=15,scale=640:-1:flags=lanczos[x];[x][1:v]paletteuse",
            str(out_path),
        ], check=True)
        palette.unlink(missing_ok=True)
        tmp_mp4.unlink(missing_ok=True)

    print(f"\n✅ Sortie : {out_path}")
    if args.output_format == "gif":
        size_mb = out_path.stat().st_size / 1e6
        print(f"   Taille : {size_mb:.1f} MB")
        if size_mb > 20:
            print(f"   ⚠ >20MB : envisage --duration plus court ou --output-format mp4")


if __name__ == "__main__":
    main()
