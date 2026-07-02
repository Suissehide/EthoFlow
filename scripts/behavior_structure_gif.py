"""Reproduit la visu "behavior manifold" du README VAME.

Le GIF `behavior_structure_crop.gif` du repo LINCellularNeuroscience/VAME
montre la projection 2D des latents du VAE (UMAP ou PCA), avec :

    - Tous les frames d'une session tracés comme des points en background,
      color-codés par motif (l'espace latent est structuré : les frames
      qui correspondent au même comportement se regroupent en clusters
      distincts, matérialisant la « structure comportementale »).
    - Un marqueur animé qui suit la position courante de la souris dans
      cet espace latent au fil du temps. Une queue (trailing trajectory)
      permet de voir les 1-2 dernières secondes de trajectoire, révélant
      comment la souris navigue entre motifs.

Ce visuel est parlant pour deux choses :
    - Argumenter que VAME a appris quelque chose (clusters distincts ==
      motifs bien séparés dans l'espace latent).
    - Montrer que le comportement est structuré (la souris ne « saute »
      pas au hasard, elle suit des chemins récurrents).

Données requises (produites par VAME) :
    <vame_project>/results/<session>/<model>/latent_vector_<session>.npy
    <vame_project>/results/<session>/<model>/<algo>-<n>/<n>_<algo>_label_<session>.npy

Pré-requis :
    - conda activate vame (ou ethoflow avec numpy+matplotlib)
    - sklearn (PCA) et optionnellement umap-learn (meilleure projection).
      Si umap-learn absent → fallback PCA silencieux.

Usage :
    # GIF complet (session de 20 min → GIF long, on downsample)
    python scripts/behavior_structure_gif.py \\
        --project-dir D:/ethoflow/projects/bottomview-MCC-2026-06 \\
        --session BV-970

    # Extrait de 60 s (moins de frames = GIF léger)
    python scripts/behavior_structure_gif.py \\
        --project-dir <...> --session BV-970 \\
        --start 120 --duration 60

    # Force PCA au lieu de UMAP (plus rapide, moins joli)
    python scripts/behavior_structure_gif.py \\
        --project-dir <...> --session BV-970 --projection pca

    # Sortie MP4 au lieu de GIF (meilleur ratio taille/qualité)
    python scripts/behavior_structure_gif.py \\
        --project-dir <...> --session BV-970 --output-format mp4
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import add_project_dir_arg, resolve_project, vame_dir  # noqa: E402


# Même palette que motif_gif.py pour cohérence
TAB20 = np.array([
    (31, 119, 180), (255, 127, 14), (44, 160, 44), (214, 39, 40),
    (148, 103, 189), (140, 86, 75), (227, 119, 194), (127, 127, 127),
    (188, 189, 34), (23, 190, 207), (174, 199, 232), (255, 187, 120),
    (152, 223, 138), (255, 152, 150), (197, 176, 213), (196, 156, 148),
    (247, 182, 210), (199, 199, 199), (219, 219, 141), (158, 218, 229),
]) / 255.0  # matplotlib attend des couleurs 0-1


def find_latent_vector(vame_project: Path, session: str) -> Path | None:
    """Trouve le fichier de latent vectors pour cette session.

    Deux conventions VAME possibles selon la version :
    - VAME 0.11 : results/<session>/<model>/latent_vector_<session>.npy
    - VAME 0.13+ : results/<session>/<model>/latent_vectors.npy (pluriel,
      pas de nom de session dans le fichier)
    """
    results = vame_project / "results" / session
    if not results.exists():
        return None
    # Nouveau format d'abord (plus courant)
    for f in results.rglob("latent_vectors.npy"):
        return f
    # Legacy avec session dans le nom
    for f in results.rglob(f"latent_vector_{session}.npy"):
        return f
    return None


def find_label_file(vame_project: Path, session: str,
                     algo: str = "hmm") -> Path | None:
    """Trouve label_<session>.npy (partagé avec motif_gif.py)."""
    results = vame_project / "results" / session
    if not results.exists():
        return None
    for algo_dir in results.rglob(f"{algo}-*"):
        for f in algo_dir.glob(f"*_{algo}_label_{session}.npy"):
            return f
    return None


def load_motif_names(labels_csv: Path | None) -> dict[int, str]:
    """Load short motif names from CSV/YAML if provided."""
    if labels_csv is None or not labels_csv.exists():
        return {}
    import pandas as pd
    if labels_csv.suffix.lower() in (".yaml", ".yml"):
        import yaml
        with open(labels_csv) as f:
            raw = yaml.safe_load(f) or {}
        return {int(k): str(v) for k, v in raw.items()}
    with open(labels_csv, encoding="utf-8") as f:
        first = f.readline()
    sep = ";" if first.count(";") > first.count(",") else ","
    df = pd.read_csv(labels_csv, sep=sep, dtype=str, keep_default_na=False)
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


def project_to_2d(latents: np.ndarray, method: str) -> np.ndarray:
    """Projette (N, D) → (N, 2) via UMAP ou PCA."""
    if method == "umap":
        try:
            import umap  # type: ignore
            print(f"  UMAP fitting sur {len(latents)} points...")
            reducer = umap.UMAP(
                n_components=2, n_neighbors=30, min_dist=0.3,
                random_state=42, low_memory=True,
            )
            return reducer.fit_transform(latents)
        except ImportError:
            print("  umap-learn non installé, fallback sur PCA",
                  file=sys.stderr)
            method = "pca"
    # PCA
    from sklearn.decomposition import PCA
    print(f"  PCA fitting sur {len(latents)} points...")
    return PCA(n_components=2, random_state=42).fit_transform(latents)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    add_project_dir_arg(parser, required=True)
    parser.add_argument("--session", required=True,
                        help="Session ID (ex: BV-970)")
    parser.add_argument("--algo", default="hmm", choices=["hmm", "kmeans"])
    parser.add_argument("--projection", choices=["umap", "pca"], default="umap",
                        help="Méthode de projection 2D (défaut umap, "
                             "fallback pca si umap-learn absent)")
    parser.add_argument("--labels", type=Path, default=None,
                        help="motif_labels.csv (auto : <vame>/motif_labels.csv)")
    parser.add_argument("--start", type=float, default=0.0,
                        help="Début en secondes")
    parser.add_argument("--duration", type=float, default=None,
                        help="Durée en secondes (défaut : toute la session, "
                             "downsamplée)")
    parser.add_argument("--fps-source", type=float, default=30.0,
                        help="FPS de la vidéo source (défaut 30)")
    parser.add_argument("--fps-output", type=float, default=15.0,
                        help="FPS du GIF/MP4 de sortie (défaut 15). Baisse "
                             "pour un GIF plus léger.")
    parser.add_argument("--trail-length-sec", type=float, default=2.0,
                        help="Durée de la queue derrière le marqueur, en "
                             "secondes source (défaut 2).")
    parser.add_argument("--output-format", choices=["gif", "mp4"], default="gif")
    parser.add_argument("--marker-size", type=float, default=80,
                        help="Taille du marqueur courant")
    parser.add_argument("--background-alpha", type=float, default=0.15,
                        help="Transparence des points background (défaut 0.15)")
    args = parser.parse_args()

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
    except ImportError:
        print("❌ matplotlib requis", file=sys.stderr)
        sys.exit(1)

    project = resolve_project(args)
    vame_proj = vame_dir(project)
    if not (vame_proj / "config.yaml").exists():
        print(f"❌ Projet VAME introuvable : {vame_proj}", file=sys.stderr)
        sys.exit(1)

    latent_file = find_latent_vector(vame_proj, args.session)
    label_file = find_label_file(vame_proj, args.session, args.algo)
    if latent_file is None or label_file is None:
        print(f"❌ latent_vector ou label file introuvable pour {args.session}.\n"
              f"   Cherché dans : {vame_proj / 'results' / args.session}",
              file=sys.stderr)
        sys.exit(1)

    print(f"Session   : {args.session}")
    print(f"Latent    : {latent_file.name}  → shape ...")
    latents = np.load(latent_file)
    print(f"           {latents.shape}")
    labels = np.load(label_file).astype(int)
    print(f"Labels    : {label_file.name}  → {len(labels)} frames")

    if args.labels is None:
        default = vame_proj / "motif_labels.csv"
        if default.exists():
            args.labels = default
    motif_names = load_motif_names(args.labels)

    # Aligne latents et labels sur la même longueur (peuvent différer d'1 frame)
    L = min(len(latents), len(labels))
    latents = latents[:L]
    labels = labels[:L]

    # Projection 2D
    coords_2d = project_to_2d(latents, args.projection)

    # Downsample pour l'animation : cible n_frames_out
    fps_src = args.fps_source
    fps_out = args.fps_output
    start_frame = int(args.start * fps_src)
    end_frame = (int((args.start + args.duration) * fps_src)
                 if args.duration else L)
    end_frame = min(end_frame, L)
    src_indices = np.arange(start_frame, end_frame)
    if len(src_indices) == 0:
        print("❌ Intervalle vide.", file=sys.stderr)
        sys.exit(1)

    # Sous-échantillonne : garde 1 frame tous les (fps_src / fps_out)
    step = max(1, int(fps_src / fps_out))
    anim_indices = src_indices[::step]
    n_anim = len(anim_indices)
    trail_frames = int(args.trail_length_sec * fps_src / step)
    print(f"  animation : {n_anim} frames à {fps_out} fps "
          f"(step={step}, trail={trail_frames} points)\n")

    # ----- Setup figure -----
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.set_facecolor("white")

    # Background : tous les points de la session (color par motif)
    unique_motifs = np.unique(labels)
    for m in unique_motifs:
        mask = labels == m
        color = TAB20[m % len(TAB20)]
        ax.scatter(
            coords_2d[mask, 0], coords_2d[mask, 1],
            s=8, c=[color], alpha=args.background_alpha,
            edgecolors="none",
        )

    # Marqueur courant + trail
    trail_line, = ax.plot([], [], "-", color="black", linewidth=1.5, alpha=0.6)
    marker = ax.scatter([], [], s=args.marker_size, c="black",
                        edgecolors="white", linewidths=2, zorder=10)

    # Texte OSD
    osd = ax.text(
        0.02, 0.98, "", transform=ax.transAxes, va="top", ha="left",
        fontsize=11, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                  edgecolor="black", alpha=0.85),
    )
    time_text = ax.text(
        0.98, 0.02, "", transform=ax.transAxes, va="bottom", ha="right",
        fontsize=9, family="monospace",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  edgecolor="gray", alpha=0.7),
    )

    ax.set_xlabel(f"{args.projection.upper()}-1")
    ax.set_ylabel(f"{args.projection.upper()}-2")
    ax.set_title(f"Behavior manifold — {args.session}")
    # Retire les ticks pour un look propre
    ax.set_xticks([]); ax.set_yticks([])

    # Légende compacte des motifs (nom si dispo, sinon "motif_N")
    from matplotlib.lines import Line2D
    handles = []
    for m in unique_motifs[:20]:  # cap à 20 pour lisibilité
        name = motif_names.get(int(m), f"motif_{int(m)}")
        color = TAB20[m % len(TAB20)]
        handles.append(Line2D(
            [0], [0], marker="o", color="none", markerfacecolor=color,
            markersize=8, label=name,
        ))
    ax.legend(
        handles=handles, loc="center left", bbox_to_anchor=(1.02, 0.5),
        fontsize=8, frameon=False,
    )
    fig.tight_layout()

    # ----- Animation -----
    def update(frame_i: int):
        idx = anim_indices[frame_i]
        # Trail : dernières trail_frames positions dans l'animation
        start = max(0, frame_i - trail_frames)
        trail_idx = anim_indices[start:frame_i + 1]
        trail_line.set_data(coords_2d[trail_idx, 0], coords_2d[trail_idx, 1])
        # Marqueur courant
        current_motif = int(labels[idx])
        color = TAB20[current_motif % len(TAB20)]
        marker.set_offsets(np.array([[coords_2d[idx, 0], coords_2d[idx, 1]]]))
        marker.set_color([color])
        # OSD
        name = motif_names.get(current_motif, f"motif_{current_motif}")
        osd.set_text(f"[{current_motif}] {name}")
        elapsed = (idx - start_frame) / fps_src
        time_text.set_text(f"t = {elapsed:6.1f} s")
        return trail_line, marker, osd, time_text

    print("Rendu de l'animation...")
    anim = FuncAnimation(fig, update, frames=n_anim, interval=1000 / fps_out,
                        blit=False)

    out_dir = vame_proj / "analysis" / "behavior_structure"
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.start:.0f}s_{args.duration:.0f}s" if args.duration else "_full"
    stem = f"{args.session}_manifold_{args.projection}{suffix}"

    if args.output_format == "gif":
        out_path = out_dir / f"{stem}.gif"
        writer = PillowWriter(fps=fps_out)
        anim.save(str(out_path), writer=writer, dpi=80)
    else:
        out_path = out_dir / f"{stem}.mp4"
        try:
            writer = FFMpegWriter(fps=fps_out, bitrate=2000)
            anim.save(str(out_path), writer=writer, dpi=100)
        except Exception as e:
            print(f"⚠  FFMpegWriter échoué ({e}), fallback en gif+ffmpeg",
                  file=sys.stderr)
            tmp_gif = out_dir / f"{stem}_tmp.gif"
            anim.save(str(tmp_gif), writer=PillowWriter(fps=fps_out), dpi=80)
            subprocess.run(
                ["ffmpeg", "-y", "-i", str(tmp_gif), "-c:v", "libx264",
                 "-pix_fmt", "yuv420p", str(out_path)],
                check=True,
            )
            tmp_gif.unlink(missing_ok=True)

    plt.close(fig)
    size_mb = out_path.stat().st_size / 1e6
    print(f"\n✅ {out_path}  ({size_mb:.1f} MB)")
    if args.output_format == "gif" and size_mb > 30:
        print(f"   ⚠  >30MB : considère --duration plus court, "
              f"--fps-output plus bas, ou --output-format mp4")


if __name__ == "__main__":
    main()
