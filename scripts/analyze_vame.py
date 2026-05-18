"""
Analyse des résultats VAME en croisant avec les conditions expérimentales EthoFlow.

Pour un projet VAME donné, ce script :
  1. Liste toutes les sessions segmentées (motif_usage_session.npy)
  2. Récupère pour chaque session-arène : condition, stress, ANGII, timepoint
     depuis les metadata.yaml d'ethoflow
  3. Construit un DataFrame combiné session × motif × condition
  4. Sauvegarde un CSV et plusieurs plots de comparaison

Sortie : <project>/analysis/
  - motif_usage.csv          : usage normalisé de chaque motif par session
  - motif_usage_long.csv     : format long (1 ligne par session × motif)
  - heatmap_usage.png        : heatmap (sessions × motifs)
  - mean_by_condition.png    : usage moyen par groupe
  - boxplots_top_motifs.png  : distribution des motifs les plus différenciants

Usage:
    python scripts/analyze_vame.py
    python scripts/analyze_vame.py --algo hmm
    python scripts/analyze_vame.py --algo kmeans --n-clusters 15
    python scripts/analyze_vame.py --project /chemin/vers/vame-projects/<nom>
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # pas d'affichage interactif, juste fichiers
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
ETHOFLOW_RAW = ROOT / "data" / "raw"
CONFIG_POINTER = ROOT / ".vame_config_path"


def get_project_path(arg_project: str | None) -> Path:
    if arg_project:
        return Path(arg_project)
    if CONFIG_POINTER.exists():
        return Path(CONFIG_POINTER.read_text().strip()).parent
    raise FileNotFoundError(
        "Aucun projet VAME actif. Lance `run_vame.py use <nom>` ou passe --project."
    )


def find_segmentations(project_path: Path, algo: str, n_clusters: int | None) -> list[Path]:
    """Trouve tous les motif_usage_session.npy pour l'algo demandé."""
    results = project_path / "results"
    if not results.exists():
        raise FileNotFoundError(f"Pas de dossier results/ dans {project_path}")

    if n_clusters is not None:
        pattern = f"{algo}-{n_clusters}"
        glob = list(results.glob(f"*/*/{pattern}/motif_usage_session.npy"))
    else:
        glob = list(results.glob(f"*/*/{algo}-*/motif_usage_session.npy"))
    return sorted(glob)


def parse_session_name(name: str) -> tuple[str, str]:
    """'OF-M1-20251010-V01_A1' → ('OF-M1-20251010-V01', 'A1')."""
    m = re.match(r"^(.*)_([Aa]\d)$", name)
    if not m:
        return name, ""
    return m.group(1), m.group(2)


def load_metadata_index() -> dict[tuple[str, str], dict]:
    """
    Construit un dict {(session_id, arena_id) -> dict d'attributs} à partir
    des metadata.yaml d'ethoflow.
    """
    index: dict[tuple[str, str], dict] = {}
    if not ETHOFLOW_RAW.exists():
        return index
    for session_dir in ETHOFLOW_RAW.iterdir():
        meta_path = session_dir / "metadata.yaml"
        if not meta_path.exists():
            continue
        with open(meta_path) as f:
            meta = yaml.safe_load(f) or {}
        session_id = meta.get("session_id") or session_dir.name
        for ar in meta.get("arenes", []):
            arena_id = ar.get("id")
            if not arena_id:
                continue
            index[(session_id, arena_id)] = {
                "session_id": session_id,
                "arena": arena_id,
                "mouse_id": ar.get("mouse_id"),
                "condition": ar.get("condition"),
                "stress": ar.get("stress"),
                "angii": ar.get("angii"),
                "timepoint": meta.get("timepoint"),
                "date": meta.get("date"),
            }
    return index


def build_dataframe(seg_files: list[Path],
                    meta_index: dict[tuple[str, str], dict]) -> pd.DataFrame:
    rows = []
    for f in seg_files:
        usage = np.load(f).astype(float)
        # Normalise (proportion plutôt que count brut)
        total = usage.sum()
        usage_norm = usage / total if total > 0 else usage
        # Le 3e dossier parent au-dessus du fichier est le nom de session
        # (path = results/<session>/<model>/<algo-n>/motif_usage_session.npy)
        session_name = f.parent.parent.parent.name
        session_id, arena = parse_session_name(session_name)
        meta = meta_index.get((session_id, arena), {})
        for motif_i, freq in enumerate(usage_norm):
            rows.append({
                "session_full": session_name,
                "session_id": session_id,
                "arena": arena,
                "mouse_id": meta.get("mouse_id"),
                "condition": meta.get("condition"),
                "stress": meta.get("stress"),
                "angii": meta.get("angii"),
                "timepoint": meta.get("timepoint"),
                "motif": motif_i,
                "frequency": freq,
                "count": int(usage[motif_i]),
            })
    return pd.DataFrame(rows)


def plot_heatmap(df: pd.DataFrame, out_path: Path) -> None:
    pivot = df.pivot_table(
        index="session_full", columns="motif", values="frequency", aggfunc="mean"
    )
    if pivot.empty:
        return
    fig, ax = plt.subplots(figsize=(12, max(6, 0.25 * len(pivot))))
    im = ax.imshow(pivot.values, aspect="auto", cmap="viridis")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=7)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, fontsize=8)
    ax.set_xlabel("Motif")
    ax.set_title("Usage par session (proportion de frames)")
    fig.colorbar(im, ax=ax, label="proportion")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_means_by_condition(df: pd.DataFrame, condition_col: str,
                            out_path: Path, title: str) -> None:
    grouped = df.groupby([condition_col, "motif"])["frequency"].mean().reset_index()
    pivot = grouped.pivot(index="motif", columns=condition_col, values="frequency")
    if pivot.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    pivot.plot(kind="bar", ax=ax)
    ax.set_xlabel("Motif")
    ax.set_ylabel("Proportion moyenne")
    ax.set_title(title)
    ax.legend(title=condition_col, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def differentiating_motifs(df: pd.DataFrame, condition_col: str,
                           top_k: int = 6) -> list[int]:
    """Top-k motifs avec la plus grande différence absolue entre groupes."""
    means = df.groupby([condition_col, "motif"])["frequency"].mean().unstack(condition_col)
    if means.shape[1] < 2:
        return []
    diffs = means.diff(axis=1).abs().iloc[:, -1].sort_values(ascending=False)
    return diffs.head(top_k).index.tolist()


def plot_boxplots(df: pd.DataFrame, condition_col: str, motifs: list[int],
                  out_path: Path) -> None:
    if not motifs:
        return
    fig, axes = plt.subplots(1, len(motifs), figsize=(3 * len(motifs), 4), sharey=True)
    if len(motifs) == 1:
        axes = [axes]
    groups = sorted(df[condition_col].dropna().unique())
    for ax, motif in zip(axes, motifs):
        d = df[df["motif"] == motif]
        data = [d.loc[d[condition_col] == g, "frequency"].values for g in groups]
        ax.boxplot(data, labels=[str(g) for g in groups])
        ax.set_title(f"Motif {motif}")
        ax.set_ylabel("Proportion")
    fig.suptitle(f"Top motifs différenciants par {condition_col}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--project", default=None,
                        help="Chemin du projet VAME (défaut: courant via .vame_config_path)")
    parser.add_argument("--algo", default="hmm", choices=["hmm", "kmeans"],
                        help="Algo de segmentation à analyser (défaut: hmm)")
    parser.add_argument("--n-clusters", type=int, default=None,
                        help="Si plusieurs configurations existent, force un nombre de clusters")
    args = parser.parse_args()

    try:
        project = get_project_path(args.project)
    except FileNotFoundError as e:
        print(f"❌ {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Projet VAME : {project}")
    seg_files = find_segmentations(project, args.algo, args.n_clusters)
    if not seg_files:
        print(f"❌ Aucun motif_usage_session.npy trouvé pour algo='{args.algo}'.\n"
              f"   Lance d'abord `python scripts/run_vame.py segment`.",
              file=sys.stderr)
        sys.exit(1)
    print(f"  → {len(seg_files)} sessions segmentées (algo={args.algo})")

    meta_index = load_metadata_index()
    print(f"  → {len(meta_index)} entrées de metadata chargées")

    df = build_dataframe(seg_files, meta_index)
    if df.empty:
        print("❌ DataFrame vide, rien à analyser.", file=sys.stderr)
        sys.exit(1)

    n_motifs = df["motif"].nunique()
    n_sessions = df["session_full"].nunique()
    print(f"  → {n_sessions} sessions × {n_motifs} motifs = {len(df)} lignes")

    out_dir = project / "analysis"
    out_dir.mkdir(exist_ok=True)

    # CSV brut (un fichier par format)
    pivot = df.pivot_table(index="session_full", columns="motif",
                           values="frequency", aggfunc="mean")
    pivot.to_csv(out_dir / "motif_usage.csv")
    df.to_csv(out_dir / "motif_usage_long.csv", index=False)
    print(f"\n✓ CSV sauvés : {out_dir}/motif_usage.csv et motif_usage_long.csv")

    # Plots
    plot_heatmap(df, out_dir / "heatmap_usage.png")
    print(f"✓ Heatmap : {out_dir}/heatmap_usage.png")

    # Comparaisons par condition disponibles
    for col, title in [
        ("condition", "Usage moyen par condition (CUS / SHAM / + ANGII)"),
        ("timepoint", "Usage moyen par timepoint (M1 vs M2)"),
        ("stress",    "Usage moyen par stress (oui / non)"),
        ("angii",     "Usage moyen par ANGII (oui / non)"),
    ]:
        sub = df.dropna(subset=[col])
        if sub.empty or sub[col].nunique() < 2:
            continue
        plot_path = out_dir / f"mean_by_{col}.png"
        plot_means_by_condition(sub, col, plot_path, title)
        print(f"✓ Barres par {col} : {plot_path.name}")

        top = differentiating_motifs(sub, col, top_k=6)
        if top:
            box_path = out_dir / f"boxplots_top_by_{col}.png"
            plot_boxplots(sub, col, top, box_path)
            print(f"  → boxplots top motifs : {box_path.name} (motifs {top})")

    print(f"\n✅ Analyse terminée. Tout est dans {out_dir}")


if __name__ == "__main__":
    main()
