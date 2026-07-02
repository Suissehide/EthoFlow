"""Refait la dendrogramme des communautés VAME avec les labels du CSV.

VAME produit une dendrogramme (hierarchical clustering) des motifs basée
sur leurs transitions, mais avec les IDs numériques 0-14 aux feuilles.
Ce script recalcule la même hiérarchie depuis les données brutes et la
replote avec les noms lisibles issus de `motif_labels.csv`.

Algorithme :
    1. Agrège les labels par frame de TOUTES les sessions du projet
    2. Compte les transitions entre motifs consécutifs (matrice N×N)
    3. Normalise en probabilité conditionnelle P(next | current)
    4. Distance entre 2 motifs = 1 - cosine_similarity de leurs vecteurs
       de transition sortants (approche standard des community graphs VAME)
    5. Linkage 'ward' (compact) sur cette distance
    6. Dendrogramme avec les labels du CSV, taillé pour publi/screen

En prime : version color-coded par catégorie ETHOGRAM (si présente dans
motif_labels.csv), pour visualiser d'un coup d'œil quel motif appartient
à quelle catégorie.

Usage :
    python scripts/community_dendrogram.py \\
        --project-dir D:/ethoflow/projects/bottomview-MCC-2026-06

    # Filtre par groupe (dendrogramme par condition)
    python scripts/community_dendrogram.py \\
        --project-dir <...> --group MCCiECKO
    python scripts/community_dendrogram.py \\
        --project-dir <...> --group MCCf/f

    # Autre méthode de linkage
    python scripts/community_dendrogram.py \\
        --project-dir <...> --linkage average

Sortie : <vame_project>/analysis/community_dendrogram*.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import add_project_dir_arg, raw_dir, resolve_project, vame_dir  # noqa: E402


# Couleurs par catégorie ETHOGRAM (cohérent avec streamlit_app)
CATEGORY_COLORS = {
    "Locomotion":            "#e41a1c",
    "Stationary":            "#377eb8",
    "Vertical exploration":  "#4daf4a",
    "Sniffing":              "#984ea3",
    "Grooming":              "#ff7f00",
    "Exploration":           "#a65628",
    "Arena-specific":        "#f781bf",
    "Specific behaviors":    "#999999",
    "Transitions":           "#cccccc",
    "Catch-all":             "#000000",
}


def load_motif_labels(csv_path: Path | None) -> dict[int, dict]:
    """Charge motif_labels.csv (label + category) → dict par motif_id."""
    if csv_path is None or not csv_path.exists():
        return {}
    import pandas as pd
    with open(csv_path, encoding="utf-8") as f:
        first = f.readline()
    sep = ";" if first.count(";") > first.count(",") else ","
    df = pd.read_csv(csv_path, sep=sep, dtype=str, keep_default_na=False)
    df.columns = [c.strip() for c in df.columns]
    out = {}
    for _, row in df.iterrows():
        try:
            mid = int(row.get("motif_id", "").strip())
        except (TypeError, ValueError):
            continue
        label = row.get("label", "").strip() or row.get("category", "").strip()
        category = row.get("category", "").strip() or None
        # Si label vide, met la category en fallback
        if not label and category:
            label = category
        out[mid] = {"label": label or f"motif_{mid}", "category": category}
    return out


def load_session_group(project_ethoflow: Path, session_id: str) -> str | None:
    """Lit le groupe (MCCiECKO/MCCf/f) depuis la metadata."""
    import yaml
    meta = raw_dir(project_ethoflow) / session_id / "metadata.yaml"
    if not meta.exists():
        return None
    with open(meta) as f:
        d = yaml.safe_load(f) or {}
    return d.get("group") or d.get("condition")


def find_label_files(vame_project: Path, algo: str) -> list[tuple[str, Path]]:
    """Toutes les paires (session_id, path vers <n>_<algo>_label_<s>.npy)."""
    out = []
    results = vame_project / "results"
    if not results.exists():
        return out
    for session_dir in results.iterdir():
        if not session_dir.is_dir():
            continue
        for lf in session_dir.rglob(f"*_{algo}_label_{session_dir.name}.npy"):
            out.append((session_dir.name, lf))
            break  # une seule par session
    return out


def aggregate_transitions(label_files: list[tuple[str, Path]],
                          project_ethoflow: Path,
                          group_filter: str | None,
                          n_motifs: int) -> np.ndarray:
    """Matrice de transitions cumulée (self-loops exclus) toutes sessions."""
    T = np.zeros((n_motifs, n_motifs), dtype=float)
    n_sessions = 0
    for session_id, lf in label_files:
        if group_filter:
            g = load_session_group(project_ethoflow, session_id)
            if g != group_filter:
                continue
        labels = np.load(lf).astype(int)
        curr, nxt = labels[:-1], labels[1:]
        mask = curr != nxt
        for i, j in zip(curr[mask], nxt[mask]):
            if 0 <= i < n_motifs and 0 <= j < n_motifs:
                T[i, j] += 1
        n_sessions += 1
    print(f"  → {n_sessions} sessions incluses "
          f"({'filtre ' + group_filter if group_filter else 'toutes'})")
    return T


def motif_distance_from_transitions(T: np.ndarray) -> np.ndarray:
    """Distance symétrique entre motifs basée sur similarité cosine des
    transitions sortantes. Renvoie une matrice N×N de distances dans [0, 1].
    """
    # Normalise en distribution de transition sortante
    row_sums = T.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    P = T / row_sums
    # Cosine distance = 1 - dot(u, v) / (||u|| ||v||)
    N = P.shape[0]
    D = np.zeros((N, N), dtype=float)
    norms = np.linalg.norm(P, axis=1)
    for i in range(N):
        for j in range(i + 1, N):
            denom = norms[i] * norms[j]
            if denom == 0:
                d = 1.0
            else:
                d = 1.0 - float(np.dot(P[i], P[j]) / denom)
            D[i, j] = D[j, i] = d
    return D


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    add_project_dir_arg(parser, required=True)
    parser.add_argument("--algo", default="hmm", choices=["hmm", "kmeans"])
    parser.add_argument("--labels", type=Path, default=None,
                        help="motif_labels.csv (auto : <vame>/motif_labels.csv)")
    parser.add_argument("--group", default=None,
                        help="Filtrer par groupe (ex: MCCiECKO, 'MCCf/f')")
    parser.add_argument("--linkage", default="ward",
                        choices=["ward", "average", "complete", "single"],
                        help="Méthode de linkage scipy (défaut ward)")
    parser.add_argument("--figsize", nargs=2, type=float, default=[12, 8],
                        help="Taille de la figure en pouces (W H)")
    args = parser.parse_args()

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from scipy.cluster import hierarchy
        from scipy.spatial.distance import squareform
    except ImportError as e:
        print(f"❌ Dépendance manquante : {e}. pip install scipy matplotlib",
              file=sys.stderr)
        sys.exit(1)

    project = resolve_project(args)
    vame_proj = vame_dir(project)
    if not (vame_proj / "config.yaml").exists():
        print(f"❌ Projet VAME introuvable : {vame_proj}", file=sys.stderr)
        sys.exit(1)

    if args.labels is None:
        default = vame_proj / "motif_labels.csv"
        if default.exists():
            args.labels = default
    labels_dict = load_motif_labels(args.labels)
    if labels_dict:
        print(f"  → {len(labels_dict)} labels chargés depuis {args.labels}")

    # Trouve les fichiers de labels par frame
    label_files = find_label_files(vame_proj, args.algo)
    if not label_files:
        print(f"❌ Aucun fichier de labels trouvé "
              f"(algo={args.algo}, cherché dans {vame_proj / 'results'})",
              file=sys.stderr)
        sys.exit(1)
    print(f"  → {len(label_files)} sessions détectées\n")

    # Détermine n_motifs à partir du premier fichier
    first_labels = np.load(label_files[0][1])
    n_motifs = int(first_labels.max()) + 1
    print(f"Motifs détectés : {n_motifs}\n")

    # Agrégation transitions
    print("Agrégation des transitions...")
    T = aggregate_transitions(label_files, project, args.group, n_motifs)

    # Distance
    print("Calcul de la distance inter-motifs...")
    D = motif_distance_from_transitions(T)

    # Linkage (scipy attend la forme condensée)
    condensed = squareform(D, checks=False)
    Z = hierarchy.linkage(condensed, method=args.linkage)

    # Labels des feuilles
    leaf_labels = [
        labels_dict.get(i, {}).get("label", f"motif_{i}")
        for i in range(n_motifs)
    ]
    # Couleur de chaque feuille selon la catégorie
    leaf_colors = [
        CATEGORY_COLORS.get(
            labels_dict.get(i, {}).get("category", ""),
            "#333333",
        )
        for i in range(n_motifs)
    ]

    # Plot
    fig, ax = plt.subplots(figsize=args.figsize)
    title = "Communautés de motifs — dendrogramme des transitions"
    if args.group:
        title += f"  ({args.group})"
    ax.set_title(title, fontsize=13, pad=15)

    dendro = hierarchy.dendrogram(
        Z,
        labels=leaf_labels,
        leaf_rotation=45,
        leaf_font_size=10,
        color_threshold=0,  # branches en noir uniforme
        above_threshold_color="#333333",
        ax=ax,
    )
    ax.set_ylabel(f"Distance ({args.linkage} linkage)")
    ax.set_xlabel("Motif")

    # Colorie les labels des feuilles selon leur catégorie
    xticklabels = ax.get_xticklabels()
    for label_txt in xticklabels:
        name = label_txt.get_text()
        # Retrouve l'index original du motif via leaf_labels
        try:
            motif_i = leaf_labels.index(name)
            label_txt.set_color(leaf_colors[motif_i])
        except ValueError:
            pass

    # Légende catégories utilisées
    from matplotlib.patches import Patch
    used_categories = {
        labels_dict.get(i, {}).get("category")
        for i in range(n_motifs)
        if labels_dict.get(i, {}).get("category")
    }
    if used_categories:
        handles = [
            Patch(facecolor=CATEGORY_COLORS.get(cat, "#333333"), label=cat)
            for cat in sorted(used_categories)
        ]
        ax.legend(
            handles=handles, title="Catégorie ETHOGRAM",
            loc="center left", bbox_to_anchor=(1.02, 0.5),
            fontsize=9, frameon=False,
        )

    fig.tight_layout()

    out_dir = vame_proj / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.group.replace('/', '-')}" if args.group else ""
    out_path = out_dir / f"community_dendrogram{suffix}.png"
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)

    print(f"\n✅ {out_path}")


if __name__ == "__main__":
    main()
