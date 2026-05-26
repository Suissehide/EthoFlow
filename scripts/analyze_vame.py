"""
Analyse des résultats VAME en croisant avec les conditions expérimentales EthoFlow.

Pour un projet VAME donné, ce script :
  1. Liste toutes les sessions segmentées (motif_usage_<session>.npy)
  2. Récupère pour chaque session-arène : condition, stress, ANGII, timepoint
     depuis les metadata.yaml d'ethoflow
  3. Construit un DataFrame combiné session × motif × condition
  4. (optionnel) Détecte les frames "empty arena" en bord d'enregistrement
     via --validity-source ; --mask-empty les exclut des fréquences
  5. Sauvegarde un CSV et plusieurs plots de comparaison

Sortie : <project>/analysis/
  - motif_usage.csv             : usage normalisé de chaque motif par session
  - motif_usage_long.csv        : format long (1 ligne par session × motif)
  - validity_per_session.csv    : (si --validity-source) frames empty-arena par session
  - heatmap_usage.png           : heatmap (sessions × motifs)
  - mean_by_condition.png       : usage moyen par groupe
  - boxplots_top_motifs.png     : distribution des motifs les plus différenciants

Usage:
    python scripts/analyze_vame.py
    python scripts/analyze_vame.py --algo hmm
    python scripts/analyze_vame.py --algo kmeans --n-clusters 15
    python scripts/analyze_vame.py --project /chemin/vers/vame-projects/<nom>
    python scripts/analyze_vame.py --labels data/results/motif_labels_hmm15.yaml
    python scripts/analyze_vame.py --validity-source data/vame-input/single-enhanced-2026-05
    python scripts/analyze_vame.py --validity-source ... --mask-empty

Le fichier de labels (--labels) est un YAML simple {motif_id: étiquette} :

    0: immobilité
    1: locomotion lente
    2: rearing
    ...

Les motifs non listés gardent une étiquette par défaut "motif_<i>". Output :
colonne 'label' dans motif_usage_long.csv, en-têtes renommées dans
motif_usage.csv, étiquettes sur les axes et titres des plots.
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


def load_motif_labels(labels_path: Path | None) -> dict[int, str]:
    """
    Charge un YAML qui mappe motif_id → étiquette comportementale.

    Format attendu (clés entières, valeurs chaînes) :

        0: immobilité
        1: locomotion lente
        2: rearing
        ...

    Retourne {} si aucun fichier fourni. Les motifs non listés sont tolérés
    et gardent leur étiquette par défaut "motif_<i>".
    """
    if labels_path is None:
        return {}
    if not labels_path.exists():
        raise FileNotFoundError(f"Fichier de labels introuvable : {labels_path}")
    with open(labels_path) as f:
        raw = yaml.safe_load(f) or {}
    labels: dict[int, str] = {}
    for k, v in raw.items():
        try:
            labels[int(k)] = str(v)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Clé non-entière dans {labels_path} : {k!r}"
            ) from exc
    return labels


def motif_display(motif_id: int, labels: dict[int, str]) -> str:
    """Forme compacte pour CSV/plots : '3: grooming' si labellisé, sinon 'motif_3'."""
    label = labels.get(motif_id)
    return f"{motif_id}: {label}" if label else f"motif_{motif_id}"


def find_label_file(motif_usage_path: Path) -> Path | None:
    """
    À partir d'un motif_usage_<session>.npy, retrouve le fichier de label
    par frame <n>_<algo>_label_<session>.npy situé dans le même dossier.
    """
    parent = motif_usage_path.parent
    if "-" not in parent.name:
        return None
    algo, n = parent.name.split("-", 1)
    prefix = "motif_usage_"
    stem = motif_usage_path.stem
    if not stem.startswith(prefix):
        return None
    session_name = stem[len(prefix):]
    label = parent / f"{n}_{algo}_label_{session_name}.npy"
    return label if label.exists() else None


def find_prefill_h5(validity_source: Path, session_name: str) -> Path | None:
    """
    Localise le .h5 pré-fill (avant filter_keypoints / fill_nan_h5) pour
    une session_full donnée (ex. 'OF-M1-20251010-V02_A4').
    Structure attendue : <validity_source>/<session_id>/<session_full>.h5
    """
    session_id, arena = parse_session_name(session_name)
    if not arena:
        return None
    candidate = validity_source / session_id / f"{session_name}.h5"
    return candidate if candidate.exists() else None


def detect_empty_arena_edges(prefill_h5: Path,
                             min_edge_frames: int = 25) -> tuple[int, int]:
    """
    Compte les frames d'« empty arena » au tout début et à la toute fin
    du h5 pré-fill : blocs contigus où aucun keypoint n'est détecté.

    Critère « pas de souris » : toutes les colonnes x sont NaN sur la frame.
    Le fill_nan_h5 a remplacé ces NaN dans le dossier -clean/, mais le
    pré-fill (passé en --validity-source) garde le pattern d'origine.

    Renvoie (n_empty_start, n_empty_end). Un bloc en bordure n'est compté
    que s'il fait au moins min_edge_frames frames — sinon c'est du bruit
    DLC ponctuel (un raté de détection sur 1-2 frames au démarrage).
    """
    df = pd.read_hdf(prefill_h5)
    x_cols = [c for c in df.columns if isinstance(c, tuple) and c[-1] == "x"]
    if not x_cols:
        return 0, 0
    no_det = df[x_cols].isna().all(axis=1).to_numpy()
    n = len(no_det)
    start = 0
    while start < n and no_det[start]:
        start += 1
    end = 0
    while end < n and no_det[n - 1 - end]:
        end += 1
    return (start if start >= min_edge_frames else 0,
            end if end >= min_edge_frames else 0)


def compute_session_validity(seg_file: Path, validity_source: Path,
                             min_edge_frames: int) -> dict | None:
    """
    Pour une session segmentée (chemin du motif_usage_<session>.npy),
    construit le masque valid_mask + détecte les frames empty-arena.

    Renvoie None si le label par frame ou le h5 pré-fill sont introuvables
    (la session sera analysée sans masquage, avec un warning).
    """
    session_name = seg_file.parent.parent.parent.name
    prefill = find_prefill_h5(validity_source, session_name)
    label_file = find_label_file(seg_file)
    if prefill is None or label_file is None:
        return None
    labels_per_frame = np.load(label_file).astype(int)
    n_total = len(labels_per_frame)
    n_empty_start, n_empty_end = detect_empty_arena_edges(prefill, min_edge_frames)
    n_empty_start = min(n_empty_start, n_total)
    n_empty_end = min(n_empty_end, n_total - n_empty_start)
    valid_mask = np.ones(n_total, dtype=bool)
    if n_empty_start > 0:
        valid_mask[:n_empty_start] = False
    if n_empty_end > 0:
        valid_mask[-n_empty_end:] = False
    return {
        "session_name": session_name,
        "n_total": n_total,
        "n_empty_start": n_empty_start,
        "n_empty_end": n_empty_end,
        "valid_mask": valid_mask,
        "labels_per_frame": labels_per_frame,
    }


def find_segmentations(project_path: Path, algo: str, n_clusters: int | None) -> list[Path]:
    """Trouve tous les motif_usage_<session>.npy pour l'algo demandé.

    Structure VAME-py : results/<session>/<model>/<algo>-<n>/motif_usage_<session>.npy
    """
    results = project_path / "results"
    if not results.exists():
        raise FileNotFoundError(f"Pas de dossier results/ dans {project_path}")

    if n_clusters is not None:
        pattern = f"{algo}-{n_clusters}"
        glob = list(results.glob(f"*/*/{pattern}/motif_usage_*.npy"))
    else:
        glob = list(results.glob(f"*/*/{algo}-*/motif_usage_*.npy"))
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
                    meta_index: dict[tuple[str, str], dict],
                    labels: dict[int, str],
                    validity_source: Path | None = None,
                    min_edge_frames: int = 25,
                    mask_empty: bool = False
                    ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Construit (motif_df, validity_df).

    motif_df : 1 ligne par session × motif. Colonnes principales : motif,
    label, frequency, count, et — si validity_source — empty_arena_count
    et empty_arena_fraction par motif × session.

    validity_df : 1 ligne par session, avec n_empty_start, n_empty_end,
    n_valid_frames, valid_fraction. Vide si validity_source=None.

    Si mask_empty=True, la `frequency` est recalculée en excluant les frames
    empty-arena (le `count` reste l'original pour audit).
    """
    rows = []
    validity_rows = []
    for f in seg_files:
        usage_original = np.load(f).astype(float)
        n_clusters = len(usage_original)
        # path = results/<session>/<model>/<algo-n>/motif_usage_<session>.npy
        session_name = f.parent.parent.parent.name
        session_id, arena = parse_session_name(session_name)
        meta = meta_index.get((session_id, arena), {})

        per_motif_empty = None  # array (n_clusters,) si validity activée
        n_empty_start = n_empty_end = 0
        if validity_source is not None:
            vinfo = compute_session_validity(f, validity_source, min_edge_frames)
            if vinfo is None:
                print(f"  ⚠️  {session_name} : pré-fill ou label par frame "
                      f"introuvable — pas de masquage empty-arena ici.",
                      file=sys.stderr)
            else:
                n_empty_start = vinfo["n_empty_start"]
                n_empty_end = vinfo["n_empty_end"]
                labels_per_frame = vinfo["labels_per_frame"]
                valid_mask = vinfo["valid_mask"]
                per_motif_empty = np.zeros(n_clusters, dtype=int)
                for m in range(n_clusters):
                    per_motif_empty[m] = int(
                        ((labels_per_frame == m) & ~valid_mask).sum()
                    )
                n_valid = vinfo["n_total"] - n_empty_start - n_empty_end
                validity_rows.append({
                    "session_full": session_name,
                    "session_id": session_id,
                    "arena": arena,
                    "n_frames_total": vinfo["n_total"],
                    "n_empty_start": n_empty_start,
                    "n_empty_end": n_empty_end,
                    "n_valid_frames": n_valid,
                    "valid_fraction": n_valid / max(vinfo["n_total"], 1),
                })

        # Si mask_empty, on retire les frames empty-arena de la frequency
        # (numérateur ET dénominateur). Le `count` reste l'original.
        if mask_empty and per_motif_empty is not None:
            usage_for_freq = usage_original - per_motif_empty
        else:
            usage_for_freq = usage_original
        total = usage_for_freq.sum()
        usage_norm = usage_for_freq / total if total > 0 else usage_for_freq

        for motif_i in range(n_clusters):
            row = {
                "session_full": session_name,
                "session_id": session_id,
                "arena": arena,
                "mouse_id": meta.get("mouse_id"),
                "condition": meta.get("condition"),
                "stress": meta.get("stress"),
                "angii": meta.get("angii"),
                "timepoint": meta.get("timepoint"),
                "motif": motif_i,
                "label": labels.get(motif_i, f"motif_{motif_i}"),
                "frequency": float(usage_norm[motif_i]),
                "count": int(usage_original[motif_i]),
            }
            if per_motif_empty is not None:
                orig = int(usage_original[motif_i])
                row["empty_arena_count"] = int(per_motif_empty[motif_i])
                row["empty_arena_fraction"] = (
                    per_motif_empty[motif_i] / orig if orig > 0 else 0.0
                )
            rows.append(row)
    return pd.DataFrame(rows), pd.DataFrame(validity_rows)


def plot_heatmap(df: pd.DataFrame, out_path: Path,
                 labels: dict[int, str]) -> None:
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
    xtick_labels = [motif_display(int(m), labels) for m in pivot.columns]
    ax.set_xticklabels(xtick_labels, fontsize=8, rotation=45, ha="right")
    ax.set_xlabel("Motif")
    ax.set_title("Usage par session (proportion de frames)")
    fig.colorbar(im, ax=ax, label="proportion")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_means_by_condition(df: pd.DataFrame, condition_col: str,
                            out_path: Path, title: str,
                            labels: dict[int, str]) -> None:
    grouped = df.groupby([condition_col, "motif"])["frequency"].mean().reset_index()
    pivot = grouped.pivot(index="motif", columns=condition_col, values="frequency")
    if pivot.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    pivot.plot(kind="bar", ax=ax)
    ax.set_xticklabels(
        [motif_display(int(m), labels) for m in pivot.index],
        rotation=45, ha="right", fontsize=8,
    )
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
                  out_path: Path, labels: dict[int, str]) -> None:
    if not motifs:
        return
    fig, axes = plt.subplots(1, len(motifs), figsize=(3 * len(motifs), 4), sharey=True)
    if len(motifs) == 1:
        axes = [axes]
    groups = sorted(df[condition_col].dropna().unique())
    for ax, motif in zip(axes, motifs):
        d = df[df["motif"] == motif]
        data = [d.loc[d[condition_col] == g, "frequency"].values for g in groups]
        ax.boxplot(data, tick_labels=[str(g) for g in groups])
        ax.set_title(motif_display(motif, labels), fontsize=10)
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
    parser.add_argument("--labels", type=Path, default=None,
                        help="YAML de mapping motif_id → étiquette comportementale. "
                             "Ajoute une colonne 'label' aux CSV et utilise ces "
                             "étiquettes sur les axes des graphiques.")
    parser.add_argument("--validity-source", type=Path, default=None,
                        help="Dossier des .h5 PRÉ-fill (avant fill_nan_h5), "
                             "typiquement data/vame-input/single-enhanced-2026-05. "
                             "Active la détection des frames empty-arena en bord "
                             "d'enregistrement (NaN au tout début/fin).")
    parser.add_argument("--min-edge-frames", type=int, default=25,
                        help="Longueur minimale d'un bloc NaN en bord pour le "
                             "compter comme empty-arena (défaut 25 = 1 s @ 25 fps). "
                             "En-dessous c'est traité comme du bruit DLC ponctuel.")
    parser.add_argument("--mask-empty", action="store_true",
                        help="Recalcule les fréquences en excluant les frames "
                             "empty-arena du dénominateur (le count original "
                             "reste dans la colonne 'count' pour audit).")
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

    try:
        labels = load_motif_labels(args.labels)
    except (FileNotFoundError, ValueError) as e:
        print(f"❌ {e}", file=sys.stderr)
        sys.exit(1)
    if labels:
        print(f"  → {len(labels)} étiquettes de motifs chargées depuis {args.labels}")
    else:
        print("  ℹ Aucun label — passe --labels <fichier.yaml> "
              "pour étiqueter les motifs (YAML format: motif_id: nom).")

    df, validity_df = build_dataframe(
        seg_files, meta_index, labels,
        validity_source=args.validity_source,
        min_edge_frames=args.min_edge_frames,
        mask_empty=args.mask_empty,
    )
    if df.empty:
        print("❌ DataFrame vide, rien à analyser.", file=sys.stderr)
        sys.exit(1)

    n_motifs = df["motif"].nunique()
    n_sessions = df["session_full"].nunique()
    print(f"  → {n_sessions} sessions × {n_motifs} motifs = {len(df)} lignes")
    if not validity_df.empty:
        n_affected = int(
            (validity_df["n_empty_start"] + validity_df["n_empty_end"] > 0).sum()
        )
        total_empty = int(
            validity_df["n_empty_start"].sum() + validity_df["n_empty_end"].sum()
        )
        mode_txt = ("exclues des fréquences (--mask-empty)" if args.mask_empty
                    else "diagnostique uniquement — repasse avec --mask-empty "
                         "pour les exclure")
        print(f"  → empty-arena : {n_affected}/{len(validity_df)} sessions "
              f"affectées, {total_empty} frames au total ({mode_txt})")

    out_dir = project / "analysis"
    out_dir.mkdir(exist_ok=True)

    # CSV brut (un fichier par format)
    pivot = df.pivot_table(index="session_full", columns="motif",
                           values="frequency", aggfunc="mean")
    if labels:
        # En-têtes lisibles si on a un mapping, ex : "3: grooming"
        pivot = pivot.rename(columns=lambda m: motif_display(int(m), labels))
    pivot.to_csv(out_dir / "motif_usage.csv")
    df.to_csv(out_dir / "motif_usage_long.csv", index=False)
    print(f"\n✓ CSV sauvés : {out_dir}/motif_usage.csv et motif_usage_long.csv")
    if not validity_df.empty:
        validity_df.to_csv(out_dir / "validity_per_session.csv", index=False)
        print(f"✓ Validity par session : {out_dir}/validity_per_session.csv")

    # Plots
    plot_heatmap(df, out_dir / "heatmap_usage.png", labels)
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
        plot_means_by_condition(sub, col, plot_path, title, labels)
        print(f"✓ Barres par {col} : {plot_path.name}")

        top = differentiating_motifs(sub, col, top_k=6)
        if top:
            box_path = out_dir / f"boxplots_top_by_{col}.png"
            plot_boxplots(sub, col, top, box_path, labels)
            print(f"  → boxplots top motifs : {box_path.name} (motifs {top})")

    print(f"\n✅ Analyse terminée. Tout est dans {out_dir}")


if __name__ == "__main__":
    main()
