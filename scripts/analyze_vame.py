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

# Import des chemins projet-aware
sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import (  # noqa: E402
    REPO_ROOT,
    add_project_dir_arg,
    raw_dir,
    resolve_project,
)

# Le pointeur du projet VAME courant reste au repo root : c'est un état
# session-wide (quelle config VAME utiliser), pas une donnée du projet.
CONFIG_POINTER = REPO_ROOT / ".vame_config_path"


def get_project_path(arg_project: str | None) -> Path:
    if arg_project:
        return Path(arg_project)
    if CONFIG_POINTER.exists():
        return Path(CONFIG_POINTER.read_text().strip()).parent
    raise FileNotFoundError(
        "Aucun projet VAME actif. Lance `run_vame.py use <nom>` ou passe --project."
    )


# Catégories reconnues du référentiel ETHOGRAM (cf. streamlit_app/lib/config.py).
# Utilisé pour détecter le sens des colonnes du CSV utilisateur : si `label`
# contient une catégorie et `category` un label spécifique, on swap.
_KNOWN_CATEGORIES = {
    "locomotion", "stationary", "vertical exploration", "sniffing",
    "grooming", "exploration", "arena-specific", "specific behaviors",
    "catch-all", "transitions",
}


def load_motif_labels(labels_path: Path | None) -> dict[int, dict]:
    """
    Charge un mapping motif_id → {label, category, confidence, notes}.

    Deux formats supportés :

    1. YAML plat (rétrocompat historique) :
           0: immobilité
           1: locomotion lente
       → label seul, pas de catégorie.

    2. CSV avec en-tête (recommandé, produit par motif_labels_template.csv) :
           motif_id;label;category;confidence;qc_inspected_sessions;notes
           0;rearing_exploration;Vertical exploration;high;BV-970;...
       Séparateur détecté automatiquement (`,` ou `;`).

       **Auto-correction du column swap** : si un utilisateur a rempli le
       label dans la colonne `category` (piège classique du template), on
       détecte que les valeurs de `label` correspondent à des catégories
       ETHOGRAM connues (ou l'inverse) et on swap silencieusement.

    Retourne {} si aucun fichier fourni.
    """
    if labels_path is None:
        return {}
    if not labels_path.exists():
        raise FileNotFoundError(f"Fichier de labels introuvable : {labels_path}")

    # --- YAML (legacy) ---
    if labels_path.suffix.lower() in {".yaml", ".yml"}:
        with open(labels_path) as f:
            raw = yaml.safe_load(f) or {}
        return {
            int(k): {"label": str(v), "category": None}
            for k, v in raw.items()
        }

    # --- CSV (nouveau format) ---
    # Détection auto du séparateur
    with open(labels_path, encoding="utf-8") as f:
        first_line = f.readline()
    sep = ";" if first_line.count(";") > first_line.count(",") else ","

    df = pd.read_csv(labels_path, sep=sep, dtype=str, keep_default_na=False)
    df.columns = [c.strip() for c in df.columns]
    required = {"motif_id"}
    if not required.issubset(df.columns):
        raise ValueError(
            f"{labels_path} : colonne 'motif_id' manquante "
            f"(colonnes trouvées : {list(df.columns)})"
        )

    label_col = "label" if "label" in df.columns else None
    cat_col = "category" if "category" in df.columns else None

    # Swap `label` ↔ `category` si nécessaire.
    # Deux cas de figure gérés :
    # (a) label vide partout, category rempli   → l'utilisateur s'est trompé
    #     de colonne, on prend category comme label et on met category à None
    # (b) label ressemble à des catégories ETHOGRAM et category à des noms
    #     spécifiques → swap complet
    if label_col and cat_col:
        label_has_content = any(
            str(v).strip() for v in df[label_col]
        )
        cat_has_content = any(
            str(v).strip() for v in df[cat_col]
        )

        if not label_has_content and cat_has_content:
            # Cas (a) : promeut category → label, laisse category vide
            print(f"ℹ️  {labels_path.name} : colonne 'label' vide, "
                  f"utilisation de 'category' comme label.", file=sys.stderr)
            df[label_col] = df[cat_col]
            df[cat_col] = ""
        elif label_has_content and cat_has_content:
            # Cas (b) : vérifie si les colonnes sont inversées
            label_vals = [str(v).strip().lower() for v in df[label_col] if str(v).strip()]
            cat_vals = [str(v).strip().lower() for v in df[cat_col] if str(v).strip()]

            def _looks_like_category(vals: list[str]) -> bool:
                if not vals:
                    return False
                n_cat = sum(
                    1 for v in vals
                    if any(kc in v for kc in _KNOWN_CATEGORIES) and len(v.split()) <= 3
                )
                return n_cat / len(vals) > 0.5

            if _looks_like_category(label_vals) and not _looks_like_category(cat_vals):
                print(f"ℹ️  {labels_path.name} : colonnes 'label' et 'category' "
                      f"inversées — swap automatique.", file=sys.stderr)
                label_col, cat_col = cat_col, label_col

    labels: dict[int, dict] = {}
    for _, row in df.iterrows():
        mid_raw = row.get("motif_id", "").strip()
        if not mid_raw:
            continue
        try:
            mid = int(mid_raw)
        except (TypeError, ValueError):
            continue
        entry = {
            "label": row.get(label_col, "").strip() if label_col else "",
            "category": row.get(cat_col, "").strip() if cat_col else "",
            "confidence": row.get("confidence", "").strip(),
            "notes": row.get("notes", "").strip(),
        }
        # Nettoie les strings vides → None pour lisibilité
        for k in ("label", "category", "confidence", "notes"):
            if not entry[k]:
                entry[k] = None
        labels[mid] = entry
    return labels


def motif_display(motif_id: int, labels: dict[int, dict] | dict[int, str]) -> str:
    """Forme compacte pour CSV/plots : '3: grooming' si labellisé, sinon 'motif_3'.

    Accepte dict[int, dict] (nouveau format riche) ou dict[int, str] (legacy).
    """
    entry = labels.get(motif_id)
    if entry is None:
        return f"motif_{motif_id}"
    if isinstance(entry, dict):
        label = entry.get("label")
    else:
        label = entry
    return f"{motif_id}: {label}" if label else f"motif_{motif_id}"


def motif_category(motif_id: int, labels: dict[int, dict]) -> str | None:
    """Retourne la catégorie ETHOGRAM pour ce motif, ou None."""
    entry = labels.get(motif_id)
    if not isinstance(entry, dict):
        return None
    return entry.get("category")


def is_artifact_motif(motif_id: int, labels: dict[int, dict]) -> bool:
    """True si le motif est marqué confidence=artifact (à exclure de l'analyse)."""
    entry = labels.get(motif_id)
    if not isinstance(entry, dict):
        return False
    conf = entry.get("confidence")
    return conf is not None and conf.lower() == "artifact"


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


def load_metadata_index(project: Path) -> dict[tuple[str, str], dict]:
    """
    Construit un dict {(session_id, arena_id) -> dict d'attributs} à partir
    des metadata.yaml d'ethoflow.

    Deux formats auto-détectés :

    - **Topview** : metadata avec `arenes: [...]` — une ligne par arène
      dans l'index, arena_id = "A1"..."A4", condition = `arene.condition`.
    - **Bottomview** : metadata plate (pas d'`arenes:`) — une seule ligne
      par session, arena_id = "", condition = `metadata.group`
      (MCCiECKO / MCCf/f).

    Cette double approche permet à un même script de traiter les deux
    types de projets sans branchement en amont.
    """
    index: dict[tuple[str, str], dict] = {}
    ethoflow_raw = raw_dir(project)
    if not ethoflow_raw.exists():
        return index

    for session_dir in ethoflow_raw.iterdir():
        meta_path = session_dir / "metadata.yaml"
        if not meta_path.exists():
            continue
        with open(meta_path) as f:
            meta = yaml.safe_load(f) or {}
        session_id = meta.get("session_id") or session_dir.name
        arenes = meta.get("arenes") or []

        if arenes:
            # Topview : une entrée par arène
            for ar in arenes:
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
        else:
            # Bottomview : metadata plate, une entrée par session
            index[(session_id, "")] = {
                "session_id": session_id,
                "arena": "",
                "mouse_id": meta.get("mouse_id"),
                "condition": meta.get("group"),  # MCCiECKO / MCCf/f
                "sex": meta.get("sex"),
                "cage": meta.get("cage"),
                "birth_date": meta.get("birth_date"),
                "date": meta.get("date_recorded"),
                "line": meta.get("line"),
                "genotype_mcc": meta.get("genotype_mcc"),
                "genotype_cdh5_cre": meta.get("genotype_cdh5_cre"),
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
            # Skip les motifs marqués artifact dans motif_labels
            if is_artifact_motif(motif_i, labels):
                continue

            entry = labels.get(motif_i)
            if isinstance(entry, dict):
                label_str = entry.get("label") or f"motif_{motif_i}"
                category = entry.get("category")
            else:
                label_str = entry if entry else f"motif_{motif_i}"
                category = None

            row = {
                "session_full": session_name,
                "session_id": session_id,
                "arena": arena,
                "mouse_id": meta.get("mouse_id"),
                "condition": meta.get("condition"),
                # Métadonnées topview (peuvent être None sur bottomview)
                "stress": meta.get("stress"),
                "angii": meta.get("angii"),
                "timepoint": meta.get("timepoint"),
                # Métadonnées bottomview (peuvent être None sur topview)
                "sex": meta.get("sex"),
                "cage": meta.get("cage"),
                "date": meta.get("date"),
                "motif": motif_i,
                "label": label_str,
                "category": category,
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


def aggregate_by_category(df: pd.DataFrame) -> pd.DataFrame:
    """Agrège la fréquence par catégorie ETHOGRAM (ex: 'Sniffing') plutôt
    que par motif individuel. Ignore les motifs sans catégorie assignée.

    Retourne un DataFrame avec 1 ligne par session × catégorie.
    """
    df_cat = df[df["category"].notna()].copy()
    if df_cat.empty:
        return pd.DataFrame()
    agg = (
        df_cat.groupby(
            ["session_full", "session_id", "arena", "condition",
             "sex", "cage", "date", "category"],
            dropna=False,
        )["frequency"]
        .sum()
        .reset_index()
        .rename(columns={"frequency": "frequency_total"})
    )
    return agg


def stats_by_motif(df: pd.DataFrame, condition_col: str) -> pd.DataFrame:
    """Mann-Whitney U par motif entre les groupes de `condition_col`
    (typiquement 'condition' = MCCiECKO vs MCCf/f pour bottomview).

    Correction BH (Benjamini-Hochberg) sur l'ensemble des motifs testés.
    Renvoie un DataFrame trié par p-value croissante.
    """
    try:
        from scipy import stats
    except ImportError:
        print("⚠  scipy non installé, stats skippées", file=sys.stderr)
        return pd.DataFrame()

    sub = df.dropna(subset=[condition_col])
    groups = sorted(sub[condition_col].unique())
    if len(groups) != 2:
        print(f"⚠  stats_by_motif : {len(groups)} groupes trouvés dans "
              f"'{condition_col}' — test Mann-Whitney nécessite exactement 2.",
              file=sys.stderr)
        return pd.DataFrame()

    rows = []
    for m in sorted(sub["motif"].unique()):
        d = sub[sub["motif"] == m]
        a = d.loc[d[condition_col] == groups[0], "frequency"].values
        b = d.loc[d[condition_col] == groups[1], "frequency"].values
        if len(a) < 3 or len(b) < 3:
            continue
        try:
            u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
        except ValueError:
            continue
        rows.append({
            "motif": m,
            "label": d["label"].iloc[0],
            "category": d["category"].iloc[0] if "category" in d.columns else None,
            f"mean_{groups[0]}": float(np.mean(a)),
            f"mean_{groups[1]}": float(np.mean(b)),
            "diff": float(np.mean(b) - np.mean(a)),
            "u_stat": float(u),
            "p_value": float(p),
            f"n_{groups[0]}": len(a),
            f"n_{groups[1]}": len(b),
        })
    result = pd.DataFrame(rows)
    if result.empty:
        return result

    # Correction Benjamini-Hochberg (FDR)
    ps = result["p_value"].values
    order = np.argsort(ps)
    ranked = ps[order]
    n = len(ranked)
    q = np.zeros(n)
    prev = 1.0
    for i in range(n - 1, -1, -1):
        prev = min(prev, ranked[i] * n / (i + 1))
        q[i] = prev
    q_full = np.zeros(n)
    q_full[order] = q
    result["p_adj_bh"] = q_full
    result["significant_0.05"] = result["p_adj_bh"] < 0.05
    return result.sort_values("p_value").reset_index(drop=True)


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
    add_project_dir_arg(parser)
    parser.add_argument("--project", default=None,
                        help="Chemin du projet VAME (défaut: courant via .vame_config_path)")
    parser.add_argument("--algo", default="hmm", choices=["hmm", "kmeans"],
                        help="Algo de segmentation à analyser (défaut: hmm)")
    parser.add_argument("--n-clusters", type=int, default=None,
                        help="Si plusieurs configurations existent, force un nombre de clusters")
    parser.add_argument("--labels", type=Path, default=None,
                        help="Fichier de labels : YAML (legacy, motif_id: nom) "
                             "OU CSV (recommandé, colonnes "
                             "motif_id;label;category;confidence;notes). "
                             "Défaut auto : <vame_project>/motif_labels.csv.")
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

    ethoflow_project = resolve_project(args)

    try:
        project = get_project_path(args.project)
    except FileNotFoundError as e:
        print(f"❌ {e}", file=sys.stderr)
        sys.exit(1)

    # Auto-résolution du path des labels : par défaut, motif_labels.csv à la
    # racine du projet VAME (convention EthoFlow).
    if args.labels is None:
        default_labels = project / "motif_labels.csv"
        if default_labels.exists():
            args.labels = default_labels
            print(f"ℹ  Labels auto-détectés : {default_labels}")

    print(f"Projet VAME : {project}")
    seg_files = find_segmentations(project, args.algo, args.n_clusters)
    if not seg_files:
        print(f"❌ Aucun motif_usage_session.npy trouvé pour algo='{args.algo}'.\n"
              f"   Lance d'abord `python scripts/run_vame.py segment`.",
              file=sys.stderr)
        sys.exit(1)
    print(f"  → {len(seg_files)} sessions segmentées (algo={args.algo})")

    meta_index = load_metadata_index(ethoflow_project)
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
        ("condition", "Usage moyen par condition (MCCiECKO / MCCf/f pour bottomview, "
                      "SHAM / CUS pour topview)"),
        ("timepoint", "Usage moyen par timepoint (M1 vs M2)"),
        ("stress",    "Usage moyen par stress (oui / non)"),
        ("angii",     "Usage moyen par ANGII (oui / non)"),
        ("sex",       "Usage moyen par sexe (M vs F)"),
    ]:
        if col not in df.columns:
            continue
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

        # Stats Mann-Whitney U par motif (seulement pour condition à 2 groupes)
        if col == "condition" and sub[col].nunique() == 2:
            stats_df = stats_by_motif(sub, col)
            if not stats_df.empty:
                stats_path = out_dir / f"stats_by_motif_{col}.csv"
                stats_df.to_csv(stats_path, index=False)
                n_sig = int(stats_df["significant_0.05"].sum())
                print(f"✓ Stats Mann-Whitney (BH-corrected) : {stats_path.name}  "
                      f"({n_sig}/{len(stats_df)} motifs significatifs à q<0.05)")

    # Agrégation par catégorie ETHOGRAM si des labels ont fourni des catégories
    if labels and any(
        isinstance(e, dict) and e.get("category") for e in labels.values()
    ):
        cat_df = aggregate_by_category(df)
        if not cat_df.empty:
            cat_path = out_dir / "usage_by_category.csv"
            cat_df.to_csv(cat_path, index=False)
            print(f"✓ Agrégation par catégorie : {cat_path.name}")

    print(f"\n✅ Analyse terminée. Tout est dans {out_dir}")


if __name__ == "__main__":
    main()
