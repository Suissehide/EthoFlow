"""
Analyse des résultats VAME en croisant avec les conditions expérimentales EthoFlow.

Pour un projet VAME donné, ce script :
  1. Liste toutes les sessions segmentées (motif_usage_<session>.npy)
  2. Récupère TOUTES les colonnes des metadata.yaml d'ethoflow — donc toutes
     les colonnes de ton Excel, que sync_from_excel.py y a recopiées
  3. Construit un DataFrame combiné session × motif × métadonnées
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

Choisir ses axes de comparaison
-------------------------------
N'importe quelle colonne de ton Excel peut servir d'axe. Le script n'en
connaît aucune à l'avance : il détecte celles qui ont entre 2 et 12 valeurs
distinctes (en dessous il n'y a rien à comparer, au-dessus c'est un
identifiant, pas un facteur expérimental).

    # 1. Voir ce qui est disponible et avec combien de sessions par valeur
    python scripts/analyze_vame.py --list-columns

    # 2. Sans rien préciser : une série de graphes pour CHAQUE colonne
    python scripts/analyze_vame.py

    # 3. Restreindre
    python scripts/analyze_vame.py --group-by captopril
    python scripts/analyze_vame.py --group-by condition captopril

    # 4. Croiser deux colonnes (design factoriel → 4 groupes)
    python scripts/analyze_vame.py --cross condition captopril

Ajouter un facteur : colonne dans l'Excel → sync_from_excel.py → --group-by
ma_colonne. Ni DLC ni VAME ne sont à relancer, les motifs sont déjà calculés.

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

# Le pointer legacy repo-root .vame_config_path a été supprimé lors du flatten
# du layout VAME (data/vame/ EST le projet VAME). On garde le nom pour la
# détection legacy uniquement, au cas où un utilisateur aurait encore ce
# fichier lui traînant.
CONFIG_POINTER = REPO_ROOT / ".vame_config_path"


def get_project_path(arg_project: str | None,
                     ethoflow_project: Path | None = None) -> Path:
    """Résout le chemin du projet VAME à analyser.

    Ordre de priorité :
      1. --project explicite (chemin absolu vers un dossier VAME)
      2. Si --project-dir (projet EthoFlow) est fourni : vame_dir(project)
         = <ethoflow_project>/data/vame/ (convention EthoFlow actuelle)
      3. Pointer legacy `<repo>/.vame_config_path` pour compat.
      4. Sinon erreur explicite.
    """
    if arg_project:
        return Path(arg_project)
    if ethoflow_project is not None:
        # Le vame_dir(project) est directement le projet VAME (flat layout).
        from paths import vame_dir  # noqa: WPS433
        vame_project = vame_dir(ethoflow_project)
        if (vame_project / "config.yaml").exists():
            return vame_project
    if CONFIG_POINTER.exists():
        return Path(CONFIG_POINTER.read_text().strip()).parent
    raise FileNotFoundError(
        "Aucun projet VAME trouvé.\n"
        "  Passe --project-dir <ethoflow_project> pour utiliser "
        "<project>/data/vame/,\n"
        "  ou --project <chemin_absolu> pour un dossier VAME spécifique."
    )


# Catégories reconnues du référentiel ETHOGRAM (cf. streamlit_app/lib/config.py).
# Utilisé pour détecter le sens des colonnes du CSV utilisateur : si `label`
# contient une catégorie et `category` un label spécifique, on swap.
_KNOWN_CATEGORIES = {
    "locomotion", "stationary", "vertical exploration", "sniffing",
    "grooming", "exploration", "arena-specific", "specific behaviors",
    "catch-all", "transitions",
}


def _map_captopril(raw) -> str | None:
    """Normalise une valeur captopril de la metadata vers un label lisible.

    - oui / yes / true / 1  → "Captopril"
    - non / no  / false / 0 → "Control"
    - autre chose (déjà normalisé, ex. "Captopril", "Control") : renvoie tel quel
    - None / vide : renvoie None

    Ce mapping est appliqué au moment de l'analyse, la metadata reste brute
    (oui/non côté YAML) pour rester alignée avec l'Excel maître.
    """
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None
    low = s.lower()
    if low in {"oui", "yes", "y", "true", "1"}:
        return "Captopril"
    if low in {"non", "no", "n", "false", "0"}:
        return "Control"
    return s  # déjà lisible


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
    """Retourne la catégorie ETHOGRAM pour ce motif, ou None.

    Un motif marqué artifact n'a pas de catégorie : la renvoyer formerait
    un faux groupe « artifact » dans les analyses par catégorie.
    """
    entry = labels.get(motif_id)
    if not isinstance(entry, dict):
        return None
    if is_artifact_motif(motif_id, labels):
        return None
    return entry.get("category")


def is_artifact_motif(motif_id: int, labels: dict[int, dict]) -> bool:
    """True si le motif est marqué artifact, donc à exclure de l'analyse.

    La marque est acceptée dans `confidence` — l'emplacement canonique —
    **ou** dans `category`, parce que le README a longtemps indiqué
    `category` et que des CSV annotés ainsi existent déjà. Ne reconnaître
    qu'une seule des deux colonnes laisse le motif compté comme un
    comportement, silencieusement.

    `artifact` n'appartient à aucune catégorie ETHOGRAM, donc la valeur est
    sans ambiguïté dans les deux colonnes. Comparaison insensible à la
    casse et aux espaces de bord.
    """
    entry = labels.get(motif_id)
    if not isinstance(entry, dict):
        return False
    for champ in ("confidence", "category"):
        valeur = entry.get(champ)
        if valeur is not None and str(valeur).strip().lower() == "artifact":
            return True
    return False


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
            captopril = meta.get("captopril")
            birth = meta.get("birth_date")
            # Fallback : si captopril absent mais birth_date présente, dérive
            # depuis l'année (convention MCC-2026-06 : 2024=oui, 2026=non).
            # À sur-écrire via patch_captopril.py si les birth_dates sont fausses.
            if captopril is None and birth is not None:
                y = str(birth).strip()[:4]
                if y == "2024":
                    captopril = "oui"
                elif y == "2026":
                    captopril = "non"

            # Toutes les colonnes de l'Excel arrivent ici via sync_from_excel
            # (il recopie chaque colonne dans metadata.yaml). On les reprend
            # telles quelles : n'importe quelle colonne que l'utilisateur
            # ajoute à son Excel devient utilisable comme axe d'analyse
            # (`--group-by ma_colonne`) sans toucher au code.
            entry = {
                k: v for k, v in meta.items()
                if not isinstance(v, (dict, list))
            }
            # Puis les alias canoniques, qui écrasent en cas de collision.
            entry.update({
                "session_id": session_id,
                "arena": "",
                "mouse_id": meta.get("mouse_id"),
                "condition": meta.get("group"),  # MCCiECKO / MCCf/f
                "captopril": captopril,
                "birth_date": birth,
                "date": meta.get("date_recorded") or meta.get("date"),
            })
            index[(session_id, "")] = entry
    return index


# Colonnes de metadata qui ne sont pas des axes d'analyse plausibles :
# identifiants uniques, chemins, champs libres. Exclues de --list-columns
# et de la détection automatique, mais utilisables si explicitement demandées.
_NON_GROUPING_COLS = {
    "session_id", "session_full", "arena", "source_video", "notes",
    "id", "motif", "label", "frequency", "count", "category",
    "empty_arena_count", "empty_arena_fraction",
}


def usable_group_columns(df: pd.DataFrame, max_groups: int = 12
                          ) -> list[tuple[str, int, list]]:
    """Colonnes du DataFrame utilisables comme axe de comparaison.

    Critère : au moins 2 valeurs distinctes (sinon il n'y a rien à comparer)
    et pas plus de `max_groups` (au-delà c'est un identifiant, pas un
    facteur expérimental).

    Retourne [(colonne, n_groupes, valeurs)] trié par nombre de groupes.
    """
    out = []
    seen_signatures: dict[tuple, str] = {}
    for col in df.columns:
        if col in _NON_GROUPING_COLS:
            continue
        vals = df[col].dropna().unique()
        if not (2 <= len(vals) <= max_groups):
            continue
        # Dédoublonnage : `group` et `condition` sont le même champ sous deux
        # noms (alias historique), pareil pour `date` / `date_recorded`.
        # Produire deux fois les mêmes figures n'apporte rien.
        signature = tuple(df[col].astype("string").fillna("␀"))
        if signature in seen_signatures:
            continue
        seen_signatures[signature] = col
        out.append((col, len(vals), sorted(str(v) for v in vals)))
    return sorted(out, key=lambda t: (t[1], t[0]))


# Libellés lisibles pour les colonnes connues. Toute colonne absente d'ici
# est affichée telle quelle — un nom d'axe est toujours le nom de la colonne
# Excel, donc l'utilisateur reconnaît sa propre nomenclature.
_COLUMN_TITLES = {
    "condition": "génotype",
    "group": "génotype",
    "captopril": "traitement Captopril",
    "group4": "génotype × Captopril",
    "sex": "sexe",
    "cage": "cage",
    "timepoint": "timepoint",
    "stress": "stress",
    "angii": "ANGII",
    "line": "lignée",
    "date": "date d'enregistrement",
}


def group_title(col: str, n_groups: int) -> str:
    """Titre de figure pour un axe de comparaison."""
    nice = _COLUMN_TITLES.get(col, col.replace("_x_", " × ").replace("_", " "))
    return f"Usage moyen par {nice} ({n_groups} groupes)"


def make_cross_column(df: pd.DataFrame, cols: list[str]) -> str:
    """Crée une colonne composite `a_x_b` dans df et retourne son nom.

    Sert aux designs factoriels : croiser génotype × traitement donne les
    4 groupes de l'analyse 2×2. Les lignes où l'une des colonnes est
    manquante restent NaN et seront filtrées par le dropna en aval.
    """
    name = "_x_".join(cols)
    parts = [df[c].astype("string") for c in cols]
    combined = parts[0]
    for p in parts[1:]:
        combined = combined + "_" + p
    # `+` sur des StringDtype propage déjà NaN, mais on force pour être sûr
    mask = df[cols].isna().any(axis=1)
    df[name] = combined.mask(mask)
    return name


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

            # Groupe composite 4-way pour bottomview : condition × captopril.
            # `captopril` est normalisé oui/non → Captopril/Control pour tous
            # les affichages et regroupements. None si l'une des deux est
            # manquante — les lignes seront filtrées auto par dropna en aval.
            cond = meta.get("condition")
            capto = _map_captopril(meta.get("captopril"))
            group4 = f"{cond}_{capto}" if (cond and capto) else None

            row = {
                "session_full": session_name,
                "session_id": session_id,
                "arena": arena,
                "mouse_id": meta.get("mouse_id"),
                "condition": cond,
                # Bottomview : traitement pharmacologique
                "captopril": capto,
                "group4": group4,
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
            # Toute autre colonne de la metadata (donc de l'Excel) est
            # recopiée telle quelle. C'est ce qui rend `--group-by` ouvert :
            # ajoute une colonne `traitement` dans ton Excel, resynchronise,
            # et `--group-by traitement` marche sans modifier ce script.
            for k, v in meta.items():
                if k not in row and not isinstance(v, (dict, list)):
                    row[k] = v

            if per_motif_empty is not None:
                orig = int(usage_original[motif_i])
                row["empty_arena_count"] = int(per_motif_empty[motif_i])
                row["empty_arena_fraction"] = (
                    per_motif_empty[motif_i] / orig if orig > 0 else 0.0
                )
            rows.append(row)
    return pd.DataFrame(rows), pd.DataFrame(validity_rows)


def aggregate_by_category(df: pd.DataFrame,
                           extra_keys: list[str] | None = None) -> pd.DataFrame:
    """Agrège la fréquence par catégorie ETHOGRAM (ex: 'Sniffing') plutôt
    que par motif individuel. Ignore les motifs sans catégorie assignée.

    Retourne un DataFrame avec 1 ligne par session × catégorie.
    """
    df_cat = df[df["category"].notna()].copy()
    if df_cat.empty:
        return pd.DataFrame()
    # `captopril` et `group4` sont conservés dans le groupby s'ils existent
    # (bottomview) — un groupby sur des colonnes absentes casse en pandas.
    keys = ["session_full", "session_id", "arena"]
    optional = ["condition", "captopril", "group4", "sex", "cage", "date"]
    optional += [c for c in (extra_keys or []) if c not in optional]
    for opt in optional:
        if opt in df_cat.columns and opt not in keys:
            keys.append(opt)
    keys.append("category")
    agg = (
        df_cat.groupby(keys, dropna=False)["frequency"]
        .sum()
        .reset_index()
        .rename(columns={"frequency": "frequency_total"})
    )
    return agg


def load_per_frame_labels(seg_file: Path) -> np.ndarray | None:
    """Charge le fichier de labels par frame associé à un motif_usage."""
    lf = find_label_file(seg_file)
    if lf is None or not lf.exists():
        return None
    return np.load(lf).astype(int)


def compute_transition_matrix(labels_per_frame: np.ndarray,
                              n_motifs: int) -> np.ndarray:
    """Matrice T[i,j] = P(motif_{t+1} = j | motif_t = i), self-loops exclus."""
    T = np.zeros((n_motifs, n_motifs), dtype=float)
    curr = labels_per_frame[:-1]
    nxt = labels_per_frame[1:]
    mask = curr != nxt  # ignore self-loops (durée dans motif = non-transition)
    for i, j in zip(curr[mask], nxt[mask]):
        if 0 <= i < n_motifs and 0 <= j < n_motifs:
            T[i, j] += 1
    row_sums = T.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    return T / row_sums


def plot_transition_matrix(T: np.ndarray, title: str, out_path: Path,
                           labels: dict[int, dict]) -> None:
    n = T.shape[0]
    labs = [motif_display(i, labels) for i in range(n)]
    fig, ax = plt.subplots(figsize=(max(8, 0.5 * n), max(6, 0.5 * n)))
    im = ax.imshow(T, cmap="viridis", aspect="auto", vmin=0)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labs, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(labs, fontsize=8)
    ax.set_xlabel("Motif suivant")
    ax.set_ylabel("Motif courant")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="P(next | current)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def compute_bouts(labels_per_frame: np.ndarray, fps: float) -> pd.DataFrame:
    """Renvoie une liste de bouts (motif, duration_sec)."""
    if len(labels_per_frame) == 0:
        return pd.DataFrame(columns=["motif", "duration_sec"])
    changes = np.diff(labels_per_frame) != 0
    boundaries = np.concatenate(([0], np.where(changes)[0] + 1, [len(labels_per_frame)]))
    rows = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        rows.append({
            "motif": int(labels_per_frame[start]),
            "duration_sec": (end - start) / fps,
        })
    return pd.DataFrame(rows)


def compute_temporal_quarters(labels_per_frame: np.ndarray,
                              n_motifs: int,
                              n_quarters: int = 4) -> dict[int, np.ndarray]:
    """Fréquence par motif dans chaque quart de session."""
    result = {}
    L = len(labels_per_frame)
    edges = np.linspace(0, L, n_quarters + 1, dtype=int)
    for q in range(n_quarters):
        seg = labels_per_frame[edges[q]:edges[q + 1]]
        if len(seg) == 0:
            result[q] = np.zeros(n_motifs)
            continue
        counts = np.bincount(seg, minlength=n_motifs).astype(float)
        result[q] = counts / counts.sum() if counts.sum() > 0 else counts
    return result


def compute_spatial_time_in_center(project_ethoflow: Path, session_id: str,
                                    labels_per_frame: np.ndarray,
                                    center_frac: float = 0.5) -> dict:
    """Temps en centre par motif à partir du h5 nettoyé (tail_base).

    center_frac : fraction du diamètre de l'arène considérée comme "centre"
                  (défaut 0.5 = disque central couvrant 25% de l'aire).
    """
    from paths import cleaned_h5_path
    h5 = cleaned_h5_path(project_ethoflow, session_id)
    if not h5.exists():
        return {}
    df = pd.read_hdf(h5)
    # Cherche tail_base x,y (peu importe le scorer)
    tail_x_col = tail_y_col = None
    for col in df.columns:
        if not isinstance(col, tuple) or len(col) < 3:
            continue
        bp, coord = col[-2], col[-1]
        if bp == "tail_base" and coord == "x":
            tail_x_col = col
        elif bp == "tail_base" and coord == "y":
            tail_y_col = col
    if tail_x_col is None or tail_y_col is None:
        return {}
    x = df[tail_x_col].values
    y = df[tail_y_col].values
    # Centre de l'arène = médiane des positions (robuste aux NaN)
    cx = np.nanmedian(x)
    cy = np.nanmedian(y)
    # Rayon de l'arène = 95e percentile de la distance au centre
    dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    arena_radius = np.nanpercentile(dist, 95)
    center_radius = arena_radius * center_frac
    in_center = dist < center_radius  # bool per frame
    # Aligne les longueurs — labels_per_frame et in_center devraient avoir la même
    # longueur, mais des décalages ±1 arrivent parfois. Coupe au min.
    L = min(len(labels_per_frame), len(in_center))
    labels_L = labels_per_frame[:L]
    in_center_L = in_center[:L]
    # % dans le centre global + par motif
    result = {
        "in_center_frac_total": float(np.nanmean(in_center_L)),
        "arena_radius_px": float(arena_radius),
    }
    for m in np.unique(labels_L):
        mask = labels_L == m
        n = int(mask.sum())
        if n == 0:
            continue
        result[f"motif_{int(m)}_in_center_frac"] = (
            float(np.nanmean(in_center_L[mask])) if n > 0 else 0.0
        )
    return result


def stats_by_motif(df: pd.DataFrame, condition_col: str) -> pd.DataFrame:
    """Tests statistiques par motif entre les groupes de `condition_col`.

    - 2 groupes → Mann-Whitney U (colonne stat = 'u_stat')
    - ≥3 groupes → Kruskal-Wallis H (colonne stat = 'h_stat')

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
    if len(groups) < 2:
        print(f"⚠  stats_by_motif : {len(groups)} groupe(s) dans "
              f"'{condition_col}' — il en faut au moins 2.", file=sys.stderr)
        return pd.DataFrame()

    is_multi = len(groups) > 2
    rows = []
    for m in sorted(sub["motif"].unique()):
        d = sub[sub["motif"] == m]
        samples = [d.loc[d[condition_col] == g, "frequency"].values for g in groups]
        # Chaque groupe doit avoir ≥3 obs pour un test défensif
        if any(len(s) < 3 for s in samples):
            continue
        try:
            if is_multi:
                stat, p = stats.kruskal(*samples)
                stat_key = "h_stat"
            else:
                stat, p = stats.mannwhitneyu(samples[0], samples[1],
                                             alternative="two-sided")
                stat_key = "u_stat"
        except ValueError:
            continue
        row = {
            "motif": m,
            "label": d["label"].iloc[0],
            "category": d["category"].iloc[0] if "category" in d.columns else None,
        }
        for g, s in zip(groups, samples):
            row[f"mean_{g}"] = float(np.mean(s))
            row[f"n_{g}"] = len(s)
        if not is_multi:
            row["diff"] = float(np.mean(samples[1]) - np.mean(samples[0]))
        row[stat_key] = float(stat)
        row["p_value"] = float(p)
        rows.append(row)
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
    """Heatmap sessions × motifs, sessions triées alphabétiquement.

    Version historique (aucun regroupement). Conservée pour compat.
    Voir plot_heatmap_grouped pour la version regroupée par condition.
    """
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


def plot_heatmap_grouped(df: pd.DataFrame, group_col: str, out_path: Path,
                          labels: dict[int, str]) -> None:
    """Heatmap sessions × motifs, sessions triées ET séparées par `group_col`.

    Différences avec plot_heatmap :
      1. Les sessions sont triées par valeur de `group_col`, puis par nom.
         Toutes les MCCiECKO d'affilée, puis toutes les MCCf/f, etc.
      2. Un bandeau coloré à gauche indique le groupe de chaque session
         (une couleur par valeur unique du `group_col`).
      3. Des lignes noires horizontales séparent visuellement les groupes.
      4. Les tick labels des sessions sont préfixés par leur groupe pour
         lecture rapide (ex. "MCCiECKO_Captopril | BV-970").
    """
    sub = df.dropna(subset=[group_col])
    if sub.empty:
        return
    # 1 valeur de group_col par session
    session_group = (
        sub[["session_full", group_col]]
        .drop_duplicates()
        .set_index("session_full")[group_col]
    )
    # Tri stable : d'abord par groupe, puis par nom de session
    session_order = (
        session_group
        .sort_values(kind="stable")
        .index
        .tolist()
    )
    pivot = sub.pivot_table(
        index="session_full", columns="motif", values="frequency", aggfunc="mean"
    ).reindex(session_order)

    # Palette : une couleur par groupe unique, dans l'ordre alphabétique
    groups_sorted = sorted(session_group.unique())
    # Palette qualitative fiable jusqu'à 8 groupes
    base_palette = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
        "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
    ]
    color_by_group = {g: base_palette[i % len(base_palette)]
                      for i, g in enumerate(groups_sorted)}
    row_colors = [color_by_group[session_group[s]] for s in session_order]

    # Layout : bandeau couleurs (col 0, largeur fixe) + heatmap (col 1)
    fig = plt.figure(figsize=(13, max(6, 0.28 * len(pivot))))
    gs = fig.add_gridspec(1, 2, width_ratios=[0.03, 1.0], wspace=0.02)
    ax_band = fig.add_subplot(gs[0, 0])
    ax = fig.add_subplot(gs[0, 1], sharey=ax_band)

    # Bandeau couleurs : imshow d'une colonne RGB
    from matplotlib.colors import to_rgb
    band = np.array([to_rgb(c) for c in row_colors]).reshape(-1, 1, 3)
    ax_band.imshow(band, aspect="auto")
    ax_band.set_xticks([])
    ax_band.set_yticks(range(len(session_order)))
    ax_band.set_yticklabels(
        [f"{session_group[s]} | {s}" for s in session_order],
        fontsize=7,
    )
    ax_band.set_ylabel(group_col, fontsize=9)

    # Heatmap principale
    im = ax.imshow(pivot.values, aspect="auto", cmap="viridis")
    ax.set_yticks([])  # tick labels sont côté bandeau
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(
        [motif_display(int(m), labels) for m in pivot.columns],
        fontsize=8, rotation=45, ha="right",
    )
    ax.set_xlabel("Motif")

    # Lignes horizontales entre groupes
    boundaries = []
    prev_g = None
    for i, s in enumerate(session_order):
        g = session_group[s]
        if prev_g is not None and g != prev_g:
            boundaries.append(i - 0.5)
        prev_g = g
    for b in boundaries:
        ax.axhline(b, color="black", linewidth=1.2)
        ax_band.axhline(b, color="black", linewidth=1.2)

    ax.set_title(f"Usage par session (regroupé par {group_col})")

    # Colorbar à droite (créée sur ax pour être bien alignée)
    fig.colorbar(im, ax=ax, label="proportion", fraction=0.03, pad=0.02)

    # Légende des groupes en bas
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=color_by_group[g], label=str(g))
               for g in groups_sorted]
    fig.legend(
        handles=handles, title=group_col,
        loc="lower center", bbox_to_anchor=(0.5, -0.02),
        ncol=min(len(groups_sorted), 4), frameon=False, fontsize=9,
    )

    fig.savefig(out_path, dpi=120, bbox_inches="tight")
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
    groups = sorted(df[condition_col].dropna().unique())
    n_groups = len(groups)
    # Grille : max 3 colonnes, autant de lignes que nécessaire. 6 motifs
    # → 2×3 (plus lisible qu'un long 1×6 quand les labels sont longs).
    ncols = min(3, len(motifs))
    nrows = (len(motifs) + ncols - 1) // ncols
    # Largeur adaptative : les labels de group4 (MCCiECKO_Captopril...) sont
    # longs. Compte le pire label pour ajuster la largeur par subplot.
    max_label_len = max((len(str(g)) for g in groups), default=1)
    per_subplot_w = max(2.8, 0.15 * max_label_len * n_groups / 2)
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(per_subplot_w * ncols, 4.2 * nrows),
        sharey=True,
    )
    # Normalise en array 1D pour itérer proprement quelque soit la forme
    axes = np.atleast_1d(axes).flatten()
    for ax, motif in zip(axes, motifs):
        d = df[df["motif"] == motif]
        data = [d.loc[d[condition_col] == g, "frequency"].values for g in groups]
        ax.boxplot(data, tick_labels=[str(g) for g in groups])
        # Rotation adaptative selon la longueur du plus grand label :
        # 30° pour des labels courts (Captopril, Control, F, M...)
        # 45° pour des labels longs (MCCiECKO_Captopril, etc.)
        rot = 45 if max_label_len > 12 else 30
        # `ha="right"` aligne le coin haut-droit du texte avec le tick,
        # empêche que les labels rentrent sous le subplot voisin
        plt.setp(ax.get_xticklabels(), rotation=rot, ha="right",
                 fontsize=8)
        ax.set_title(motif_display(motif, labels), fontsize=10)
        ax.set_ylabel("Proportion")
    # Cache les subplots vides (ex : 5 motifs sur grille 2×3 → 1 case vide)
    for ax in axes[len(motifs):]:
        ax.set_visible(False)
    fig.suptitle(f"Top motifs différenciants par {condition_col}", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
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
    parser.add_argument("--extended", action="store_true",
                        help="Analyses étendues (~5-10 min supplémentaires) : "
                             "matrices de transitions par groupe, durée moyenne "
                             "de bout par motif, dynamique temporelle (4 quarts), "
                             "analyse spatiale (center/periphery via tail_base).")
    parser.add_argument("--extended-by", default="condition",
                        help="Colonne de groupement pour les analyses "
                             "étendues (défaut : condition). N'importe quelle "
                             "colonne de l'Excel, ou un croisement déclaré "
                             "avec --cross (ex : genotype_x_captopril).")
    parser.add_argument("--group-by", nargs="+", default=None, metavar="COL",
                        help="Colonnes de l'Excel à utiliser comme axes de "
                             "comparaison. Une série complète de graphes est "
                             "produite pour chacune. Défaut : toutes les "
                             "colonnes exploitables détectées automatiquement "
                             "(2 à 12 valeurs distinctes).")
    parser.add_argument("--cross", nargs="+", action="append", default=None,
                        metavar="COL",
                        help="Croise 2 colonnes (ou plus) en un facteur "
                             "composite, ex : --cross group captopril donne "
                             "les 4 groupes MCCf/f_Control, MCCf/f_Captopril, "
                             "etc. Répétable pour plusieurs croisements.")
    parser.add_argument("--list-columns", action="store_true",
                        help="Affiche les colonnes disponibles (issues de ton "
                             "Excel via les metadata) avec leurs valeurs et le "
                             "nombre de sessions par valeur, puis quitte. "
                             "À lancer en premier pour savoir quoi passer à "
                             "--group-by.")
    parser.add_argument("--fps", type=float, default=30.0,
                        help="Framerate des vidéos source (pour la conversion "
                             "durée frame → secondes). Défaut : 30.")
    args = parser.parse_args()

    ethoflow_project = resolve_project(args)

    try:
        project = get_project_path(args.project, ethoflow_project)
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

    # ------------------------------------------------------------------
    # Résolution des axes de comparaison
    # ------------------------------------------------------------------
    # Toute colonne de l'Excel est candidate (sync_from_excel les recopie
    # dans metadata.yaml, build_dataframe les recopie dans le DataFrame).
    # --cross fabrique des facteurs composites, --group-by restreint la
    # liste, sinon on détecte automatiquement ce qui est exploitable.
    cross_names = []
    for combo in (args.cross or []):
        missing = [c for c in combo if c not in df.columns]
        if missing:
            print(f"❌ --cross {' '.join(combo)} : colonne(s) inconnue(s) "
                  f"{missing}. Lance --list-columns pour voir les colonnes "
                  f"disponibles.", file=sys.stderr)
            sys.exit(1)
        cross_names.append(make_cross_column(df, combo))
        print(f"  → facteur croisé '{cross_names[-1]}' : "
              f"{df[cross_names[-1]].nunique()} groupes")

    # `group4` historique : équivaut à --cross condition captopril. Conservé
    # tel quel pour ne pas casser les analyses déjà produites.
    available = usable_group_columns(df)
    available_names = [c for c, _, _ in available]

    if args.list_columns:
        print(f"\nColonnes exploitables comme axe de comparaison "
              f"({len(available)}) :\n")
        for col, n, vals in available:
            counts = (df.drop_duplicates("session_full")[col]
                      .value_counts(dropna=True))
            detail = ", ".join(f"{v} ({counts.get(v, 0)} sessions)"
                                for v in counts.index[:6])
            if len(counts) > 6:
                detail += ", …"
            print(f"  {col:<24} {n} groupes : {detail}")
        print("\nExemples :")
        if available_names:
            print(f"  python scripts/analyze_vame.py --group-by "
                  f"{available_names[0]}")
        if len(available_names) >= 2:
            print(f"  python scripts/analyze_vame.py --cross "
                  f"{available_names[0]} {available_names[1]}")
        print("\nUne colonne absente d'ici a soit une seule valeur (rien à "
              "comparer),\nsoit plus de 12 (c'est un identifiant, pas un "
              "facteur expérimental).")
        sys.exit(0)

    if args.group_by:
        unknown = [c for c in args.group_by if c not in df.columns]
        if unknown:
            print(f"❌ Colonne(s) inconnue(s) : {unknown}\n"
                  f"   Disponibles : {', '.join(available_names)}\n"
                  f"   (ou lance --list-columns pour le détail)",
                  file=sys.stderr)
            sys.exit(1)
        group_cols = list(args.group_by) + cross_names
    else:
        group_cols = available_names + [c for c in cross_names
                                        if c not in available_names]

    print(f"  → axes de comparaison : {', '.join(group_cols) or '(aucun)'}")

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

    # Heatmap "brute" (sessions triées alphabétiquement, aucun regroupement).
    # Utile pour un audit global mais peu lisible dès qu'il y a plusieurs
    # conditions — les heatmaps groupées ci-dessous sont plus parlantes.
    plot_heatmap(df, out_dir / "heatmap_usage.png", labels)
    print(f"✓ Heatmap : heatmap_usage.png")

    # Heatmaps regroupées par condition / captopril / group4 (bottomview).
    # Chaque heatmap = sessions triées par groupe, bandeau couleur à gauche,
    # séparateurs entre groupes. Beaucoup plus lisible pour repérer les
    # motifs qui distinguent visuellement les groupes.
    for col in group_cols:
        sub = df.dropna(subset=[col])
        if sub.empty or sub[col].nunique() < 2:
            continue
        heat_path = out_dir / f"heatmap_usage_by_{col}.png"
        plot_heatmap_grouped(sub, col, heat_path, labels)
        print(f"✓ Heatmap groupée par {col} : {heat_path.name}")

    # Comparaisons par groupe : une série complète (barres + boxplots + stats)
    # pour chaque axe. Les colonnes non renseignées dans la metadata sont
    # skippées automatiquement (dropna + nunique >= 2).
    for col in group_cols:
        sub = df.dropna(subset=[col])
        if sub.empty or sub[col].nunique() < 2:
            continue
        title = group_title(col, sub[col].nunique())
        plot_path = out_dir / f"mean_by_{col}.png"
        plot_means_by_condition(sub, col, plot_path, title, labels)
        print(f"✓ Barres par {col} ({sub[col].nunique()} groupes) : {plot_path.name}")

        top = differentiating_motifs(sub, col, top_k=6)
        if top:
            box_path = out_dir / f"boxplots_top_by_{col}.png"
            plot_boxplots(sub, col, top, box_path, labels)
            print(f"  → boxplots top motifs : {box_path.name} (motifs {top})")

        # Stats par motif : Mann-Whitney (2 groupes) ou Kruskal-Wallis (≥3)
        stats_df = stats_by_motif(sub, col)
        if not stats_df.empty:
            stats_path = out_dir / f"stats_by_motif_{col}.csv"
            stats_df.to_csv(stats_path, index=False)
            n_sig = int(stats_df["significant_0.05"].sum())
            test_name = ("Kruskal-Wallis" if sub[col].nunique() > 2
                         else "Mann-Whitney")
            print(f"✓ Stats {test_name} (BH-corrected) : {stats_path.name}  "
                  f"({n_sig}/{len(stats_df)} motifs significatifs à q<0.05)")

    # Agrégation par catégorie ETHOGRAM si des labels ont fourni des catégories
    has_categories = labels and any(
        isinstance(e, dict) and e.get("category") for e in labels.values()
    )
    if has_categories:
        cat_df = aggregate_by_category(df, extra_keys=group_cols)
        if not cat_df.empty:
            cat_path = out_dir / "usage_by_category.csv"
            cat_df.to_csv(cat_path, index=False)
            print(f"✓ Agrégation par catégorie : {cat_path.name}")

            # Plots + stats par catégorie × chaque axe de comparaison.
            # Un fichier par axe ; l'utilisateur les compare visuellement.
            for grp_col in group_cols:
                grp_title = _COLUMN_TITLES.get(
                    grp_col, grp_col.replace("_x_", " × ").replace("_", " "))
                if grp_col not in cat_df.columns:
                    continue
                sub_cat = cat_df.dropna(subset=[grp_col])
                if sub_cat.empty or sub_cat[grp_col].nunique() < 2:
                    continue

                # ---- Bar plot catégorie × groupe ----
                fig, ax = plt.subplots(
                    figsize=(max(8, 0.6 * sub_cat["category"].nunique()), 5)
                )
                pivot = (
                    sub_cat.groupby(["category", grp_col])["frequency_total"]
                    .mean().unstack(grp_col)
                )
                pivot.plot(kind="bar", ax=ax)
                ax.set_ylabel("Proportion moyenne (somme des motifs par catégorie)")
                ax.set_title(f"Usage par catégorie ETHOGRAM × {grp_title}")
                ax.set_xlabel("Catégorie")
                ax.legend(title=grp_col, fontsize=8)
                plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
                fig.tight_layout()
                bar_path = out_dir / f"mean_by_category_by_{grp_col}.png"
                fig.savefig(bar_path, dpi=120)
                plt.close(fig)
                print(f"✓ Barres catégorie × {grp_col} : {bar_path.name}")

                # ---- Boxplots par catégorie × groupe ----
                cats = sorted(sub_cat["category"].dropna().unique())
                groups = sorted(sub_cat[grp_col].dropna().unique())
                fig, axes = plt.subplots(
                    1, len(cats), figsize=(3 * len(cats), 4), sharey=True
                )
                if len(cats) == 1:
                    axes = [axes]
                for ax, cat in zip(axes, cats):
                    d = sub_cat[sub_cat["category"] == cat]
                    data = [
                        d.loc[d[grp_col] == g, "frequency_total"].values
                        for g in groups
                    ]
                    ax.boxplot(data, tick_labels=[str(g) for g in groups])
                    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=8)
                    ax.set_title(cat, fontsize=10)
                    ax.set_ylabel("Proportion")
                fig.suptitle(f"Distribution par catégorie × {grp_title}")
                fig.tight_layout()
                box_path = out_dir / f"boxplots_by_category_by_{grp_col}.png"
                fig.savefig(box_path, dpi=120)
                plt.close(fig)
                print(f"  → boxplots catégorie × {grp_col} : {box_path.name}")

                # ---- Stats par catégorie (Mann-Whitney ou Kruskal-Wallis) ----
                try:
                    from scipy import stats as _stats
                    is_multi = len(groups) > 2
                    rows = []
                    for cat in cats:
                        d = sub_cat[sub_cat["category"] == cat]
                        samples = [
                            d.loc[d[grp_col] == g, "frequency_total"].values
                            for g in groups
                        ]
                        if any(len(s) < 3 for s in samples):
                            continue
                        try:
                            if is_multi:
                                stat, p = _stats.kruskal(*samples)
                                stat_key = "h_stat"
                            else:
                                stat, p = _stats.mannwhitneyu(
                                    samples[0], samples[1], alternative="two-sided"
                                )
                                stat_key = "u_stat"
                        except ValueError:
                            continue
                        row = {"category": cat}
                        for g, s in zip(groups, samples):
                            row[f"mean_{g}"] = float(np.mean(s))
                            row[f"n_{g}"] = len(s)
                        row[stat_key] = float(stat)
                        row["p_value"] = float(p)
                        rows.append(row)
                    if rows:
                        stats_cat = pd.DataFrame(rows).sort_values("p_value")
                        # BH correction
                        ps = stats_cat["p_value"].values
                        nps = len(ps)
                        order = np.argsort(ps)
                        ranked = ps[order]
                        q = np.zeros(nps); prev = 1.0
                        for i in range(nps - 1, -1, -1):
                            prev = min(prev, ranked[i] * nps / (i + 1))
                            q[i] = prev
                        q_full = np.zeros(nps); q_full[order] = q
                        stats_cat["p_adj_bh"] = q_full
                        stats_cat["significant_0.05"] = stats_cat["p_adj_bh"] < 0.05
                        stats_path = out_dir / f"stats_by_category_by_{grp_col}.csv"
                        stats_cat.to_csv(stats_path, index=False)
                        test_name = "Kruskal-Wallis" if is_multi else "Mann-Whitney"
                        n_sig = int(stats_cat["significant_0.05"].sum())
                        print(f"  → Stats {test_name} : {stats_path.name} "
                              f"({n_sig}/{len(stats_cat)} catégories q<0.05)")
                except ImportError:
                    pass

    # =========================================================================
    # Analyses étendues (--extended)
    # =========================================================================
    if args.extended:
        group_col = args.extended_by
        if group_col not in df.columns:
            print(f"❌ --extended-by {group_col} : colonne inconnue.\n"
                  f"   Disponibles : {', '.join(available_names + cross_names)}",
                  file=sys.stderr)
            sys.exit(1)
        print(f"\n=== Analyses étendues (groupement : {group_col}) ===")
        n_motifs = df["motif"].nunique()

        # La valeur du groupe est relue depuis le DataFrame (qui contient
        # déjà toutes les colonnes de l'Excel + les facteurs croisés), donc
        # --extended-by accepte n'importe quelle colonne sans branchement.
        group_of_session = (
            df.drop_duplicates("session_full")
              .set_index("session_full")[group_col].to_dict()
        )

        # Pré-charge labels-per-frame et groupe pour chaque session.
        session_data = {}
        for f in seg_files:
            lpf = load_per_frame_labels(f)
            if lpf is None:
                continue
            session_name = f.parent.parent.parent.name
            sid, arena = parse_session_name(session_name)
            meta = meta_index.get((sid, arena), {})
            group_val = group_of_session.get(session_name)
            if group_val is not None and pd.isna(group_val):
                group_val = None
            session_data[session_name] = {
                "labels_per_frame": lpf,
                "condition": group_val,  # sert de clé de groupement générique
                "session_id": sid,
            }
        print(f"  → {len(session_data)} sessions avec labels par frame chargés")

        # Note : les matrices de transitions ont été retirées de l'analyse
        # standard. Pour une version labellisée et propre au niveau projet,
        # utilise `scripts/community_dendrogram.py` qui synthétise la même
        # information (proximité des motifs par transitions) de manière plus
        # lisible qu'une matrice N×N.

        # (2) Durée moyenne de bout par motif × groupe -----------------------
        bouts_rows = []
        for sname, s in session_data.items():
            b = compute_bouts(s["labels_per_frame"], args.fps)
            b["session_full"] = sname
            b["condition"] = s["condition"]
            bouts_rows.append(b)
        if bouts_rows:
            bouts_df = pd.concat(bouts_rows, ignore_index=True)
            bout_summary = (
                bouts_df.groupby(["condition", "motif"])["duration_sec"]
                .agg(["mean", "median", "count"])
                .reset_index()
            )
            bout_csv = out_dir / f"bout_durations_by_{group_col}.csv"
            bout_summary.to_csv(bout_csv, index=False)
            print(f"  ✓ Durées de bout : {bout_csv.name}")
            # Plot
            if "condition" in bout_summary.columns and bout_summary["condition"].nunique() >= 2:
                fig, ax = plt.subplots(figsize=(max(8, 0.6 * n_motifs), 5))
                pivot = bout_summary.groupby(["motif", "condition"])["mean"].first().unstack("condition")
                pivot.index = [motif_display(int(m), labels) for m in pivot.index]
                pivot.plot(kind="bar", ax=ax)
                ax.set_ylabel("Durée moyenne d'un bout (s)")
                ax.set_title(f"Persistance dans chaque motif (par {group_col})")
                ax.set_xlabel("Motif")
                ax.legend(title=group_col, fontsize=8)
                plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
                fig.tight_layout()
                bout_png = out_dir / f"bout_duration_by_{group_col}.png"
                fig.savefig(bout_png, dpi=120)
                plt.close(fig)
                print(f"  ✓ Barres durées : {bout_png.name}")

        # (3) Dynamique temporelle par quart de session ----------------------
        temporal_rows = []
        for sname, s in session_data.items():
            quarters = compute_temporal_quarters(s["labels_per_frame"], n_motifs, 4)
            for q, freqs in quarters.items():
                for m, f in enumerate(freqs):
                    temporal_rows.append({
                        "session_full": sname,
                        "condition": s["condition"],
                        "quarter": q + 1,
                        "motif": m,
                        "frequency": float(f),
                    })
        if temporal_rows:
            tmp_df = pd.DataFrame(temporal_rows)
            tmp_csv = out_dir / f"temporal_quarters_by_{group_col}.csv"
            tmp_df.to_csv(tmp_csv, index=False)
            print(f"  ✓ Dynamique temporelle : {tmp_csv.name}")

            # Plot par motif : moyenne par quart × groupe
            if tmp_df["condition"].nunique() >= 2:
                cols_per_row = 4
                rows_needed = (n_motifs + cols_per_row - 1) // cols_per_row
                fig, axes = plt.subplots(rows_needed, cols_per_row,
                                          figsize=(4 * cols_per_row, 2.5 * rows_needed),
                                          sharex=True)
                axes = axes.flatten() if rows_needed > 1 else np.atleast_1d(axes)
                for m in range(n_motifs):
                    ax = axes[m]
                    d = tmp_df[tmp_df["motif"] == m]
                    for cond, sub in d.groupby("condition"):
                        agg = sub.groupby("quarter")["frequency"].agg(["mean", "std"])
                        ax.errorbar(agg.index, agg["mean"], yerr=agg["std"],
                                     marker="o", label=cond, capsize=3)
                    ax.set_title(motif_display(m, labels), fontsize=9)
                    ax.set_xlabel("Quart de session"); ax.set_xticks([1, 2, 3, 4])
                    ax.set_ylabel("Freq.", fontsize=8)
                    ax.legend(fontsize=7)
                for m in range(n_motifs, len(axes)):
                    axes[m].set_visible(False)
                fig.suptitle("Évolution des motifs au cours de la session")
                fig.tight_layout()
                tmp_png = out_dir / f"temporal_by_motif_{group_col}.png"
                fig.savefig(tmp_png, dpi=120)
                plt.close(fig)
                print(f"  ✓ Évolution temporelle : {tmp_png.name}")

        # (4) Analyse spatiale (center vs periphery) -------------------------
        spatial_rows = []
        for sname, s in session_data.items():
            spatial = compute_spatial_time_in_center(
                ethoflow_project, s["session_id"], s["labels_per_frame"]
            )
            if spatial:
                spatial["session_full"] = sname
                spatial["condition"] = s["condition"]
                spatial_rows.append(spatial)
        if spatial_rows:
            spatial_df = pd.DataFrame(spatial_rows)
            spat_csv = out_dir / f"spatial_center_periphery_by_{group_col}.csv"
            spatial_df.to_csv(spat_csv, index=False)
            print(f"  ✓ Spatial : {spat_csv.name}")

            # Barplot thigmotaxie (temps dans le centre) par groupe.
            # Palette étendue à 4 couleurs pour group4.
            if ("in_center_frac_total" in spatial_df.columns
                    and spatial_df["condition"].nunique() >= 2):
                fig, ax = plt.subplots(figsize=(max(6, 1.5 * spatial_df["condition"].nunique()), 4))
                grouped = spatial_df.groupby("condition")["in_center_frac_total"]
                means = grouped.mean(); errs = grouped.sem()
                palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"][:len(means)]
                means.plot(kind="bar", yerr=errs, ax=ax, color=palette, capsize=5)
                ax.set_ylabel("Fraction du temps dans le centre")
                ax.set_title(f"Thigmotaxie par {group_col} (moins de centre = plus de thigmotaxie)")
                ax.set_xlabel(group_col)
                plt.setp(ax.get_xticklabels(), rotation=0)
                fig.tight_layout()
                thig_png = out_dir / f"thigmotaxis_by_{group_col}.png"
                fig.savefig(thig_png, dpi=120)
                plt.close(fig)
                print(f"  ✓ Thigmotaxie : {thig_png.name}")

    print(f"\n✅ Analyse terminée. Tout est dans {out_dir}")


if __name__ == "__main__":
    main()
