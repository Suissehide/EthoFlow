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

    # Side-by-side avec la vidéo réelle de la souris à gauche
    python scripts/behavior_structure_gif.py \\
        --project-dir <...> --session BV-970 \\
        --with-video --start 120 --duration 30 --output-format mp4

    # Manifold POOLÉ sur toutes les sessions du projet (référentiel commun).
    # ~5-15 min pour fit UMAP sur ~1M points. Le cache est nommé par session
    # ANIMÉE (elle seule est gardée en full-res, ce qui change les données
    # envoyées à UMAP) : relancer sur la même session est instantané, sur
    # une autre session refait l'ajustement.
    python scripts/behavior_structure_gif.py \\
        --project-dir <...> --session BV-970 \\
        --pool-all-sessions --with-video --output-format mp4

    # Idem, mais le nuage de fond est restreint aux sessions qui partagent
    # la `condition` de BV-970 — une figure par groupe expérimental, avec
    # des axes identiques d'une figure à l'autre (l'ajustement UMAP, lui,
    # reste commun à tout le projet).
    python scripts/behavior_structure_gif.py \\
        --project-dir <...> --session BV-970 \\
        --pool-by condition --with-video --output-format mp4
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from interactive import prompt_session  # noqa: E402
from paths import add_project_dir_arg, raw_dir, resolve_project, vame_dir  # noqa: E402


def _read_metadata(project_ethoflow: Path, session: str) -> dict:
    """metadata.yaml d'une session, dict vide si absente ou illisible."""
    import yaml
    meta_path = raw_dir(project_ethoflow) / session / "metadata.yaml"
    if not meta_path.exists():
        return {}
    try:
        return yaml.safe_load(meta_path.read_text(encoding="utf-8")) or {}
    except Exception:  # noqa: BLE001 — une metadata cassée ne doit pas
        return {}      # faire tomber le rendu, elle rend juste la session
                       # inclassable (traitée comme sans valeur)


def session_group_values(project_ethoflow: Path, sessions: list[str],
                          column: str) -> dict[str, str | None]:
    """{session: valeur de `column`}, None quand la colonne n'est pas remplie.

    Même source que le reste du pipeline : les colonnes de l'Excel
    recopiées dans `metadata.yaml` par `sync_from_excel.py`.
    """
    out: dict[str, str | None] = {}
    for s in sessions:
        raw = _read_metadata(project_ethoflow, s).get(column)
        out[s] = None if raw is None or str(raw).strip() == "" else str(raw)
    return out


def resolve_pool_group(project_ethoflow: Path,
                        session_slices: dict[str, tuple[int, int]],
                        column: str, target_session: str) -> str:
    """Valeur de `column` sur laquelle restreindre le nuage de fond.

    C'est celle de la session animée : pooler « toutes les MCCiECKO » en
    animant une session MCCf/f donnerait une trajectoire hors de son
    propre nuage.
    """
    vals = session_group_values(project_ethoflow, list(session_slices), column)
    valeur = vals.get(target_session)
    if valeur is not None:
        return valeur

    # Rien à quoi se raccrocher : on distingue les deux causes, qui ne se
    # corrigent pas de la même façon.
    colonnes = sorted({
        c for s in session_slices
        for c in _read_metadata(project_ethoflow, s)
        if c != "source_video"
    })
    if column not in colonnes:
        print(f"❌ --pool-by {column} : colonne absente des metadata.\n"
              f"   Disponibles : {', '.join(colonnes) or '(aucune)'}\n"
              f"   (ajoute-la à ton Excel puis relance sync_from_excel.py)",
              file=sys.stderr)
    else:
        print(f"❌ --pool-by {column} : la session animée {target_session} "
              f"n'a pas de valeur pour cette colonne.\n"
              f"   Impossible de savoir à quel groupe restreindre le nuage.",
              file=sys.stderr)
    sys.exit(1)


def group_background_mask(project_ethoflow: Path,
                           session_slices: dict[str, tuple[int, int]],
                           column: str, value: str, total: int) -> np.ndarray:
    """Masque des frames du pool appartenant aux sessions du groupe.

    Les bornes viennent de `session_slices`, renvoyé par
    `build_pooled_projection` : la projection n'est PAS recalculée, on ne
    fait que cacher les points des autres groupes. C'est ce qui garde les
    axes identiques d'une figure de groupe à l'autre — deux UMAP ajustés
    séparément ne sont pas superposables.
    """
    vals = session_group_values(project_ethoflow, list(session_slices), column)
    retenues = [s for s, v in vals.items() if v == value]
    mask = np.zeros(total, dtype=bool)
    for s in retenues:
        start, end = session_slices[s]
        mask[start:end] = True
    if len(retenues) == 1:
        print(f"    · {column}={value} ne compte qu'1 session ({retenues[0]}) "
              f"— le nuage sera maigre.", file=sys.stderr)
    return mask


def pool_name_fragment(pool_all: bool, column: str | None,
                        value: str | None) -> str:
    """Fragment de nom de fichier décrivant l'étendue du nuage de fond.

    Le groupe doit être dans le nom : sans lui, la figure « MCCiECKO » et
    la figure « MCCf/f » de la même session animée s'écrasent l'une
    l'autre. Les caractères illégaux en nom de fichier sont remplacés —
    `MCCf/f` est une valeur parfaitement normale dans un Excel.
    """
    if value is None:
        return "_pooled" if pool_all else ""
    safe = "".join(c if c.isalnum() or c in "-_." else "-"
                   for c in f"{column}-{value}")
    return f"_pooled-{safe}"


def axis_limits(coords: np.ndarray, margin: float = 0.05
                 ) -> tuple[tuple[float, float], tuple[float, float]]:
    """Bornes (xlim, ylim) englobant `coords`, avec une marge relative.

    Sert à figer le cadrage sur le pool COMPLET quand `--pool-by` ne
    trace qu'un groupe : matplotlib cadre sinon sur les seuls points
    tracés, et deux figures de groupes différents se retrouveraient à des
    zooms différents — ce qui annulerait tout l'intérêt d'un ajustement
    UMAP commun.
    """
    x, y = coords[:, 0], coords[:, 1]
    out = []
    for v in (x, y):
        lo, hi = float(np.min(v)), float(np.max(v))
        span = hi - lo
        pad = span * margin if span > 0 else 1.0  # nuage dégénéré
        out.append((lo - pad, hi + pad))
    return out[0], out[1]


def find_source_video(project_ethoflow: Path, session: str) -> Path | None:
    """Lit source_video depuis la metadata.yaml de la session.

    Retourne None si metadata ou vidéo introuvable — dans ce cas le GIF est
    généré sans le panneau vidéo. Écrit sur stderr la raison exacte du
    fallback pour faciliter le debug (drive débranché vs metadata manquante
    vs champ vide sont trois problèmes très différents à corriger).
    """
    import yaml
    meta_path = raw_dir(project_ethoflow) / session / "metadata.yaml"
    if not meta_path.exists():
        print(f"    · pas de metadata.yaml pour {session} à {meta_path}",
              file=sys.stderr)
        return None
    with open(meta_path) as f:
        meta = yaml.safe_load(f) or {}
    src = meta.get("source_video")
    if not src:
        print(f"    · champ 'source_video' absent de {meta_path}",
              file=sys.stderr)
        return None
    p = Path(src)
    if not p.exists():
        print(f"    · source_video pointe vers {p}, qui n'existe pas.\n"
              f"    · Vérifie que le disque contenant les vidéos est monté,\n"
              f"    · ou passe --source-video <chemin> pour override.",
              file=sys.stderr)
        return None
    return p


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


def discover_all_sessions(vame_project: Path, algo: str) -> list[str]:
    """Liste les sessions VAME qui ont à la fois un latent + un label file."""
    results = vame_project / "results"
    if not results.exists():
        return []
    out = []
    for d in sorted(results.iterdir()):
        if not d.is_dir():
            continue
        s = d.name
        if find_latent_vector(vame_project, s) and find_label_file(vame_project, s, algo):
            out.append(s)
    return out


def build_pooled_projection(
    vame_project: Path,
    algo: str,
    method: str,
    cache_path: Path,
    target_session: str,
    max_frames: int = 300000,
    reproducible: bool = False,
) -> tuple[np.ndarray, np.ndarray, dict[str, tuple[int, int]]]:
    """Projette en 2D toutes les sessions du projet dans un référentiel commun.

    Renvoie :
        - all_coords_2d (N, 2) : projection 2D poolée
        - all_labels (N,)       : motif de chaque frame
        - session_slices        : {session_id: (start_idx, end_idx)} pour
                                  retrouver la tranche d'une session dans
                                  les arrays poolés.
        - target_dependent      : True si le cap a forcé un
                                  sous-échantillonnage, donc si
                                  l'ajustement dépend de `target_session`
                                  (deux sessions cibles → deux manifolds
                                  différents, non superposables).

    Le résultat est cache-persistent : la 1re fois on fit UMAP/PCA sur
    l'ensemble (peut prendre plusieurs minutes) ; les fois suivantes on
    recharge depuis le .npz.

    **Gestion de la taille** : au-delà de `max_frames` frames au total,
    la session `target_session` est gardée en full-res (l'animation en a
    besoin frame-par-frame), tandis que les autres sessions sont
    subsampled uniformément avec un step calculé pour que le pool
    approche `max_frames`. Passer `max_frames=0` désactive le cap.
    """
    if cache_path.exists():
        print(f"  Cache hit : réutilise {cache_path.name}")
        cache = np.load(cache_path, allow_pickle=True)
        # `target_dependent` est apparu après les premiers caches : absent
        # = ancien fichier, on ne peut rien affirmer, on suppose le cas
        # prudent (dépendant) pour ne pas taire un avertissement mérité.
        dep = bool(cache["target_dependent"]) if "target_dependent" in cache \
            else True
        return (cache["coords_2d"], cache["labels"],
                cache["session_slices"].item(), dep)

    sessions = discover_all_sessions(vame_project, algo)
    if not sessions:
        raise RuntimeError(f"Aucune session avec latent+label dans "
                            f"{vame_project / 'results'}")
    print(f"  Poolage sur {len(sessions)} sessions...")

    # Premier passage : lit juste les tailles (mmap_mode évite de tout
    # charger en mémoire à ce stade)
    session_info = []
    for s in sessions:
        lat_f = find_latent_vector(vame_project, s)
        lab_f = find_label_file(vame_project, s, algo)
        lat_shape = np.load(lat_f, mmap_mode="r").shape
        lab_shape = np.load(lab_f, mmap_mode="r").shape
        L = min(lat_shape[0], lab_shape[0])
        session_info.append((s, lat_f, lab_f, L))

    total_raw = sum(L for _, _, _, L in session_info)
    target_L = next(L for s, _, _, L in session_info if s == target_session)
    non_target_total = total_raw - target_L

    # Décide le step d'échantillonnage pour les non-target
    if max_frames > 0 and total_raw > max_frames and non_target_total > 0:
        budget_others = max(1, max_frames - target_L)
        keep_frac = min(1.0, budget_others / non_target_total)
        step_others = max(1, int(round(1 / keep_frac)))
        print(f"  Cap {max_frames:,} → target ({target_session}) en full-res, "
              f"non-target subsampled 1/{step_others}")
    else:
        step_others = 1

    latent_arrays = []
    label_arrays = []
    session_slices: dict[str, tuple[int, int]] = {}
    cursor = 0
    for s, lat_f, lab_f, L in session_info:
        lat = np.load(lat_f)[:L]
        lab = np.load(lab_f)[:L].astype(int)
        if s != target_session and step_others > 1:
            lat = lat[::step_others]
            lab = lab[::step_others]
        latent_arrays.append(lat)
        label_arrays.append(lab)
        session_slices[s] = (cursor, cursor + len(lat))
        cursor += len(lat)
        print(f"    · {s}: {L:,} → {len(lat):,} frames")

    all_latents = np.concatenate(latent_arrays, axis=0)
    all_labels = np.concatenate(label_arrays, axis=0)
    print(f"  Total poolé : {total_raw:,} → {len(all_latents):,} frames "
          f"× {all_latents.shape[1]} dims")

    all_coords_2d = project_to_2d(all_latents, method, reproducible=reproducible)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        coords_2d=all_coords_2d,
        labels=all_labels,
        session_slices=np.array(session_slices, dtype=object),
        target_dependent=step_others > 1,
    )
    print(f"  Cache écrit : {cache_path}")
    return all_coords_2d, all_labels, session_slices, step_others > 1


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


def project_to_2d(latents: np.ndarray, method: str,
                   reproducible: bool = False) -> np.ndarray:
    """Projette (N, D) → (N, 2) via UMAP ou PCA.

    Par défaut UMAP tourne en parallèle (n_jobs=-1, sans seed). Passer
    `reproducible=True` fixe random_state=42 mais retombe sur un seul
    thread — sur >200k points, c'est de l'ordre de 5-10x plus lent.
    Pour de la viz, la reproductibilité au pixel près n'est pas critique :
    la topologie des clusters est stable entre runs.
    """
    if method == "umap":
        try:
            import umap  # type: ignore
            print(f"  UMAP fitting sur {len(latents):,} points"
                  f" ({'reproducible/single-thread' if reproducible else 'parallel'})...")
            kwargs = dict(n_components=2, n_neighbors=30, min_dist=0.3,
                          low_memory=True)
            if reproducible:
                kwargs["random_state"] = 42
            # Sinon on ne fixe PAS n_jobs (par défaut -1 = tous les cores)
            reducer = umap.UMAP(**kwargs)
            return reducer.fit_transform(latents)
        except ImportError:
            print("  umap-learn non installé, fallback sur PCA",
                  file=sys.stderr)
            method = "pca"
    # PCA
    from sklearn.decomposition import PCA
    print(f"  PCA fitting sur {len(latents):,} points...")
    return PCA(n_components=2, random_state=42).fit_transform(latents)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    add_project_dir_arg(parser, required=True)
    parser.add_argument("--session", default=None,
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
    parser.add_argument("--with-video", action="store_true",
                        help="Ajoute un panneau vidéo à côté du manifold "
                             "(la vraie souris à gauche, sa trajectoire "
                             "dans l'espace latent à droite). Nécessite "
                             "OpenCV et un source_video valide dans la "
                             "metadata. Ralentit un peu le rendu.")
    parser.add_argument("--video-max-width", type=int, default=480,
                        help="Largeur max du panneau vidéo en pixels "
                             "(l'image est downscalée à cette taille pour "
                             "garder le GIF léger). Défaut 480.")
    parser.add_argument("--source-video", type=Path, default=None,
                        help="Chemin explicite vers la vidéo source à "
                             "afficher dans le panneau. Ignore ce qui est "
                             "dans metadata.yaml (utile quand le drive de "
                             "recording n'est plus mappé à la même lettre).")
    parser.add_argument("--pool-all-sessions", action="store_true",
                        help="Calcule le manifold sur TOUTES les sessions "
                             "du projet (référentiel commun) au lieu de la "
                             "seule session --session. Le nuage de fond "
                             "représente alors ton dataset entier ; "
                             "l'animation reste sur la trajectoire de la "
                             "session ciblée. Plus long (UMAP sur N× plus "
                             "de points) mais beaucoup plus parlant pour "
                             "une figure de publi.")
    parser.add_argument("--pool-by", default=None, metavar="COLONNE",
                        help="Restreint le nuage de fond aux sessions qui "
                             "partagent la valeur de COLONNE (une colonne de "
                             "ton Excel, ex : condition) avec la session "
                             "--session. Implique --pool-all-sessions. "
                             "L'ajustement UMAP reste commun à TOUT le "
                             "projet — donc le cache poolé est réutilisé tel "
                             "quel et deux figures de groupes différents ont "
                             "exactement les mêmes axes, donc se comparent.")
    parser.add_argument("--pool-cache", type=Path, default=None,
                        help="Fichier .npz où mettre en cache la projection "
                             "poolée. Défaut auto : "
                             "<vame>/analysis/behavior_structure/"
                             "pooled_<projection>.npz. Supprime-le pour "
                             "recomputer.")
    parser.add_argument("--background-max-points", type=int, default=80000,
                        help="Nombre max de points affichés en background. "
                             "En mode --pool-all-sessions on peut avoir "
                             "1M+ points, ce qui rend le rendu prohibitif. "
                             "On subsample stratifié par motif pour "
                             "préserver la structure. Défaut 80000.")
    parser.add_argument("--pool-max-frames", type=int, default=300000,
                        help="Cap sur le nombre total de frames envoyées à "
                             "UMAP en mode --pool-all-sessions. La session "
                             "--session est TOUJOURS gardée en full-res (pour "
                             "l'animation) ; les autres sont subsampled "
                             "uniformément si nécessaire. Défaut 300000 "
                             "(~5-10 min UMAP parallèle). Mets à 0 pour "
                             "désactiver le cap.")
    parser.add_argument("--umap-reproducible", action="store_true",
                        help="Fixe random_state=42 sur UMAP → même manifold "
                             "d'un run à l'autre, MAIS force single-thread "
                             "(5-10x plus lent). Utile pour un run final "
                             "de publication ; à éviter en exploration.")
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
    args.session = prompt_session(
        project, args.session,
        no_prompt=getattr(args, "no_prompt", False),
        title="Quelle session animer ?",
    )
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

    # Projection 2D — deux modes :
    # - single-session (défaut) : UMAP/PCA sur les frames de --session seul
    # - pool-all-sessions : UMAP/PCA sur tout le projet, extrait la tranche
    #   de la session ciblée pour l'animation. Le background scatter reste
    #   sur TOUT le pool (le contexte global du dataset).
    if args.pool_by and not args.pool_all_sessions:
        # L'ajustement commun EST ce qui rend les figures de groupes
        # comparables : demander l'un sans l'autre n'a pas de sens.
        print(f"ℹ  --pool-by {args.pool_by} implique --pool-all-sessions.")
        args.pool_all_sessions = True

    pool_group = None
    if args.pool_all_sessions:
        # Cache : sensible à target_session + max_frames car ils changent
        # les données envoyées à UMAP (target en full-res, autres subsampled)
        cap_tag = "nocap" if args.pool_max_frames == 0 else f"max{args.pool_max_frames}"
        cache_path = args.pool_cache or (
            vame_proj / "analysis" / "behavior_structure"
            / f"pooled_{args.projection}_{cap_tag}_target-{args.session}.npz"
        )
        pool_coords, pool_labels, pool_slices, fit_depends_on_target = build_pooled_projection(
            vame_proj, args.algo, args.projection, cache_path,
            target_session=args.session,
            max_frames=args.pool_max_frames,
            reproducible=args.umap_reproducible,
        )
        if args.session not in pool_slices:
            print(f"❌ {args.session} n'est pas dans le pool "
                  f"(sessions présentes : {list(pool_slices)})",
                  file=sys.stderr)
            sys.exit(1)
        s_start, s_end = pool_slices[args.session]
        # Coords + labels utilisés pour l'ANIMATION (marker + trail)
        coords_2d = pool_coords[s_start:s_end]
        labels = pool_labels[s_start:s_end]
        L = len(coords_2d)
        # `pool_coords` + `pool_labels` restent dispo pour le background
        bg_coords = pool_coords
        bg_labels = pool_labels
        if args.pool_by:
            pool_group = resolve_pool_group(
                project, pool_slices, args.pool_by, args.session)
            mask = group_background_mask(
                project, pool_slices, args.pool_by, pool_group,
                total=len(pool_coords))
            bg_coords = pool_coords[mask]
            bg_labels = pool_labels[mask]
            n_grp = sum(1 for s in pool_slices
                        if mask[pool_slices[s][0]:pool_slices[s][1]].any())
            print(f"  Nuage restreint à {args.pool_by}={pool_group} : "
                  f"{n_grp} sessions, {len(bg_coords):,} points "
                  f"(projection inchangée, axes communs)")
            if fit_depends_on_target:
                # Le cap a sous-échantillonné : l'ajustement a vu la session
                # animée en full-res et les autres 1/N. Une figure sur une
                # AUTRE session animée part donc d'un ajustement différent —
                # les deux ne se superposent plus, ce qui est précisément la
                # promesse de --pool-by.
                print(f"  ⚠  Le cap --pool-max-frames a sous-échantillonné le "
                      f"pool, donc cet ajustement dépend de la session animée "
                      f"({args.session}).\n"
                      f"     Une figure sur une autre session ne sera PAS "
                      f"superposable à celle-ci.\n"
                      f"     Pour une série de figures comparables : "
                      f"--pool-max-frames 0 --umap-reproducible "
                      f"(plus long, mais l'ajustement devient identique "
                      f"quelle que soit la session animée).",
                      file=sys.stderr)
    else:
        coords_2d = project_to_2d(latents, args.projection)
        bg_coords = coords_2d  # single-session : background = anim
        bg_labels = labels

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

    # ----- (Optionnel) Ouvre la vidéo source pour panneau side-by-side -----
    video_cap = None
    video_size = None  # (H, W) après resize
    if args.with_video:
        try:
            import cv2  # noqa: WPS433
        except ImportError:
            print("⚠  OpenCV absent, --with-video ignoré", file=sys.stderr)
        else:
            # Priorité : --source-video CLI, sinon metadata.yaml
            if args.source_video is not None:
                if args.source_video.exists():
                    src_video = args.source_video
                    print(f"    Override CLI : source_video = {src_video}")
                else:
                    print(f"⚠  --source-video={args.source_video} n'existe pas",
                          file=sys.stderr)
                    src_video = None
            else:
                src_video = find_source_video(project, args.session)
            if src_video is None:
                print(f"⚠  source_video introuvable pour {args.session}, "
                      f"--with-video ignoré (voir raison ci-dessus)",
                      file=sys.stderr)
            else:
                video_cap = cv2.VideoCapture(str(src_video))
                if not video_cap.isOpened():
                    print(f"⚠  Impossible d'ouvrir {src_video}, "
                          f"--with-video ignoré", file=sys.stderr)
                    video_cap = None
                else:
                    src_w = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    src_h = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    if src_w > args.video_max_width:
                        scale = args.video_max_width / src_w
                        video_size = (int(src_h * scale), args.video_max_width)
                    else:
                        video_size = (src_h, src_w)
                    print(f"Vidéo     : {src_video.name}  {src_w}×{src_h} "
                          f"→ {video_size[1]}×{video_size[0]}")

    # ----- Setup figure -----
    if video_cap is not None:
        # Layout : vidéo à gauche (carrée-ish), manifold à droite (carré)
        fig, (ax_video, ax) = plt.subplots(
            1, 2, figsize=(14, 7),
            gridspec_kw={"width_ratios": [1.0, 1.1]},
        )
        ax_video.set_facecolor("black")
        ax_video.set_xticks([]); ax_video.set_yticks([])
        ax_video.set_title(f"Vidéo — {args.session}", fontsize=11)
        # Placeholder gris tant qu'on n'a pas la 1re frame
        video_im = ax_video.imshow(
            np.zeros((*video_size, 3), dtype=np.uint8)
        )
        # Petit badge motif superposé sur la vidéo
        video_osd = ax_video.text(
            0.02, 0.98, "", transform=ax_video.transAxes,
            va="top", ha="left", fontsize=11, fontweight="bold",
            color="white",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="black",
                      edgecolor="white", alpha=0.75),
        )
    else:
        fig, ax = plt.subplots(figsize=(9, 8))
        ax_video = None
        video_im = None
        video_osd = None
    ax.set_facecolor("white")

    # Background : tous les points (color par motif). En mode pooled c'est
    # le dataset entier ; sinon la seule session active. On cape le nombre
    # de points affichés (subsample stratifié par motif) car sinon le
    # rendu matplotlib devient prohibitif à partir de ~500k points.
    n_bg_total = len(bg_coords)
    if n_bg_total > args.background_max_points:
        rng = np.random.default_rng(42)
        keep_frac = args.background_max_points / n_bg_total
        keep_mask = np.zeros(n_bg_total, dtype=bool)
        for m in np.unique(bg_labels):
            m_idx = np.where(bg_labels == m)[0]
            n_keep_m = max(1, int(round(len(m_idx) * keep_frac)))
            chosen = rng.choice(m_idx, size=min(n_keep_m, len(m_idx)),
                                 replace=False)
            keep_mask[chosen] = True
        bg_coords_show = bg_coords[keep_mask]
        bg_labels_show = bg_labels[keep_mask]
        print(f"  Background : {n_bg_total:,} → {len(bg_coords_show):,} points "
              f"(subsample stratifié pour rendu rapide)")
    else:
        bg_coords_show = bg_coords
        bg_labels_show = bg_labels

    unique_motifs = np.unique(bg_labels_show)
    for m in unique_motifs:
        mask = bg_labels_show == m
        color = TAB20[m % len(TAB20)]
        ax.scatter(
            bg_coords_show[mask, 0], bg_coords_show[mask, 1],
            s=8, c=[color], alpha=args.background_alpha,
            edgecolors="none",
            rasterized=True,  # transforme en bitmap, ~10x plus rapide à re-render
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

    if pool_group is not None:
        # Cadrage figé sur le pool entier : sans ça, chaque groupe serait
        # zoomé sur son propre nuage et les figures ne se superposeraient
        # plus, alors même que la projection est commune.
        (x0, x1), (y0, y1) = axis_limits(pool_coords)
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)

    ax.set_xlabel(f"{args.projection.upper()}-1")
    ax.set_ylabel(f"{args.projection.upper()}-2")
    if pool_group is not None:
        title_scope = f"{args.pool_by} = {pool_group}"
    elif args.pool_all_sessions:
        title_scope = "pooled dataset"
    else:
        title_scope = args.session
    ax.set_title(f"Behavior manifold — {title_scope}\n"
                 f"animated trajectory : {args.session}"
                 if args.pool_all_sessions
                 else f"Behavior manifold — {args.session}")
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

    # ----- Pré-extraction des frames vidéo (bien plus rapide que seek par frame) -----
    # cv2.VideoCapture.set(POS_FRAMES) est très lent sur MP4/h264 (décode
    # depuis la dernière keyframe à chaque appel). On lit séquentiellement
    # une seule fois et on garde en mémoire les frames dont on aura besoin.
    video_frames = None
    if video_cap is not None:
        import cv2  # noqa: WPS433
        print("Pré-lecture des frames vidéo (évite les seeks lents)...")
        needed = set(anim_indices.tolist())
        max_idx = max(needed)
        video_frames = {}
        # Seek une seule fois au début du range utile, puis read séquentiel
        video_cap.set(cv2.CAP_PROP_POS_FRAMES, int(min(needed)))
        cur = int(min(needed))
        while cur <= max_idx:
            ok, frame = video_cap.read()
            if not ok:
                break
            if cur in needed:
                if (frame.shape[0], frame.shape[1]) != video_size:
                    frame = cv2.resize(
                        frame, (video_size[1], video_size[0]),
                        interpolation=cv2.INTER_AREA,
                    )
                video_frames[cur] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            cur += 1
        print(f"  → {len(video_frames)} frames en cache mémoire "
              f"(~{sum(f.nbytes for f in video_frames.values()) / 1e6:.0f} MB)")

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

        # Panneau vidéo : lecture depuis le cache mémoire
        if video_frames is not None and idx in video_frames:
            video_im.set_data(video_frames[idx])
            video_osd.set_text(f"[{current_motif}] {name}")

        return trail_line, marker, osd, time_text

    print("Rendu de l'animation...")
    anim = FuncAnimation(fig, update, frames=n_anim, interval=1000 / fps_out,
                        blit=False)

    out_dir = vame_proj / "analysis" / "behavior_structure"
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.start:.0f}s_{args.duration:.0f}s" if args.duration else "_full"
    side = "_sidebyside" if video_cap is not None else ""
    pool = pool_name_fragment(args.pool_all_sessions, args.pool_by, pool_group)
    stem = f"{args.session}_manifold_{args.projection}{pool}{side}{suffix}"

    # Callback de progression : matplotlib appelle ce truc à chaque frame
    # sauvée par le writer → on affiche un pourcentage rassurant.
    def _progress(current: int, total: int) -> None:
        pct = int(100 * (current + 1) / total)
        # Écrit sur la même ligne (ne pas polluer le terminal)
        print(f"\r  Encoding frame {current + 1}/{total}  ({pct}%)",
              end="", flush=True)

    if args.output_format == "gif":
        out_path = out_dir / f"{stem}.gif"
        writer = PillowWriter(fps=fps_out)
        anim.save(str(out_path), writer=writer, dpi=80,
                  progress_callback=_progress)
        print()  # newline après la barre
    else:
        out_path = out_dir / f"{stem}.mp4"
        try:
            writer = FFMpegWriter(fps=fps_out, bitrate=2000)
            anim.save(str(out_path), writer=writer, dpi=100,
                      progress_callback=_progress)
            print()
        except Exception as e:
            print(f"\n⚠  FFMpegWriter échoué ({e}), fallback en gif+ffmpeg",
                  file=sys.stderr)
            tmp_gif = out_dir / f"{stem}_tmp.gif"
            anim.save(str(tmp_gif), writer=PillowWriter(fps=fps_out), dpi=80,
                      progress_callback=_progress)
            print()
            subprocess.run(
                ["ffmpeg", "-y", "-i", str(tmp_gif), "-c:v", "libx264",
                 "-pix_fmt", "yuv420p", str(out_path)],
                check=True,
            )
            tmp_gif.unlink(missing_ok=True)

    plt.close(fig)
    if video_cap is not None:
        video_cap.release()
    size_mb = out_path.stat().st_size / 1e6
    print(f"\n✅ {out_path}  ({size_mb:.1f} MB)")
    if args.output_format == "gif" and size_mb > 30:
        print(f"   ⚠  >30MB : considère --duration plus court, "
              f"--fps-output plus bas, ou --output-format mp4")
    if args.with_video and args.output_format == "gif":
        print(f"   ℹ Avec --with-video le GIF grossit vite — préfère "
              f"--output-format mp4 pour les extraits > 30 s.")


if __name__ == "__main__":
    main()
