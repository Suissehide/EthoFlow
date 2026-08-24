"""Construction des commandes du pipeline — sans exécution.

Chaque fonction publique retourne un `Command` décrivant quoi lancer et
dans quel env conda. C'est `lib/runner.py` qui exécute. Séparer les deux
rend la construction testable sans conda, sans GPU et sans données.

Trois règles valent pour toute commande (voir la spec §5.1) :
  1. `--project-dir` toujours passé, sinon le script demande le projet à
     l'invite et le subprocess se fige.
  2. `--no-prompt` toujours passé, pour un échec franc au lieu d'une
     attente silencieuse.
  3. Tout paramètre que le script demanderait est fourni explicitement.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from lib.project import SCRIPTS_DIR

# Mapping script -> env conda. Voir spec §5.2. Se tromper d'env produit un
# ImportError après plusieurs minutes d'attente.
SCRIPT_ENVS: dict[str, str] = {
    # env ethoflow : pandas / yaml / openpyxl / cv2
    "create_project.py": "ethoflow",
    "excel_templates.py": "ethoflow",
    "sync_from_excel.py": "ethoflow",
    "crop_arenes.py": "ethoflow",
    "assign_arenas.py": "ethoflow",
    "inspect_session.py": "ethoflow",
    "diagnose_dlc_model.py": "ethoflow",
    "motif_gif.py": "ethoflow",
    "post_process_cropped.py": "ethoflow",
    "filter_keypoints.py": "ethoflow",
    "fill_nan_h5.py": "ethoflow",
    "trim_empty_arena.py": "ethoflow",
    "rekey_h5.py": "ethoflow",
    # env dlc : importe deeplabcut
    "run_dlc_inference.py": "dlc",
    "prepare_vame_input_custom.py": "dlc",   # dlc.filterpredictions
    # env vame : vame, matplotlib, scipy, umap, sklearn
    "run_vame.py": "vame",
    "analyze_vame.py": "vame",               # matplotlib + scipy
    "behavior_structure_gif.py": "vame",     # umap + sklearn
    "community_dendrogram.py": "vame",       # scipy
    "inspect_vame_project.py": "vame",
    "reencode_vame_videos.py": "vame",
}


@dataclass(frozen=True)
class Command:
    env: str
    script: str
    args: list[str] = field(default_factory=list)
    label: str = ""


def to_argv(cmd: Command) -> list[str]:
    """argv complet pour `subprocess.Popen`.

    Pas de `--no-capture-output` : ce flag renvoie la sortie au terminal
    au lieu du pipe, et c'est précisément le bug de l'ancienne version.
    """
    env = SCRIPT_ENVS[cmd.script]   # KeyError volontaire si script inconnu
    return [
        "conda", "run", "-n", env,
        "python", str(SCRIPTS_DIR / cmd.script),
        *cmd.args,
    ]


def _base(project: Path) -> list[str]:
    return ["--project-dir", str(Path(project).resolve()), "--no-prompt"]


def _cmd(script: str, args: list[str], label: str) -> Command:
    return Command(env=SCRIPT_ENVS[script], script=script, args=args, label=label)


# ============================================================
# Constructeurs
# ============================================================

def create_project(project: Path, *, kind: str,
                   dlc_config: str | None = None,
                   force: bool = False) -> Command:
    args = ["--project-dir", str(Path(project).resolve()), "--kind", kind]
    if dlc_config:
        args += ["--dlc-config", str(dlc_config)]
    if force:
        args.append("--force")
    args.append("--no-prompt")
    return _cmd("create_project.py", args, f"Créer le projet {Path(project).name}")


def sync_from_excel(project: Path, *, videos_dir: str | Path,
                    excel: str | Path | None = None,
                    video_ext: str = "mp4",
                    overwrite: bool = False,
                    dry_run: bool = False) -> Command:
    args = _base(project) + ["--videos-dir", str(videos_dir), "--video-ext", video_ext]
    if excel:
        args += ["--excel", str(excel)]
    if overwrite:
        args.append("--overwrite")
    if dry_run:
        args.append("--dry-run")
    return _cmd("sync_from_excel.py", args,
                "Aperçu du sync" if dry_run else "Sync depuis Excel")


def crop_arenes(project: Path, *, sessions: list[str] | None = None,
                all_sessions: bool = False, all_new: bool = False) -> Command:
    args = _base(project)
    if all_sessions:
        args.append("--all")
    elif all_new:
        args.append("--all-new")
    else:
        args += list(sessions or [])
    return _cmd("crop_arenes.py", args, "Crop des arènes")


def run_dlc_inference(project: Path, *, mode: str,
                      sessions: list[str] | None = None,
                      all_sessions: bool = False,
                      skip_existing: bool = True,
                      video_adapt: bool = False,
                      video_adapt_batch_size: int = 2) -> Command:
    args = _base(project)
    if all_sessions:
        args.append("--all")
    else:
        args += list(sessions or [])
    args += ["--mode", mode]
    if skip_existing:
        args.append("--skip-existing")
    if video_adapt:
        args += ["--video-adapt",
                 "--video-adapt-batch-size", str(video_adapt_batch_size)]
    return _cmd("run_dlc_inference.py", args, f"Inférence DLC ({mode})")


def diagnose_dlc_model(project: Path, *, model_dir: str | Path | None = None,
                       fix: bool = False) -> Command:
    args = _base(project)
    if model_dir:
        args += ["--model-dir", str(model_dir)]
    if fix:
        args.append("--fix")
    return _cmd("diagnose_dlc_model.py", args,
                "Réparer le modèle DLC" if fix else "Diagnostiquer le modèle DLC")


def prepare_vame_input(project: Path, *, likelihood_threshold: float,
                       max_speed: float,
                       px_per_cm: float | None = None,
                       sessions: list[str] | None = None,
                       sticky_detection: bool = True,
                       qc_plot: bool = True,
                       qc_bodypart: str = "tail_base",
                       interp_limit: int = 25,
                       window_length: int = 5,
                       skip_existing: bool = False) -> Command:
    args = _base(project) + list(sessions or [])
    args += ["--likelihood-threshold", str(likelihood_threshold),
             "--max-speed", str(max_speed),
             "--interp-limit", str(interp_limit),
             "--window-length", str(window_length),
             "--qc-bodypart", qc_bodypart]
    if px_per_cm is not None:
        args += ["--px-per-cm", str(px_per_cm)]
    if not sticky_detection:
        args.append("--no-sticky-detection")
    if not qc_plot:
        args.append("--no-qc-plot")
    if skip_existing:
        args.append("--skip-existing")
    return _cmd("prepare_vame_input_custom.py", args, "Nettoyage des poses")


def parse_prepare_vame_input_args(argv: list[str]) -> dict | None:
    """Reconstruit les kwargs de `prepare_vame_input` depuis l'`argv` d'un job déjà lancé.

    Sert au bouton « régénérer sur un autre keypoint » de la galerie QC
    (page Nettoyage, Task 21) : les graphes ne sont comparables entre eux
    que si les seuils et les sessions sont *identiques* à ceux du run
    d'origine — seul `--qc-bodypart` doit changer. `argv` (persisté par
    `lib.runner` dans le JSON du job, sur disque) est la seule source qui
    survit à une navigation, un rafraîchissement de page ou un redémarrage
    de l'app ; les widgets du formulaire (`session_state`), eux, ne
    survivent à rien de tout ça et peuvent avoir été modifiés depuis.

    Retourne `None` si `argv` ne vient pas d'un run de
    `prepare_vame_input_custom.py`, s'il lui manque un des deux réglages
    que le script exige toujours (`--likelihood-threshold`/`--max-speed`,
    jamais absents d'une commande construite par `prepare_vame_input`), ou
    s'il contient un flag `--xxx` non reconnu (ruling R21.1 : un format
    futur du script, potentiellement avec une valeur associée qu'on ne
    saurait pas distinguer d'un `session_id` positionnel) — dans tous les
    cas, refuser plutôt que de deviner une valeur par défaut ou de risquer
    de faire passer une valeur de flag pour une session, ce qui rendrait
    les graphes incomparables ou ciblerait une session inexistante.
    """
    args = list(argv)
    try:
        i_script = next(
            i for i, a in enumerate(args) if a.endswith("prepare_vame_input_custom.py")
        )
    except StopIteration:
        return None

    rest = args[i_script + 1:]
    kwargs: dict = {}
    sessions: list[str] = []
    i = 0
    while i < len(rest):
        tok = rest[i]
        if tok in ("--project-dir", "--likelihood-threshold", "--max-speed",
                   "--interp-limit", "--window-length", "--qc-bodypart",
                   "--px-per-cm"):
            valeur = rest[i + 1] if i + 1 < len(rest) else None
            if tok == "--likelihood-threshold" and valeur is not None:
                kwargs["likelihood_threshold"] = float(valeur)
            elif tok == "--max-speed" and valeur is not None:
                kwargs["max_speed"] = float(valeur)
            elif tok == "--interp-limit" and valeur is not None:
                kwargs["interp_limit"] = int(valeur)
            elif tok == "--window-length" and valeur is not None:
                kwargs["window_length"] = int(valeur)
            elif tok == "--px-per-cm" and valeur is not None:
                kwargs["px_per_cm"] = float(valeur)
            # --project-dir et --qc-bodypart : lus mais ignorés (le premier
            # est fourni par l'appelant, le second est justement ce qu'on
            # s'apprête à changer).
            i += 2
            continue
        if tok == "--no-prompt":
            i += 1
            continue
        if tok == "--no-sticky-detection":
            kwargs["sticky_detection"] = False
            i += 1
            continue
        if tok == "--no-qc-plot":
            kwargs["qc_plot"] = False
            i += 1
            continue
        if tok == "--skip-existing":
            kwargs["skip_existing"] = True
            i += 1
            continue
        if tok.startswith("--"):
            # Ruling R21.1 : un flag inconnu (format futur du script)
            # invalide tout le parse plutôt que d'être juste sauté — s'il
            # prend une valeur, cette valeur retomberait sinon dans la
            # branche positionnelle et polluerait silencieusement
            # `sessions` (ex. « --fps 30 » ferait apparaître « 30 » comme
            # un session_id). Refuser est le choix sûr : l'appelant sait
            # déjà désactiver le bouton avec une explication quand ceci
            # renvoie `None`, une hypothèse fausse sur la présence ou non
            # d'une valeur associée ne l'est pas.
            return None
        sessions.append(tok)
        i += 1

    if "likelihood_threshold" not in kwargs or "max_speed" not in kwargs:
        return None
    if sessions:
        kwargs["sessions"] = sessions
    return kwargs


def assign_arenas(project: Path, *, sessions: list[str] | None = None,
                  all_sessions: bool = False, all_new: bool = False,
                  likelihood_threshold: float = 0.6,
                  interp_limit: int = 25, clean: bool = True) -> Command:
    args = _base(project)
    if all_sessions:
        args.append("--all")
    elif all_new:
        args.append("--all-new")
    else:
        args += list(sessions or [])
    args += ["--likelihood-threshold", str(likelihood_threshold),
             "--interp-limit", str(interp_limit)]
    if not clean:
        args.append("--no-clean")
    return _cmd("assign_arenas.py", args, "Split par arène")


def inspect_session(project: Path, *, sessions: list[str] | None = None,
                    all_sessions: bool = False,
                    input_dir: str | Path | None = None,
                    fps: float | None = None) -> Command:
    args = _base(project)
    if all_sessions or not sessions:
        args.append("--all")
    else:
        args += list(sessions)
    if input_dir:
        args += ["--input-dir", str(input_dir)]
    if fps is not None:
        args += ["--fps", str(fps)]
    return _cmd("inspect_session.py", args, "Inspection qualité")


def filter_keypoints(*, input_dir: str | Path, output_dir: str | Path,
                     keep: list[str] | None = None,
                     drop: list[str] | None = None,
                     min_validity: float | None = None,
                     dry_run: bool = False) -> Command:
    """Ne prend pas de `project` : ce script n'est pas projet-aware, les
    chemins d'entrée/sortie sont à fournir explicitement (voir spec §5.2)."""
    args = ["--input-dir", str(input_dir), "--output-dir", str(output_dir)]
    if keep:
        args += ["--keep", *keep]
    if drop:
        args += ["--drop", *drop]
    if min_validity is not None:
        args += ["--min-validity", str(min_validity)]
    if dry_run:
        args.append("--dry-run")
    return _cmd("filter_keypoints.py", args, "Filtrer les keypoints")


def fill_nan_h5(*, root: str | Path, output_dir: str | Path | None = None,
                dry_run: bool = False) -> Command:
    """Pas projet-aware non plus — `root` est un chemin explicite."""
    args = ["--root", str(root)]
    if output_dir:
        args += ["--output-dir", str(output_dir)]
    if dry_run:
        args.append("--dry-run")
    return _cmd("fill_nan_h5.py", args, "Combler les NaN résiduels")


def trim_empty_arena(*, validity_csv: str | Path, h5_input: str | Path,
                     h5_output: str | Path, video_input: str | Path,
                     video_output: str | Path, dry_run: bool = False) -> Command:
    """Pas projet-aware — les cinq chemins sont à fournir explicitement."""
    args = ["--validity-csv", str(validity_csv),
             "--h5-input", str(h5_input), "--h5-output", str(h5_output),
             "--video-input", str(video_input), "--video-output", str(video_output)]
    if dry_run:
        args.append("--dry-run")
    return _cmd("trim_empty_arena.py", args, "Tronquer les frames d'arène vide")


def vame_stage(project: Path, stage: str, *,
               n_clusters: int | None = None,
               regen_labels: bool = False,
               extra: list[str] | None = None) -> Command:
    """`--project-dir` doit précéder la sous-commande (contrainte argparse)."""
    args = _base(project) + [stage]
    if stage == "segment" and n_clusters is not None:
        args += ["--n-clusters", str(n_clusters)]
    if stage in ("motif-videos", "motif-labels") and regen_labels:
        args.append("--regen-labels")
    args += list(extra or [])
    return _cmd("run_vame.py", args, f"VAME {stage}")


def analyze_vame(project: Path, *, algo: str = "hmm",
                 n_clusters: int | None = None,
                 labels: str | Path | None = None,
                 group_by: list[str] | None = None,
                 cross: list[tuple[str, str]] | None = None,
                 extended: bool = False,
                 extended_by: str | None = None,
                 mask_empty: bool = False,
                 validity_source: str | Path | None = None,
                 min_edge_frames: int = 25,
                 fps: float = 30.0,
                 list_columns: bool = False) -> Command:
    args = _base(project)
    if list_columns:
        # Sort la liste des axes et rend la main : tout autre flag serait
        # ignoré, autant ne pas les construire.
        return _cmd("analyze_vame.py", args + ["--list-columns"],
                    "Axes de comparaison disponibles")
    args += ["--algo", algo, "--min-edge-frames", str(min_edge_frames),
             "--fps", str(fps)]
    if n_clusters is not None:
        args += ["--n-clusters", str(n_clusters)]
    if labels:
        args += ["--labels", str(labels)]
    if group_by:
        args += ["--group-by", *group_by]
    for pair in (cross or []):
        args += ["--cross", pair[0], pair[1]]
    if extended:
        args.append("--extended")
        if extended_by:
            args += ["--extended-by", extended_by]
    if validity_source:
        args += ["--validity-source", str(validity_source)]
    if mask_empty:
        args.append("--mask-empty")
    return _cmd("analyze_vame.py", args, "Analyses statistiques")


def motif_gif(project: Path, *, session: str, algo: str = "hmm",
              start: float = 0.0, duration: float | None = None,
              output_format: str = "mp4",
              labels: str | Path | None = None) -> Command:
    args = _base(project) + ["--session", session, "--algo", algo,
                             "--start", str(start),
                             "--output-format", output_format]
    if duration is not None:
        args += ["--duration", str(duration)]
    if labels:
        args += ["--labels", str(labels)]
    return _cmd("motif_gif.py", args, f"Bande de motifs — {session}")


def behavior_structure_gif(project: Path, *, session: str, algo: str = "hmm",
                           projection: str = "umap",
                           start: float = 0.0,
                           duration: float | None = None,
                           output_format: str = "gif",
                           with_video: bool = False,
                           pool_all_sessions: bool = False,
                           labels: str | Path | None = None) -> Command:
    args = _base(project) + ["--session", session, "--algo", algo,
                             "--projection", projection,
                             "--start", str(start),
                             "--output-format", output_format]
    if duration is not None:
        args += ["--duration", str(duration)]
    if with_video:
        args.append("--with-video")
    if pool_all_sessions:
        args.append("--pool-all-sessions")
    if labels:
        args += ["--labels", str(labels)]
    return _cmd("behavior_structure_gif.py", args, f"Manifold — {session}")


def community_dendrogram(project: Path, *, algo: str = "hmm",
                         group: str | None = None,
                         linkage: str = "ward",
                         labels: str | Path | None = None) -> Command:
    args = _base(project) + ["--algo", algo, "--linkage", linkage]
    if group:
        args += ["--group", group]
    if labels:
        args += ["--labels", str(labels)]
    return _cmd("community_dendrogram.py", args, "Dendrogramme des communautés")
