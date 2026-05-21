"""
Inférence DeepLabCut sur la vidéo source d'une (ou plusieurs) session(s).

Deux modes :

1. **SuperAnimal multi-animal** (défaut) — lit `source_video` du metadata,
   lance `video_inference_superanimal` directement sur la vidéo entière
   (pas de crop). Sortie : un seul .h5 multi-animal contenant les
   trajectoires des 4 souris.

2. **Modèle custom** — si `dlc_project_config` est défini dans
   `configs/pipeline_config.yaml`, lance `analyze_videos` avec ce modèle.

Usage:
    python scripts/run_dlc_inference.py <session_id>
    python scripts/run_dlc_inference.py <s1> <s2> <s3>          # plusieurs
    python scripts/run_dlc_inference.py --all                   # toutes les sessions non traitées
    python scripts/run_dlc_inference.py --all --skip-existing   # idem, par sécurité
    python scripts/run_dlc_inference.py <session_id> --mode custom
    python scripts/run_dlc_inference.py <session_id> --video-adapt

Pré-requis :
    - Activer l'env conda 'dlc' avant de lancer
    - DLC 3.x + PyTorch
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
CROPPED_DIR = ROOT / "data" / "cropped"
DLC_OUTPUT_DIR = ROOT / "data" / "dlc-output"
VAME_INPUT_DIR = ROOT / "data" / "vame-input"
PIPELINE_CONFIG = ROOT / "configs" / "pipeline_config.yaml"


def load_session_metadata(session_id: str) -> dict:
    metadata_path = RAW_DIR / session_id / "metadata.yaml"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata absent : {metadata_path}")
    with open(metadata_path) as f:
        return yaml.safe_load(f)


def get_source_video(metadata: dict) -> Path:
    source = metadata.get("source_video")
    if not source:
        raise ValueError("Pas de `source_video` dans le metadata.yaml")
    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"Vidéo source introuvable : {path}")
    return path


def is_processed(session_id: str, mode: str = "superanimal") -> bool:
    """Vrai si une sortie DLC existe déjà pour cette session, dans le mode donné.

    - superanimal (multi-animal) : .h5 directement dans dlc-output/<session>/
    - single-animal               : .h5 finaux dans vame-input/<session>/, ou
                                    .h5 "adapté" (hors before_adapt) dans
                                    dlc-output/<session>/cropped-raw/
    - custom                      : .h5 dans dlc-output/<session>/
    """
    if mode == "single-animal":
        # Sortie finale présente dans vame-input/ → réellement traité.
        vame_session = VAME_INPUT_DIR / session_id
        if vame_session.exists() and any(vame_session.glob(f"{session_id}_A*.h5")):
            return True
        # cropped-raw/ : un run --video-adapt qui crashe à l'étape
        # "object detector training" n'y laisse que des *before_adapt*
        # (prédictions pré-adaptation). Ça ne compte PAS comme traité :
        # il faut reprendre. On exige au moins un .h5 hors before_adapt.
        cropped_raw = DLC_OUTPUT_DIR / session_id / "cropped-raw"
        if cropped_raw.exists():
            adapted_h5 = [
                p for p in cropped_raw.glob("*.h5")
                if "before_adapt" not in p.name
            ]
            if adapted_h5:
                return True
        return False

    # multi-animal / custom : .h5 directement à la racine de dlc-output/<session>/
    out = DLC_OUTPUT_DIR / session_id
    if not out.exists():
        return False
    return any(f.suffix == ".h5" for f in out.iterdir() if f.is_file())


def list_unprocessed_sessions(mode: str = "superanimal") -> list[str]:
    if not RAW_DIR.exists():
        return []
    return sorted(
        d.name for d in RAW_DIR.iterdir()
        if d.is_dir() and not d.name.startswith(".") and not is_processed(d.name, mode)
    )


def run_superanimal(
    session_id: str,
    superanimal_name: str,
    model_name: str,
    detector_name: str,
    video_adapt: bool,
) -> None:
    try:
        import deeplabcut
    except ImportError:
        print(
            "❌ DeepLabCut non installé. Active l'env conda 'dlc' :\n"
            "   conda activate dlc",
            file=sys.stderr,
        )
        sys.exit(1)

    metadata = load_session_metadata(session_id)
    source = get_source_video(metadata)

    output_dir = DLC_OUTPUT_DIR / session_id
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Vidéo : {source}")
    print(f"SuperAnimal : {superanimal_name}")
    print(f"  modèle    : {model_name}")
    print(f"  détecteur : {detector_name}")
    print(f"  video_adapt = {video_adapt}")

    deeplabcut.video_inference_superanimal(
        [str(source)],
        superanimal_name=superanimal_name,
        model_name=model_name,
        detector_name=detector_name,
        videotype="mp4",
        video_adapt=video_adapt,
        dest_folder=str(output_dir),
    )

    print(f"\n✅ Inférence SuperAnimal terminée : {output_dir}")
    print("   Étape suivante : `python scripts/assign_arenas.py "
          f"{session_id}` pour splitter par arène.")


def run_superanimal_cropped(
    session_id: str,
    superanimal_name: str = "superanimal_topviewmouse",
    model_name: str = "hrnet_w32",
    detector_name: str = "fasterrcnn_resnet50_fpn_v2",
    video_adapt: bool = False,
    likelihood_threshold: float = 0.6,
    interp_limit: int = 25,
    output_dir: Path | None = None,
) -> None:
    """
    Inférence SuperAnimal single-animal sur les vidéos déjà croppées d'une session.

    Pré-requis : avoir lancé `crop_arenes.py <session>` avant, pour avoir
    `data/cropped/<session>/<session>_A*.mp4`.

    Pour chaque vidéo croppée :
    1. Inférence SuperAnimal avec max_individuals=1 (pas de tracking inter-animal,
       beaucoup plus simple)
    2. Aplatissement du h5 (suppression du niveau 'individuals')
    3. Nettoyage (low-lk → NaN, interpolation des trous courts)
    4. Écriture dans data/vame-input/<session>/<session>_<arene>.h5

    Pas d'`assign_arenas` à faire ensuite — la sortie va directement à VAME.
    """
    try:
        import deeplabcut
    except ImportError:
        print(
            "❌ DeepLabCut non installé. Active l'env conda 'dlc' :\n"
            "   conda activate dlc",
            file=sys.stderr,
        )
        sys.exit(1)

    import pandas as pd
    # Réutilise la fonction de nettoyage d'assign_arenas
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from assign_arenas import clean_individual

    cropped_dir = CROPPED_DIR / session_id
    videos = sorted(cropped_dir.glob(f"{session_id}_A*.mp4"))
    if not videos:
        raise FileNotFoundError(
            f"Aucune vidéo croppée dans {cropped_dir}.\n"
            f"   Lance d'abord (env ethoflow) :\n"
            f"   conda activate ethoflow && python scripts/crop_arenes.py {session_id}"
        )

    print(f"Vidéos croppées trouvées ({len(videos)}) : {[v.name for v in videos]}")

    # Sortie temporaire des h5 bruts (sera nettoyée et déplacée après)
    temp_dest = DLC_OUTPUT_DIR / session_id / "cropped-raw"
    temp_dest.mkdir(parents=True, exist_ok=True)

    print(f"SuperAnimal single-animal (max_individuals=1)")
    print(f"  modèle    : {model_name}")
    print(f"  détecteur : {detector_name}")

    deeplabcut.video_inference_superanimal(
        [str(v) for v in videos],
        superanimal_name=superanimal_name,
        model_name=model_name,
        detector_name=detector_name,
        videotype="mp4",
        video_adapt=video_adapt,
        max_individuals=1,
        dest_folder=str(temp_dest),
    )

    # Post-traitement : flatten + clean + déplacement vers vame-input
    base_out = Path(output_dir) if output_dir else VAME_INPUT_DIR
    out_dir = base_out / session_id
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nPost-traitement → {out_dir}\n")

    for video in videos:
        # Cherche le h5 produit par DLC pour cette vidéo (préfixe = video.stem)
        h5_candidates = list(temp_dest.glob(f"{video.stem}*.h5"))
        h5_candidates = [c for c in h5_candidates if "filtered" not in c.stem]
        if not h5_candidates:
            print(f"  ⚠️  {video.stem} : pas de .h5 produit")
            continue
        # Avec --video-adapt, cropped-raw/ contient à la fois la prédiction
        # *_before_adapt et la version adaptée. On veut TOUJOURS l'adaptée
        # pour VAME — sinon le fine-tuning ne sert à rien.
        adapted = [c for c in h5_candidates if "before_adapt" not in c.name]
        if adapted:
            h5_candidates = adapted
        elif any("before_adapt" in c.name for c in h5_candidates):
            print(f"  ⚠️  {video.stem} : seulement du before_adapt trouvé — "
                  f"l'adaptation n'a pas abouti, qualité non optimale")
        produced_h5 = sorted(h5_candidates)[0]
        df = pd.read_hdf(produced_h5)

        # Aplatissement : drop le niveau 'individuals' (max 1 individu)
        if "individuals" in df.columns.names:
            df = df.droplevel("individuals", axis=1)

        # Nettoyage
        df_clean, stats = clean_individual(df, likelihood_threshold, interp_limit)

        # Nom de sortie : <session>_<arene>.h5 (arene = suffixe de video.stem)
        arena_suffix = video.stem.rsplit("_", 1)[-1]  # "..._A1" → "A1"
        out_path = out_dir / f"{session_id}_{arena_suffix}.h5"
        # key="df_with_missing" est la convention DLC, requise par VAME / movement
        df_clean.to_hdf(out_path, key="df_with_missing", mode="w", format="table")

        total = stats.get("total_slots") or 1
        pct_useful = 100 - 100 * stats["n_remaining_nan"] / total
        print(f"  ✓ {arena_suffix} → {out_path.name}  "
              f"({pct_useful:.1f}% utilisables après nettoyage)")

    print(f"\n✅ Single-animal cropped terminé : {out_dir}")


def run_custom(session_id: str) -> None:
    try:
        import deeplabcut
    except ImportError:
        print("❌ DeepLabCut non installé.", file=sys.stderr)
        sys.exit(1)

    if not PIPELINE_CONFIG.exists():
        raise FileNotFoundError(
            f"Config absente : {PIPELINE_CONFIG}\n"
            f"Copie configs/pipeline_config.yaml.example et adapte-le."
        )
    with open(PIPELINE_CONFIG) as f:
        config = yaml.safe_load(f)
    dlc_project_config = config.get("dlc_project_config")
    if not dlc_project_config:
        raise ValueError("Clé 'dlc_project_config' manquante dans pipeline_config.yaml")

    metadata = load_session_metadata(session_id)
    source = get_source_video(metadata)

    output_dir = DLC_OUTPUT_DIR / session_id
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Modèle DLC custom : {dlc_project_config}")
    print(f"Vidéo : {source}")

    deeplabcut.analyze_videos(
        config=dlc_project_config,
        videos=[str(source)],
        save_as_csv=True,
        destfolder=str(output_dir),
    )
    print(f"\n✅ Inférence custom terminée : {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inférence DLC sur une ou plusieurs sessions.")
    parser.add_argument("session_ids", nargs="*", help="Un ou plusieurs session_id à traiter")
    parser.add_argument("--all", action="store_true",
                        help="Traiter toutes les sessions de data/raw/ qui n'ont pas encore de sortie DLC")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Ignorer les sessions qui ont déjà une sortie DLC (utile en combinaison avec une liste)")
    parser.add_argument("--mode",
                        choices=["superanimal", "single-animal", "custom"],
                        default="superanimal",
                        help="superanimal: multi-animal sur vidéo entière ; "
                             "single-animal: SuperAnimal sur vidéos déjà croppées "
                             "(une souris par vidéo, sortie directe vers vame-input/) ; "
                             "custom: modèle DLC custom configuré dans pipeline_config.yaml")
    parser.add_argument("--superanimal-name", default="superanimal_topviewmouse")
    parser.add_argument("--superanimal-model", default="hrnet_w32")
    parser.add_argument("--superanimal-detector", default="fasterrcnn_resnet50_fpn_v2")
    parser.add_argument("--video-adapt", action="store_true",
                        help="Active le fine-tuning court (plus précis, plus lent)")
    parser.add_argument("--likelihood-threshold", type=float, default=0.6,
                        help="(mode single-animal) seuil de likelihood pour le nettoyage")
    parser.add_argument("--interp-limit", type=int, default=25,
                        help="(mode single-animal) taille max d'un trou interpolable, en frames")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="(mode single-animal) dossier de sortie alternatif "
                             "(défaut: data/vame-input/). Utile pour comparer "
                             "plusieurs runs sans écraser.")
    args = parser.parse_args()

    # Collecte de la liste de sessions à traiter (mode-aware : pour
    # single-animal, on regarde cropped-raw/ ; pour multi-animal, on regarde
    # le .h5 à la racine de dlc-output/<session>/)
    if args.all:
        sessions = list_unprocessed_sessions(args.mode)
        if not sessions:
            print(f"Aucune session à traiter en mode '{args.mode}' "
                  f"(toutes ont déjà une sortie).")
            sys.exit(0)
        print(f"{len(sessions)} session(s) non traitée(s) en mode "
              f"'{args.mode}' : {sessions}\n")
    elif args.session_ids:
        sessions = list(args.session_ids)
        if args.skip_existing:
            sessions = [s for s in sessions if not is_processed(s, args.mode)]
            if not sessions:
                print("Toutes les sessions demandées sont déjà traitées.")
                sys.exit(0)
    else:
        parser.print_help()
        sys.exit(1)

    n_ok = n_fail = 0
    for i, session_id in enumerate(sessions, 1):
        print(f"\n{'='*60}\n[{i}/{len(sessions)}] {session_id}\n{'='*60}")
        try:
            if args.mode == "superanimal":
                run_superanimal(
                    session_id,
                    superanimal_name=args.superanimal_name,
                    model_name=args.superanimal_model,
                    detector_name=args.superanimal_detector,
                    video_adapt=args.video_adapt,
                )
            elif args.mode == "single-animal":
                run_superanimal_cropped(
                    session_id,
                    superanimal_name=args.superanimal_name,
                    model_name=args.superanimal_model,
                    detector_name=args.superanimal_detector,
                    video_adapt=args.video_adapt,
                    likelihood_threshold=args.likelihood_threshold,
                    interp_limit=args.interp_limit,
                    output_dir=args.output_dir,
                )
            else:
                run_custom(session_id)
            n_ok += 1
        except (FileNotFoundError, ValueError) as e:
            print(f"❌ {session_id} : {e}", file=sys.stderr)
            n_fail += 1
            # On continue sur les autres sessions plutôt que d'aborter le batch
            continue

    if len(sessions) > 1:
        print(f"\n{'='*60}\nBatch terminé : {n_ok} OK, {n_fail} échec(s) sur {len(sessions)}\n{'='*60}")
    if n_fail > 0:
        sys.exit(1)
