"""Régénère les vidéos annotées `_labeled.mp4` à un seuil de confiance choisi.

L'inférence (`run_dlc_inference.py --mode custom`) passe le modèle sur
toutes les frames et écrit un `.h5` de prédictions — coordonnées ET
likelihood par keypoint et par frame. Elle ne produit aucune vidéo
annotée.

Ce script relit ce `.h5` et redessine simplement les keypoints au-dessus
de chaque frame, en ne gardant que ceux dont la likelihood dépasse le
seuil. Pas de ré-inférence : ~30 s pour une vidéo de 20 min, contre
plusieurs minutes de GPU.

**Pourquoi deux seuils par défaut.** À 0.6 (le défaut DLC), les keypoints
dont le modèle n'est pas sûr disparaissent purement et simplement de
l'image : la vidéo est propre, mais elle cache ce que le modèle fait
vraiment. À 0.3, tu vois ses hésitations — une patte qui saute, un point
qui colle à un reflet. Les deux côte à côte, c'est le contrôle visuel
utile. Le seuil est dans le nom du fichier, donc ils cohabitent.

Usage :
    # Une session, les deux seuils par défaut
    python scripts/relabel_video.py --session BV-970

    # Seuils choisis
    python scripts/relabel_video.py --session BV-970 --pcutoffs 0.2 0.5 0.8

    # Plusieurs sessions d'un coup
    python scripts/relabel_video.py --sessions BV-970 BV-971

    # Sans argument : la session est demandée à l'invite
    python scripts/relabel_video.py

Pré-requis :
    - conda activate dlc  (a besoin de deeplabcut)
    - Un modèle DLC configuré (`dlc_project_config` dans
      configs/pipeline_config.yaml) : `create_labeled_video` a besoin du
      `config.yaml` d'un projet DLC pour connaître bodyparts et skeleton.
      Le mode SuperAnimal n'en a pas — ce script ne s'y applique pas.
    - Le `.h5` de la session dans <project>/data/dlc-output/<session>/
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import (  # noqa: E402
    add_project_dir_arg,
    dlc_output_dir,
    raw_dir,
    resolve_project,
)
from interactive import prompt_session  # noqa: E402

# Seuils par défaut : le permissif qui montre les hésitations du modèle,
# et le défaut DLC qui montre le rendu de production.
DEFAULT_PCUTOFFS = [0.3, 0.6]


def pcutoff_tag(pcutoff: float) -> str:
    """`0.3` → `p30`. Le seuil dans le nom, sinon deux passes s'écrasent."""
    return f"p{int(round(pcutoff * 100)):02d}"


def find_labeled_video(session_out: Path, tag: str) -> Path | None:
    """Cherche une vidéo annotée déjà produite pour ce seuil.

    DLC nomme sa sortie `<original>_labeled.mp4` sans y encoder le
    pcutoff : on renomme donc nous-mêmes en `_labeled_<tag>.mp4`. Le
    fallback sur le nom DLC standard couvre les fichiers produits avant
    cette convention.
    """
    for p in session_out.glob(f"*_labeled_{tag}.mp4"):
        return p
    candidates = list(session_out.glob("*_labeled.mp4"))
    if candidates:
        return max(candidates, key=lambda x: x.stat().st_mtime)
    return None


def generate_labeled_video(dlc_config: str, source_video: Path,
                            session_out: Path, pcutoff: float) -> Path | None:
    """Redessine la vidéo annotée au seuil demandé, depuis le `.h5` existant.

    Renvoie le chemin de la vidéo produite, ou None si la session n'a pas
    de prédictions à redessiner.
    """
    try:
        import deeplabcut as dlc
    except ImportError:
        print("❌ deeplabcut non trouvé — active `conda activate dlc`",
              file=sys.stderr)
        sys.exit(1)

    if not list(session_out.glob("*.h5")):
        print(f"  ⚠  {session_out.name} : pas de .h5 DLC, rien à redessiner.\n"
              f"     Lance d'abord run_dlc_inference.py --mode custom.")
        return None

    print(f"  Redessin à pcutoff={pcutoff}...")
    dlc.create_labeled_video(
        dlc_config,
        [str(source_video)],
        destfolder=str(session_out),
        pcutoff=pcutoff,
        draw_skeleton=True,
    )

    # DLC vient d'écrire (ou d'écraser) <stem>_labeled.mp4 : on le tague
    # avec son seuil pour que la passe suivante ne l'efface pas.
    tag = pcutoff_tag(pcutoff)
    latest = sorted(session_out.glob("*_labeled.mp4"),
                    key=lambda p: p.stat().st_mtime, reverse=True)
    if not latest:
        print("  ⚠  DLC n'a pas produit de vidéo annotée.", file=sys.stderr)
        return None
    labeled = latest[0]
    tagged = labeled.with_name(f"{labeled.stem}_{tag}.mp4")
    if tagged.exists():
        tagged.unlink()
    labeled.rename(tagged)
    return tagged


def find_source_video(project: Path, session_id: str) -> Path | None:
    """Vidéo source de la session, lue dans sa `metadata.yaml`."""
    meta_path = raw_dir(project) / session_id / "metadata.yaml"
    if not meta_path.exists():
        return None
    meta = yaml.safe_load(meta_path.read_text(encoding="utf-8")) or {}
    src = meta.get("source_video")
    if not src:
        return None
    p = Path(src)
    return p if p.exists() else None


def resolve_model_config(project: Path, no_prompt: bool = False) -> str:
    """Config.yaml du modèle DLC du projet, ou échec explicite.

    `create_labeled_video` a besoin des bodyparts et du skeleton, donc du
    `config.yaml` d'un vrai projet DLC. Le mode SuperAnimal n'en produit
    pas : autant le dire ici plutôt que de laisser DLC lever quelque
    chose d'incompréhensible trois appels plus loin.
    """
    from run_dlc_inference import (
        check_model_is_trained,
        check_project_path,
        resolve_dlc_config,
    )

    # Portée resserrée : seul l'échec de `resolve_dlc_config` signifie
    # « aucun modèle configuré ». Englober les vérifications suivantes
    # rhabillerait un modèle non entraîné en modèle absent — deux
    # problèmes qui ne se corrigent pas de la même façon.
    try:
        cfg = resolve_dlc_config(project, no_prompt=no_prompt)
    except SystemExit:
        print("\n❌ Aucun modèle DLC configuré pour ce projet.\n"
              "   Redessiner une vidéo annotée demande le config.yaml d'un\n"
              "   projet DLC (bodyparts + skeleton) — les sorties SuperAnimal\n"
              "   n'en ont pas. Configure `dlc_project_config` dans\n"
              "   configs/pipeline_config.yaml, ou passe par la page Projet\n"
              "   de l'app.", file=sys.stderr)
        raise

    # Mêmes garde-fous que l'inférence : `create_labeled_video` dérive le
    # nom du scorer depuis les métadonnées d'entraînement et sort
    # « Could not find a shuffle with trainingset fraction ... » quand le
    # modèle a été déplacé ou n'est pas entraîné. Ces deux fonctions
    # diagnostiquent (et réparent le chemin) au lieu de laisser passer
    # l'erreur cryptique de DLC.
    check_project_path(cfg)
    try:
        check_model_is_trained(cfg)
    except ValueError as e:
        # Le message porte déjà le diagnostic complet : l'afficher et
        # sortir proprement vaut mieux qu'une traceback, ce script étant
        # une commande de contrôle qu'on lance à la volée.
        print(f"\n❌ {e}", file=sys.stderr)
        sys.exit(1)
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    add_project_dir_arg(parser)
    parser.add_argument("--session", default=None,
                        help="Session à traiter. Demandée à l'invite si absente.")
    parser.add_argument("--sessions", nargs="+", default=None, metavar="ID",
                        help="Plusieurs sessions d'un coup.")
    parser.add_argument("--pcutoffs", nargs="+", type=float,
                        default=DEFAULT_PCUTOFFS, metavar="SEUIL",
                        help=f"Seuils de confiance à produire "
                             f"(défaut : {' '.join(map(str, DEFAULT_PCUTOFFS))}). "
                             f"Bas = on voit les hésitations du modèle, "
                             f"haut = rendu propre.")
    # `--no-prompt` vient d'add_project_dir_arg — le rajouter ici ferait
    # un conflit d'option à la construction du parser.
    args = parser.parse_args()

    project = resolve_project(args)
    dlc_config = resolve_model_config(project, no_prompt=args.no_prompt)

    sessions = list(args.sessions or [])
    if args.session:
        sessions.append(args.session)
    if not sessions:
        sessions = [prompt_session(project, None, no_prompt=args.no_prompt,
                                    title="Quelle session redessiner ?")]

    print(f"Modèle : {dlc_config}")
    print(f"Seuils : {', '.join(str(p) for p in args.pcutoffs)}\n")

    n_ok = 0
    for sid in sessions:
        print(f"→ {sid}")
        source = find_source_video(project, sid)
        if source is None:
            print(f"  ⚠  vidéo source introuvable (metadata.yaml), skip.",
                  file=sys.stderr)
            continue
        session_out = dlc_output_dir(project) / sid
        for pcutoff in args.pcutoffs:
            out = generate_labeled_video(dlc_config, source, session_out,
                                          pcutoff)
            if out is not None:
                print(f"  ✓ {out.name}")
                n_ok += 1
        print()

    print(f"✅ {n_ok} vidéo(s) annotée(s) — dans "
          f"{dlc_output_dir(project)}/<session>/")


if __name__ == "__main__":
    main()
