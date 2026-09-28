"""Active learning : cible les frames où le modèle galère et te demande
de les corriger pour améliorer spécifiquement les keypoints à faible
likelihood (en pratique : les pattes).

⚠  Recommandation Tony (VAME/LIN) : **la sélection manuelle bat la
sélection automatique** pour ce genre de refinement. L'auto-detect
(cf. OUTLIER_ALGORITHM ci-dessous) attrape des cas évidents rapidement,
mais le vrai levier c'est ta passe à toi : regarder les vidéos
analysées, identifier les patterns d'échec (rearing, occlusion,
ambiguïté L/R), puis extraire à la main **50-100 frames AU TOTAL**,
réparties entre ces situations — c'est un budget global, pas un quota
par situation. Utilise ce script comme premier passage rapide, puis
enchaîne une passe manuelle dans la GUI DLC.

Pourquoi ce workflow plutôt que +epochs ou +mice :

    +epochs    : à 100 epochs et transfer learning depuis Quadruped, le
                 modèle est convergé. La loss train continue à descendre
                 mais c'est du overfit, pas du signal.

    +mice      : si tu rajoutes 3 souris × 20 frames aléatoires, tu auras
                 60 frames de plus mais elles couvriront surtout les
                 mêmes postures faciles. Les pattes étendues ou occultées
                 — celles qui plantent le modèle — restent sous-représentées.

    refine     : cible EXACTEMENT les frames problématiques (jumps
                 inter-frame, likelihood basse) et te les sert. Tu
                 labellises 30-40 frames qui valent chacune autant que
                 100 frames aléatoires. C'est ce que la doc DLC appelle
                 "iterative refinement" et c'est le bon outil ici.

    ⚠ **Continue le training depuis le snapshot précédent**, pas
    from scratch. Le training reprend automatiquement depuis le
    dernier snapshot enregistré si tu ne changes pas d'iteration.

Pré-requis :
    - 03_apply.py a tourné sur les vidéos listées dans
      TRAINING_VIDEOS_FOR_REFINE (.h5 doit exister dans
      <PROJECT_DIR>/result-videos/<stem>/)
    - conda activate dlc

Workflow complet :

    1. (préalable) Édite TRAINING_VIDEOS_FOR_REFINE dans _config.py pour
       inclure les vidéos du training set que tu veux raffiner. Par
       défaut ne contient que PILOT_VIDEO ; idéalement mets-en plusieurs
       pour diversifier les mice.

    2. (préalable) Analyse-les si pas déjà fait : ajoute-les à
       VIDEOS_TO_ANALYZE et lance `03_apply.py`. La vidéo annotée n'est
       pas nécessaire pour cette étape, tu peux passer
       MAKE_LABELED_VIDEO=False le temps de juste produire les .h5.

    3. Lance ce script : python scripts/dlc_model-training/05_refine_outliers.py
       Il appelle extract_outlier_frames pour chaque vidéo.

    4. Ouvre la GUI de refinement :
           conda activate dlc
           python
           >>> import deeplabcut as dlc
           >>> import sys
           >>> sys.path.insert(0, "scripts/dlc_model-training")
           >>> from _config import CONFIG
           >>> dlc.refine_labels(CONFIG)
       Concentre-toi sur les pattes. Pour chaque frame, fais glisser le
       marker à sa vraie position. Ctrl+S régulièrement.

    5. Fusionne les corrections au training set :
           >>> dlc.merge_datasets(CONFIG)

    6. Re-crée le training dataset et relance 02_train.py (ce script
       refera create_training_dataset depuis le dataset enrichi).

    7. Évalue : si le test rmse_pcutoff baisse et que les pattes
       montent en likelihood, c'est gagné. Sinon, deuxième passe de
       refine_labels.

Note sur l'algorithme "jump" :
    DLC compare la position d'un keypoint entre frames consécutives.
    Si elle saute de plus de OUTLIER_EPSILON pixels (sans cohérence
    physique souris), c'est flaggé. Parfait pour les pattes flickantes.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

# Insère le dossier du script en tête de sys.path pour trouver _load_config
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _load_config import add_config_dir_arg, completer_videos, load_config  # noqa: E402


def compter_machinelabels(labeled_dir: Path) -> tuple[int, int]:
    """(frames prédites, frames réellement nouvelles) dans un dossier.

    Les machinelabels sont les prédictions du modèle sur les frames
    extraites : c'est ce que `refine_labels` propose de corriger. Mais
    seules celles qui ne sont PAS déjà annotées à la main apportent
    quelque chose.

    Le recouvrement est la règle plutôt que l'exception : kmeans est
    déterministe, donc sur une vidéo déjà minée il re-sélectionne
    exactement les mêmes centroïdes — c'est-à-dire les frames qu'on a
    labellisées au tour précédent. On obtient alors des prédictions sur
    du déjà-fait, et les sauvegarder dans la GUI écrase de vraies
    annotations par des prédictions d'un modèle encore faible.
    """
    fichiers = sorted(labeled_dir.glob("machinelabels-iter*.h5"))
    if not fichiers:
        return 0, 0
    try:
        import pandas as pd
        machine = pd.read_hdf(fichiers[-1])
    except Exception:
        return 0, 0

    deja = set()
    for h5 in labeled_dir.glob("CollectedData_*.h5"):
        try:
            import pandas as pd
            deja |= {str(i) for i in pd.read_hdf(h5).index}
        except Exception:
            continue
    nouvelles = sum(1 for i in machine.index if str(i) not in deja)
    return len(machine), nouvelles


def update_numframes2pick(project_config_path: Path, n: int) -> int:
    """Met à jour `numframes2pick` dans le config.yaml du projet.

    Retourne l'ancienne valeur (pour info / restore manuel).
    Identique au helper utilisé par 04_add_videos.py.
    """
    with open(project_config_path) as f:
        cfg = yaml.safe_load(f)
    old = cfg.get("numframes2pick", 20)
    if old == n:
        return old
    cfg["numframes2pick"] = n
    with open(project_config_path, "w") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
    return old


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    add_config_dir_arg(parser)
    parser.add_argument(
        "--videos", nargs="+", default=None, metavar="CHEMIN",
        help="Vidéos à miner, en remplacement de TRAINING_VIDEOS_FOR_REFINE. "
             "Chacune doit déjà avoir été analysée par 03_apply.py.",
    )
    args = parser.parse_args()
    load_config(args)

    import deeplabcut as dlc  # noqa: E402
    from _config import (  # noqa: E402
        CONFIG, OUTLIER_ALGORITHM, OUTLIER_EPSILON, OUTLIER_NUMFRAMES,
        PROJECT_DIR, RESULTS_DIR, TRAINING_VIDEOS_FOR_REFINE,
    )
    try:
        from _config import OUTLIER_P_BOUND  # noqa: E402
    except ImportError:
        # _config.py antérieur à l'ajout du paramètre (config généré par
        # une ancienne version du wizard 00).
        OUTLIER_P_BOUND = 0.6

    # `--videos` prime sur _config.py : miner une vidéo de plus ne doit pas
    # demander d'éditer un fichier de config.
    videos_cibles = ([Path(v) for v in args.videos] if args.videos
                     else completer_videos(TRAINING_VIDEOS_FOR_REFINE,
                                            "TRAINING_VIDEOS_FOR_REFINE"))
    if not videos_cibles:
        print("⚠ Aucune vidéo à traiter.\n"
              "   Passe --videos <chemin> [<chemin> ...], ou renseigne\n"
              "   TRAINING_VIDEOS_FOR_REFINE dans ton _config.py.")
        return

    # Vérifie que chaque vidéo a bien été analysée (.h5 doit exister)
    print("Vérification des prédictions existantes...\n")
    ready: list[Path] = []
    for video in videos_cibles:
        if not video.exists():
            print(f"⚠ skip : vidéo introuvable {video}")
            continue
        out_dir = RESULTS_DIR / video.stem
        h5_files = list(out_dir.glob("*.h5"))
        if not h5_files:
            print(
                f"⚠ skip : pas de .h5 dans {out_dir}\n"
                f"   → ajoute {video.name} à VIDEOS_TO_ANALYZE et lance 03_apply.py"
            )
            continue
        ready.append(video)
        print(f"   ✓ {video.name}  ({h5_files[0].name})")

    if not ready:
        print("\n❌ Aucune vidéo prête. Lance 03_apply.py d'abord.")
        sys.exit(1)

    # Chaque algorithme n'utilise QUE son propre seuil : afficher celui
    # qui ne sert pas embrouille le diagnostic quand rien n'est extrait.
    seuil = (f"  p_bound (confiance) : {OUTLIER_P_BOUND}"
             if OUTLIER_ALGORITHM == "uncertain"
             else f"  epsilon (px)        : {OUTLIER_EPSILON}")
    print(
        f"\nExtraction d'outliers sur {len(ready)} vidéo(s) :\n"
        f"  algo                : {OUTLIER_ALGORITHM}\n"
        f"{seuil}\n"
        f"  max frames / vidéo  : {OUTLIER_NUMFRAMES}\n"
    )

    # `extract_outlier_frames` utilise numframes2pick du config.yaml pour
    # déterminer combien de frames extraire. On l'aligne sur OUTLIER_NUMFRAMES
    # ici (peut différer de NEW_VIDEO_FRAMES réglé par 04_add_videos.py).
    project_config = Path(CONFIG)
    old = update_numframes2pick(project_config, OUTLIER_NUMFRAMES)
    if old != OUTLIER_NUMFRAMES:
        print(f"  numframes2pick : {old} → {OUTLIER_NUMFRAMES}\n")

    # IMPORTANT : on passe `destfolder=` pour que DLC aille chercher les .h5
    # dans <result-videos>/<stem>/ au lieu du dossier de la vidéo source
    # (par défaut DLC les cherche à côté de la vidéo).
    # Le dossier des nouvelles frames extraites par DLC :
    labeled_data_root = PROJECT_DIR / "labeled-data"
    total_ajoutees = [0]

    for video in ready:
        out_dir = RESULTS_DIR / video.stem
        labeled_dir = labeled_data_root / video.stem
        # Compte les PNG avant pour pouvoir détecter si rien n'a été extrait
        before = len(list(labeled_dir.glob("*.png"))) if labeled_dir.exists() else 0

        print(f"→ {video.name}")
        dlc.extract_outlier_frames(
            CONFIG,
            [str(video)],
            outlieralgorithm=OUTLIER_ALGORITHM,
            epsilon=OUTLIER_EPSILON,
            p_bound=OUTLIER_P_BOUND,  # sinon DLC applique 0.01, trop bas
            extractionalgorithm="kmeans",
            automatic=True,  # pas de GUI à ce stade — juste extraction
            destfolder=str(out_dir),  # va chercher le .h5 dans result-videos/
        )

        after = len(list(labeled_dir.glob("*.png"))) if labeled_dir.exists() else 0
        added = after - before
        # Le nombre de PNG ajoutés ne suffit pas à juger : kmeans est
        # déterministe, donc relancer l'extraction sur la même vidéo avec
        # le même `numframes2pick` re-sélectionne EXACTEMENT les mêmes
        # frames. Le compte ne bouge pas alors que tout s'est bien passé.
        # Ce qui compte vraiment, c'est le nombre de frames en attente de
        # correction, c'est-à-dire les lignes des machinelabels.
        predites, nouvelles = compter_machinelabels(labeled_dir)
        total_ajoutees[0] += nouvelles
        if nouvelles > 0:
            print(f"   ✅ {nouvelles} frame(s) à corriger dans "
                  f"labeled-data/{video.stem}/"
                  + (f"  ({predites - nouvelles} déjà annotée(s) à la main, "
                     f"ignorée(s))" if predites > nouvelles else "") + "\n")
        elif predites > 0:
            print(f"   ⚠ Les {predites} frame(s) prédites sont TOUTES déjà "
                  f"annotées à la main — rien à gagner ici.\n"
                  f"     kmeans est déterministe : sur une vidéo déjà minée, "
                  f"il re-choisit les mêmes\n"
                  f"     frames que celles que tu as labellisées au tour "
                  f"précédent.\n"
                  f"     Ne sauvegarde PAS les machinelabels dans la GUI : tu "
                  f"remplacerais tes annotations\n"
                  f"     par les prédictions du modèle. Supprime-les :\n"
                  f"       del {labeled_dir}\\machinelabels-iter*.h5\n"
                  f"     Pour de vraies nouvelles frames, passe par d'autres "
                  f"vidéos :\n"
                  f"       python scripts/dlc_model-training/04_add_videos.py "
                  f"--videos <...>\n")
        else:
            if OUTLIER_ALGORITHM == "uncertain":
                cause = (
                    f"     - p_bound trop bas ({OUTLIER_P_BOUND}) : aucune "
                    f"prédiction n'est en dessous.\n"
                    f"       Attention au contresens — « 0 outlier » ne veut "
                    f"pas dire « tout va bien ».\n"
                    f"       Si la vidéo annotée est vide, c'est l'inverse : "
                    f"les confiances sont basses\n"
                    f"       mais toutes au-dessus du seuil. Vérifie avec :\n"
                    f"         python scripts/dlc_model-training/"
                    f"inspect_predictions.py --model-dir {PROJECT_DIR}\n"
                )
            else:
                cause = (f"     - epsilon trop strict ({OUTLIER_EPSILON} px) : "
                         f"peu de jumps détectés\n")
            print(
                f"   ⚠ Aucune frame extraite. Causes possibles :\n"
                f"{cause}"
                f"     - .h5 non trouvé dans {out_dir} (regarde les logs DLC ci-dessus)\n"
            )

    # Sans frame extraite, il n'y a rien à raffiner ni à fusionner. Et
    # `merge_datasets` n'est pas neutre quand il ne fusionne rien : il
    # incrémente quand même l'itération du projet, après quoi DLC cherche
    # un modèle pour une itération qui n'a jamais été entraînée et refuse
    # de servir l'inférence (« Could not find a shuffle […] Known
    # shuffles: none »). Le modèle est intact, mais le projet ne le voit
    # plus — et le message d'erreur n'oriente pas du tout vers la cause.
    if total_ajoutees[0] == 0:
        print(
            "Aucune frame extraite au total — il n'y a rien à raffiner.\n"
            "\n"
            "⚠ Ne lance PAS `dlc.merge_datasets(CONFIG)` maintenant : sans\n"
            "  correction à fusionner, il se contenterait d'incrémenter\n"
            "  l'itération du projet, et l'inférence échouerait ensuite avec\n"
            "  « Could not find a shuffle ». Réparation, le cas échéant :\n"
            "      python scripts/diagnose_dlc_model.py --model-dir "
            f"{PROJECT_DIR} --fix\n"
            "\n"
            "Commence par comprendre pourquoi rien n'est ressorti (voir\n"
            "les causes indiquées plus haut).\n"
        )
        return

    print(
        "Étapes suivantes :\n"
        "  1. Ouvre la GUI de raffinement :\n"
        "       conda activate dlc\n"
        "       python\n"
        "       >>> import deeplabcut as dlc\n"
        "       >>> import sys\n"
        "       >>> sys.path.insert(0, 'scripts/dlc_model-training')\n"
        "       >>> from _config import CONFIG\n"
        "       >>> dlc.refine_labels(CONFIG)\n"
        "     Corrige les pattes (drag and drop). Ctrl+S régulièrement.\n"
        "\n"
        "  2. Fusionne les corrections au dataset d'entraînement :\n"
        "       >>> dlc.merge_datasets(CONFIG)\n"
        "\n"
        "  3. NETTOIE les caches AVANT de re-créer le training dataset\n"
        "     (sinon 02_train ré-utilisera l'ancien dataset, sans tes\n"
        "     corrections — bug rencontré en phase 2) :\n"
        "       Remove-Item -Recurse -Force\n"
        "         \"<PROJECT_DIR>\\training-datasets\"\n"
        "       Remove-Item -Recurse -Force\n"
        "         \"<PROJECT_DIR>\\dlc-models-pytorch\"\n"
        "\n"
        "  4. Relance le training :\n"
        "       python scripts/dlc_model-training/02_train.py\n"
        "\n"
        "  5. Évalue : test rmse_pcutoff doit baisser, et la likelihood\n"
        "     moyenne des pattes doit monter (regarde la vidéo annotée\n"
        "     à pcutoff=0.5 — si les pattes apparaissent maintenant,\n"
        "     c'est gagné)."
    )


if __name__ == "__main__":
    main()
