"""Active learning : cible les frames où le modèle galère et te demande
de les corriger pour améliorer spécifiquement les keypoints à faible
likelihood (en pratique : les pattes).

Pourquoi ce workflow plutôt que +epochs ou +mice :

    +epochs    : à 100 epochs et transfer learning depuis Quadruped, le
                 modèle est convergé. La loss train continue à descendre
                 mais c'est du overfit, pas du signal.

    +mice      : si tu rajoutes 3 souris × 20 frames aléatoires, tu auras
                 60 frames de plus mais elles couvriront surtout les
                 mêmes postures faciles. Les pattes étendues ou occultées
                 — celles qui plantent le modèle — restent sous-représentées.

    refine     : DLC identifie EXACTEMENT les frames problématiques (jumps
                 inter-frame, likelihood basse) et te les sert. Tu
                 labellises 30-40 frames qui valent chacune autant que
                 100 frames aléatoires. C'est ce que la doc DLC appelle
                 "iterative refinement" et c'est le bon outil ici.

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

    3. Lance ce script : python scripts/dlc_bottomview/05_refine_outliers.py
       Il appelle extract_outlier_frames pour chaque vidéo.

    4. Ouvre la GUI de refinement :
           conda activate dlc
           python
           >>> import deeplabcut as dlc
           >>> import sys
           >>> sys.path.insert(0, "scripts/dlc_bottomview")
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

import sys
from pathlib import Path

import deeplabcut as dlc

# Import du config centralisé
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _config import (  # noqa: E402
    CONFIG,
    OUTLIER_ALGORITHM,
    OUTLIER_EPSILON,
    OUTLIER_NUMFRAMES,
    RESULTS_DIR,
    TRAINING_VIDEOS_FOR_REFINE,
)


def main() -> None:
    if not TRAINING_VIDEOS_FOR_REFINE:
        print("⚠ TRAINING_VIDEOS_FOR_REFINE est vide dans _config.py")
        return

    # Vérifie que chaque vidéo a bien été analysée (.h5 doit exister)
    print("Vérification des prédictions existantes...\n")
    ready: list[Path] = []
    for video in TRAINING_VIDEOS_FOR_REFINE:
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

    print(
        f"\nExtraction d'outliers sur {len(ready)} vidéo(s) :\n"
        f"  algo                : {OUTLIER_ALGORITHM}\n"
        f"  epsilon (px)        : {OUTLIER_EPSILON}\n"
        f"  max frames / vidéo  : {OUTLIER_NUMFRAMES}\n"
    )

    # extract_outlier_frames prend la liste complète des vidéos d'un coup
    # mais on passe vidéo par vidéo pour des logs plus lisibles et pour
    # que numframes2pick s'applique par vidéo (et pas globalement).
    for video in ready:
        print(f"→ {video.name}")
        dlc.extract_outlier_frames(
            CONFIG,
            [str(video)],
            outlieralgorithm=OUTLIER_ALGORITHM,
            epsilon=OUTLIER_EPSILON,
            extractionalgorithm="kmeans",
            automatic=True,  # pas de GUI à ce stade — juste extraction
        )
        print(f"   ✅ outliers extraits\n")

    print(
        "Étapes suivantes :\n"
        "  1. Ouvre la GUI de raffinement :\n"
        "       conda activate dlc\n"
        "       python\n"
        "       >>> import deeplabcut as dlc\n"
        "       >>> import sys\n"
        "       >>> sys.path.insert(0, 'scripts/dlc_bottomview')\n"
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
        "       python scripts/dlc_bottomview/02_train.py\n"
        "\n"
        "  5. Évalue : test rmse_pcutoff doit baisser, et la likelihood\n"
        "     moyenne des pattes doit monter (regarde la vidéo annotée\n"
        "     à pcutoff=0.5 — si les pattes apparaissent maintenant,\n"
        "     c'est gagné)."
    )


if __name__ == "__main__":
    main()
