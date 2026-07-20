# EthoFlow

Pipeline d'analyse comportementale souris à partir de vidéos brutes, basé sur **DeepLabCut** (estimation de pose) et **VAME** (segmentation comportementale non-supervisée).

Le pipeline gère **deux dimensions indépendantes** :

- **Angle de la caméra** — top-view (au plafond) ou bottom-view (sous plancher transparent IR). Ça détermine uniquement quelle SuperAnimal utiliser pour le transfer learning DLC (`superanimal_topviewmouse` vs `superanimal_quadruped`).
- **Nombre d'animaux par vidéo** — 1 seul animal (**1 vidéo = 1 session**) ou plusieurs animaux dans des arènes physiquement séparées (**1 vidéo = N sessions**, typiquement 4). Ça détermine si tu as besoin d'un split par arène ou pas.

Les deux se combinent librement : tu peux faire du bottom-view avec 4 souris dans 4 arènes séparées, ou du top-view avec une souris seule dans une arène ouverte. Le mode d'inférence DLC (`--mode superanimal` multi-animal + arena split, ou crop préalable + `--mode custom` single-animal) est **choisi à l'étape 5** en fonction du nombre d'animaux par vidéo, indépendamment de l'angle.

> ℹ **Note sur le CLI** : dans `create_project.py`, le flag `--kind {topview, bottomview}` est un raccourci historique qui contrôle uniquement s'il faut écrire ou non des coordonnées d'arène par défaut. Concrètement, `--kind topview` = « multi-animal par vidéo, arena split nécessaire », `--kind bottomview` = « 1 animal par vidéo, pas d'arena split ». Le nom est un abus de langage — dans le doute, choisis en fonction du nombre d'animaux par vidéo, pas de l'angle caméra.

Le pipeline part d'une acquisition brute (vidéo + Excel des souris) et produit des CSV statistiques, des figures et des vidéos annotées, groupables par n'importe quelle variable expérimentale (génotype, traitement, sexe, etc.).

---

## Table des matières

1. [Concepts](#concepts)
2. [Prérequis machine](#prérequis-machine)
3. [Installation](#installation-first-time)
4. [Environnements conda](#environnements-conda)
5. [Parcours pipeline](#parcours-pipeline)
   - [Étape 0 — bifurcation modèle DLC](#étape-0--bifurcation-modèle-dlc)
   - [Étapes 1-9 — pipeline principal](#étape-1--créer-un-projet-ethoflow)
6. [Structure d'un projet](#structure-dun-projet)
7. [Index des scripts](#index-des-scripts)
8. [Troubleshooting](#troubleshooting)

---

## Concepts

**Un projet EthoFlow** = un dossier autonome qui contient les données brutes, les sorties DLC/VAME et la config, pour une expérience donnée. Chaque projet vit à un chemin absolu (ex : `D:\ethoflow\projects\bottomview-MCC-2026-06`) et tous les scripts prennent `--project-dir <chemin>`. Tu peux avoir autant de projets en parallèle que tu veux — ils sont indépendants.

**Une session** = une acquisition (une vidéo + les metadata associées : ID souris, groupe, traitement, date, etc.). Le nombre de sessions par vidéo dépend uniquement du **nombre d'animaux dans la vidéo** :
- 1 animal par vidéo → 1 session
- N animaux dans N arènes physiquement séparées → N sessions (une par arène)

Cette distinction est indépendante de l'angle de la caméra : tu peux avoir du bottom-view mono-animal, du bottom-view multi-animal, du top-view mono-animal, du top-view multi-animal.

**Un modèle DLC** = un réseau pré-entraîné qui détecte les points anatomiques (nez, oreilles, pattes, queue, etc.). Un modèle DLC vit **hors** du projet EthoFlow (dans `E:\dlc-projects\...` par exemple) et est **réutilisé** entre projets. C'est la partie coûteuse à produire (labellisation manuelle + jour de calcul GPU) et la partie qu'on partage entre expérimentateurs.

**Un modèle VAME** = un VAE entraîné à segmenter les séquences de pose en motifs comportementaux. Contrairement à DLC, VAME s'entraîne **une fois par projet** — sa segmentation dépend des animaux qui sont dedans.

---

## Prérequis machine

- **OS** : Windows 10/11 recommandé (le pipeline a été développé sur Windows). Linux et macOS fonctionnent pour du dev et de l'inférence légère.
- **GPU** : NVIDIA avec ≥ 8 GB VRAM. Sans GPU, l'entraînement DLC est inexploitable et l'inférence prend des heures par vidéo.
- **RAM** : 32 GB minimum, 64 GB confortable pour VAME avec beaucoup de sessions.
- **Disque** : SSD ≥ 500 GB pour le travail actif. HDD/NAS secondaire pour l'archivage.
- **Logiciels** : [Miniconda](https://docs.conda.io/projects/miniconda/), [Git](https://git-scm.com), [ffmpeg](https://ffmpeg.org) (livré avec l'env `ethoflow`).

---

## Installation (first-time)

### 1. Cloner le repo

```cmd
cd D:\
git clone <URL_ETHOFLOW> EthoFlow
cd EthoFlow
```

### 2. Créer les 3 environnements conda

```cmd
conda env create -f environment-pipeline.yml    :: env "ethoflow"
conda env create -f environment-dlc.yml         :: env "dlc"
conda env create -f environment-vame.yml        :: env "vame"
```

Chaque env prend 10-20 min. Ils sont isolés : DLC (torch/DeepLabCut) et VAME ont des dépendances incompatibles, donc **jamais mélanger**.

### 3. Vérifier la GPU (côté DLC)

```cmd
conda activate dlc
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU :', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
```

Doit afficher `CUDA: True` et le nom de ta carte. Sinon → [Troubleshooting CUDA](#cuda-non-détectée-ou-torch-cpu).

### 4. Vérifier VAME

```cmd
conda activate vame
python -c "import vame; print(vame.__version__)"
```

---

## Environnements conda

| Env | À quoi ça sert |
|---|---|
| `ethoflow` (env-pipeline) | Sert principalement à faire tourner l'app Streamlit (`streamlit run streamlit_app/app.py`) et l'orchestrateur `run_pipeline.py` |
| `dlc` (env-dlc) | DeepLabCut 3.x + PyTorch — inférence, entraînement, GUI de labellisation |
| `vame` (env-vame) | VAME + scipy/matplotlib/UMAP — setup/train/segment VAME, analyses, visualisations |

---

## Parcours pipeline

Un seul parcours de bout en bout, avec **une seule bifurcation** au début : soit tu as déjà un modèle DLC utilisable, soit tu dois l'entraîner. Après cette bifurcation, la suite est identique.

### Étape 0 — bifurcation modèle DLC

**As-tu déjà un modèle DLC entraîné pour ton setup imaging ?**

- **Oui** → tu as un dossier `E:\dlc-projects\<nom_projet>\` avec un `config.yaml` dedans, produit par ton labo ou une expérience précédente. Passe directement à l'**[Étape 1](#étape-1--créer-un-projet-ethoflow)**.
- **Non** → suis d'abord le [Parcours B — entraîner un nouveau modèle DLC](#parcours-b--entraîner-un-nouveau-modèle-dlc) tout en bas, qui produit ce fameux `config.yaml`. Puis reviens à l'**Étape 1**.

Le même modèle DLC entraîné une fois est réutilisé pour tous les projets EthoFlow futurs qui partagent le même setup imaging.

---

### Étape 1 — créer un projet EthoFlow

```cmd
conda activate ethoflow
python scripts\create_project.py ^
    --project-dir D:\ethoflow\projects\bottomview-MCC-2026-06 ^
    --kind bottomview ^
    --dlc-config "E:\dlc-projects\souris-bottomview-labo-2026-06-05\config.yaml"
```

Options :
- `--kind bottomview` — **1 animal par vidéo**, pas d'arena splitting (nom historique, s'applique aussi à du top-view mono-animal)
- `--kind topview` — **N animaux par vidéo dans N arènes**, arena splitting activé + coords d'arène par défaut écrites dans `pipeline_config.yaml` (nom historique, s'applique aussi à du bottom-view multi-animal)

Le choix se fait sur le nombre d'animaux par vidéo, pas sur l'angle caméra. Pour un projet bottom-view avec 4 souris dans 4 arènes séparées, choisis `--kind topview` puis édite les `default_arenes_coords` avec `calibrate_arenes.py`.

Résultat : arborescence vide + `configs/pipeline_config.yaml` qui pointe vers ton config DLC.

### Étape 2 — préparer l'Excel de sessions

Le pipeline lit un Excel maître qui décrit tes souris. Deux schémas selon **le nombre d'animaux par vidéo** :

**1 animal / vidéo** — schéma `Sessions` — 1 ligne par souris (=1 vidéo=1 session) :

| mouse_id | sex | group | cage | tail_label | birth_date | animal_id | line | genotype_mcc | captopril |
|---|---|---|---|---|---|---|---|---|---|
| 970 | F | MCCf/f | CD329 | 1 | 2024-10-15 | 54310 | MCC\*Cdh5-cre | fl/fl | oui |
| 971 | F | MCCiECKO | CD330 | 2 | 2024-10-15 | 54311 | MCC\*Cdh5-cre | fl/fl | oui |

`mouse_id` = nom du fichier vidéo attendu (`970.mp4`, `971.mp4`). `group` = ta variable de comparaison principale.

**N animaux / vidéo** — schéma `Trials_Videos` + `Subjects` + `Arena_Mapping` — 1 ligne par vidéo dans `Trials_Videos`, mapping arène↔souris dans `Arena_Mapping`. Voir `configs/metadata_template.yaml` pour un exemple complet.

### Étape 3 — sync des sessions

Deux scripts selon le schéma Excel utilisé :

**1 animal / vidéo** — `sync_from_excel_bottomview.py` (nom historique, marche pour tout mono-animal, quel que soit l'angle caméra) :

```cmd
python scripts\sync_from_excel_bottomview.py ^
    --project-dir D:\ethoflow\projects\bottomview-MCC-2026-06 ^
    --excel D:\ethoflow\projects\bottomview-MCC-2026-06\bottomview_sessions.xlsx ^
    --videos-dir E:\data\bottom_view\08062026 ^
    --date 2026-06-08
```

Répète la commande pour chaque batch d'acquisition (`--videos-dir` change, l'Excel reste le même). Utilise `--overwrite` pour re-générer sur une metadata déjà existante.

**N animaux / vidéo** — `sync_from_excel.py` (nom historique, marche pour tout multi-animal) :

```cmd
python scripts\sync_from_excel.py ^
    --project-dir D:\ethoflow\projects\openfield-M1-2025-10 ^
    --excel D:\path\to\OpenField_trials.xlsx
```

Résultat dans les deux cas : un `metadata.yaml` par session dans `data/raw/<session_id>/`. Vérifie qu'au moins un fichier contient `source_video:` avec un chemin qui existe.

### Étape 4 — (multi-animal seulement) crop optionnel des arènes

Si tu as **N animaux par vidéo** (peu importe l'angle caméra), tu as deux voies équivalentes pour arriver aux .h5 single-animal :

- **Voie A — DLC multi-animal + split** : lance DLC directement sur la vidéo entière, puis split la sortie multi-animal en N par arène (via `assign_arenas.py`). Plus rapide au global.
- **Voie B — crop puis DLC single-animal** : découpe d'abord la vidéo en N vidéos single-animal (via `crop_arenes.py`), puis lance DLC en mode single-animal sur chacune. Sortie plus propre, indispensable si tu veux **labelliser** des frames pour améliorer le modèle.

Pour la voie B (crop) :

```cmd
:: (Une seule fois par setup) trace les N rectangles d'arène si pas déjà fait
python scripts\calibrate_arenes.py --project-dir <...>

:: Ensuite crop de toutes les sessions
python scripts\crop_arenes.py --project-dir <...> --all
```

Si tu as **1 animal par vidéo**, cette étape n'existe pas — passe directement à l'étape 5.

### Étape 5 — inférence DLC

Trois modes possibles selon le combo (nombre d'animaux, modèle DLC dispo) :

**1 animal / vidéo, modèle DLC custom** (le cas typique bottom-view) :

```cmd
conda activate dlc
python scripts\run_dlc_inference.py --project-dir <...> --all --mode custom
```

**N animaux / vidéo, DLC multi-animal SuperAnimal** (voie A, défaut sans training custom) :

```cmd
python scripts\run_dlc_inference.py --project-dir <...> --all
```

**N animaux / vidéo, single-animal sur vidéos croppées** (voie B, quand tu as croppé à l'étape 4) :

```cmd
python scripts\run_dlc_inference.py --project-dir <...> --all --mode single-animal ^
    --video-adapt --video-adapt-batch-size 2
```

Options utiles pour tous les modes :
- `--all` — traite toutes les sessions non encore traitées
- `<session_id>` en argument positionnel — cible une session précise
- `--video-adapt` sur des vidéos assez différentes du training set → adapte le modèle sur les statistiques de tes vidéos (lent mais améliore la précision)
- `--video-adapt-batch-size 2` sur GPU 16 GB (défaut 8 déborde en VRAM sur RTX 4080/5080)

Sortie : `data/dlc-output/<session>/<hash>.h5` + éventuellement `_labeled.mp4`.

### Étape 6 — préparer les fichiers pour VAME

VAME veut un h5 single-animal par session, sans NaN aggressifs, avec les mauvaises prédictions déjà masquées. Selon ce que tu as en entrée à cette étape :

**Sortie DLC custom (1 animal/vidéo ou voie B multi-animal)** — nettoyage temporel + masking + interpolation :

```cmd
python scripts\prepare_vame_input_custom.py --project-dir <...>
```

Fait pour chaque session : `dlc.filterpredictions` (median filter temporel) + masking des prédictions à likelihood < 0.3 + interpolation linéaire des trous ≤ 25 frames. Écrit `<session>_clean.h5` à côté du .h5 brut.

**Sortie DLC multi-animal (voie A)** — split par arène en amont, avant le nettoyage :

```cmd
conda activate ethoflow
python scripts\assign_arenas.py --project-dir <...> --all
```

Puis éventuellement `fill_nan_h5.py --root <project>/data/dlc-output` pour remplir les trous résiduels si VAME râle.

### Étape 7 — setup + train + segment VAME

VAME s'entraîne **une fois par projet** (le VAE apprend la structure des poses de tes souris). Compte 3-8h sur GPU pour l'entraînement.

```cmd
conda activate vame
cd D:\EthoFlow

:: 1. Init du projet VAME dans <project>/data/vame/
python scripts\run_vame.py --project-dir <...> setup

:: 2. Alignement égocentrique des poses (rotation/translation pour recentrer)
python scripts\run_vame.py --project-dir <...> align

:: 3. Construction du trainset
python scripts\run_vame.py --project-dir <...> trainset

:: 4. Entraînement du VAE (LONG — plusieurs heures)
python scripts\run_vame.py --project-dir <...> train

:: 5. Évaluation (courbes de loss, KL divergence)
python scripts\run_vame.py --project-dir <...> evaluate

:: 6. Segmentation : assigne un motif à chaque frame de chaque session
python scripts\run_vame.py --project-dir <...> segment
```

Sortie : `data/vame/results/<session>/<model>/hmm-15/15_hmm_label_<session>.npy` (1 label motif par frame).

### Étape 8 — labelliser les motifs à la main

VAME te donne 15 motifs numérotés 0-14. Il faut les nommer et les catégoriser. Deux options :

- **Générer les vidéos par motif** — 30-60 clips de 10s pour chaque motif :
  ```cmd
  python scripts\run_vame.py --project-dir <...> motif-videos
  ```
  Sortie : `data/vame/results/community_videos/motif_<N>.mp4`. Regarde chaque vidéo, décide du nom et de la catégorie ETHOGRAM (Locomotion / Sniffing / Rearing / Grooming / Stationary / Vertical exploration).

- **Remplir `data/vame/motif_labels.csv`** avec 15 lignes :
  ```csv
  motif_id;label;category;confidence;notes
  0;grooming_face;Grooming;high;
  1;walking;Locomotion;high;
  ...
  ```

Ce CSV est lu par toutes les analyses en aval. Sans lui, les figures affichent `motif_0`, `motif_1`, etc.

### Étape 9 — analyses + visualisations

```cmd
python scripts\analyze_vame.py --project-dir <...>

:: Analyses étendues (bouts, spatial, temporal quarters)
python scripts\analyze_vame.py --project-dir <...> --extended --extended-by group4
```

Sortie dans `data/vame/analysis/` :
- **CSV** : `motif_usage.csv`, `motif_usage_long.csv`, `stats_by_motif_*.csv`, `usage_by_category.csv`
- **Heatmaps groupées** : `heatmap_usage_by_condition.png`, `_by_captopril.png`, `_by_group4.png` (sessions triées par groupe, séparateurs visuels)
- **Barres + boxplots** : `mean_by_*.png`, `boxplots_top_by_*.png`, `boxplots_by_category_by_*.png`
- **Extended** : `bout_duration_by_*.png`, `thigmotaxis_by_*.png`, `temporal_by_motif_*.png`

Les stats utilisent Mann-Whitney (2 groupes) ou Kruskal-Wallis (≥3 groupes) avec correction Benjamini-Hochberg.

**Visualisations optionnelles** (parlant pour figures/posters) :

```cmd
:: GIF avec bande de motif color-codée sous la vidéo
python scripts\motif_gif.py --project-dir <...> --session BV-970 --duration 60

:: Manifold VAME style README, en pooled (référentiel commun toutes sessions)
python scripts\behavior_structure_gif.py --project-dir <...> --session BV-970 ^
    --pool-all-sessions --with-video --duration 30 --output-format mp4

:: Dendrogramme des communautés de motifs avec labels lisibles
python scripts\community_dendrogram.py --project-dir <...>
python scripts\community_dendrogram.py --project-dir <...> --group MCCiECKO
```

---

## Parcours B — entraîner un nouveau modèle DLC

**À ne faire qu'une fois par setup imaging** (nouvel angle de caméra, nouvel éclairage, nouvelles souris visuellement différentes, ou tout simplement premier setup jamais monté). Compte 1-2 semaines de travail effectif étalé (labellisation manuelle + itérations).

Les scripts vivent dans `scripts/dlc_model-training/` et sont numérotés **01 → 06** dans l'ordre d'exécution. Ils utilisent un fichier de config centralisé (`_config.py`) que tu édites une fois pour toutes. Le workflow marche pour top-view comme pour bottom-view — le seul paramètre à ajuster est `SUPERANIMAL_NAME` (`superanimal_quadruped` pour bottom-view, `superanimal_topviewmouse` pour top-view classique).

### B.0 — Recommandations qualité vidéo (avant de labelliser quoi que ce soit)

Ces recommandations viennent de l'équipe VAME/LIN (Tony) suite à un review d'une acquisition problématique. À vérifier **avant** l'acquisition finale du dataset qui servira au training.

**Exposure time** — viser ~10 ms, idéalement 5 ms ou moins. Dépendant du couple caméra + éclairage IR. À adjuster indépendamment du framerate (rester à 30 fps est OK, l'exposition contrôle le flou de mouvement).

**Netteté** — vérifier que l'image est bien focus. Un léger défocus (fréquent avec les lentilles bas prix) dégrade beaucoup les prédictions DLC, plus qu'on ne le pense en regardant l'image à l'œil nu.

**Éclairage homogène** — utiliser **plusieurs sources IR** disposées autour de l'arène, pas une seule LED en haut. Le but est que les pattes soient éclairées en permanence, y compris quand le corps de la souris bloquerait une source unique.

Un exemple de vidéo qui « fait le job » côté qualité est celui de la publication [LIN Peters et al. 2023, Neuron](https://www.sciencedirect.com/science/article/pii/S0896627323009753) — c'est la cible.

Message important : même avec une qualité vidéo moyenne, DLC peut absorber un peu de flou de mouvement **si le training dataset est bon**. Le training dataset est le levier principal, la qualité vidéo est le levier secondaire.

### B.1 — Configurer `_config.py`

Édite `scripts/dlc_model-training/_config.py` :

```python
PROJECT_NAME = "souris-bottomview"        # nom du projet DLC (arbitraire)
EXPERIMENTER = "labo"                     # ton identifiant (utilisé par DLC dans les noms de fichiers)
WORKDIR = Path(r"E:\dlc-projects")        # où créer le projet
PILOT_VIDEO = Path(r"D:\path\to\une_video_representative.mp4")
SUPERANIMAL_NAME = "superanimal_quadruped"    # ou "superanimal_topviewmouse" pour top-view
```

`PROJECT_DIR` sera calculé automatiquement à partir de `WORKDIR + PROJECT_NAME + EXPERIMENTER + date`. Tu devras mettre à jour cette variable après le setup (l'étape 01 imprime la vraie valeur).

### B.2 — Setup du projet DLC + extraction de frames

```cmd
conda activate dlc
python scripts\dlc_model-training\01_setup_project.py
```

Crée un projet DLC vierge + extrait **120 frames par k-means** de la vidéo pilote (paramètre `N_AUTO_FRAMES` dans `_config.py`). K-means garantit une couverture visuelle diversifiée — c'est le premier tiers du training set.

Sortie : `<WORKDIR>\<PROJECT_NAME>-<EXPERIMENTER>-<date>\labeled-data\<video_stem>\img*.png`.

**Mets à jour `PROJECT_DIR`** dans `_config.py` avec le vrai nom (avec la date figée par DLC).

### B.3 — Labellisation manuelle : la stratégie qui compte vraiment

C'est **la** phase qui détermine la qualité finale du modèle. Recommandations Tony/LIN pour un premier entraînement propre :

**Cible : 200-300 frames labellisées au total** pour la première passe complète. Répartition :

- **100-150 frames par k-means** (déjà extraites par 01) — couverture visuelle générale
- **50-150 frames sélectionnées manuellement** dans les vidéos disponibles, avec la stratégie ci-dessous

**Répartition des frames manuelles** :

- **15-20 %** sur des situations « faciles » : locomotion normale, pattes visibles, pas de rearing, difficulté faible à moyenne. Ces frames incluent typiquement des pattes floues — c'est normal et attendu.
- **80-85 %** sur les situations **les plus difficiles** : celles où toi-même hésites sur la position exacte du keypoint. Rearing, occlusions partielles, pattes qui sortent/rentrent sous le corps, ambiguïtés gauche/droite. **C'est là que le modèle a besoin d'exemples**, pas dans les cas triviaux qu'il apprend immédiatement.

**Stratégie de labellisation cohérente** :

- Fixe-toi une règle claire pour les cas ambigus et applique-la partout. Ex : « pour une patte floue en mouvement, je marque toujours le centre du flou » ou « pour une patte partiellement cachée, je marque la position estimée du carpe même si non visible ».
- **Capture la directionalité** : si tu labellises un rearing vers la gauche avec la patte gauche visible, prends aussi un rearing vers la droite avec la patte droite visible. Sinon le modèle apprend un biais latéral.
- Attention : une **situation difficile en temps réel se décompose en plusieurs frames difficiles distinctes** frame-par-frame. Un rearing de 2 secondes = 60 frames, dont peut-être 15 vraiment ambiguës à labelliser une par une avec la même règle.

**Lancer la GUI DLC** :

```cmd
python -c "import deeplabcut; deeplabcut.launch_dlc()"
```

Ouvre ton `config.yaml` → onglet « Label Frames ». Compte ~1 min par frame → **3-5h pour 200-300 frames** en labellisation soigneuse.

### B.4 — Premier entraînement

```cmd
python scripts\dlc_model-training\02_train.py
```

Fait le split train/test (95/5 par défaut), transfer learning depuis **SuperAnimal-Quadruped** (HRNet-w32 backbone), entraîne 50 epochs. Compte **~2-6h sur GPU 16 GB**.

Recommandation Tony : **ne pas modifier les hyperparamètres**. La tâche (12 keypoints sur souris) n'est pas assez spécifique pour justifier un tuning au-delà des défauts.

Détail technique : `NET_TYPE = "hrnet_w32"` doit matcher `MODEL_NAME = "hrnet_w32"` sinon size mismatch au chargement des poids pré-entraînés.

### B.5 — Appliquer et QC visuel

```cmd
python scripts\dlc_model-training\03_apply.py
```

Lance l'inférence sur la vidéo pilote + produit une vidéo annotée à `pcutoff=0.6`. Regarde `<PROJECT_DIR>\result-videos\<stem>\<stem>DLC*_labeled.mp4` — tu dois voir les 12 points suivre la souris correctement dans les cas normaux.

Pour voir toutes les prédictions même de basse confiance (utile pour diagnostiquer où le modèle échoue) :

```cmd
python scripts\dlc_model-training\create_labeled_video.py --pcutoff 0.3
```

Lecture clé — recommandation Tony : à `pcutoff=0.3`, un keypoint peut être **au bon endroit** même à basse confiance. La « confiance » exprime la ressemblance avec le training set : sur un modèle bien entraîné, une confiance à 30 % peut simplement signifier « la patte est floue mais c'est bien la patte, y'a rien d'autre qui lui ressemble dans l'image ». Le vrai problème apparaît quand **plusieurs zones de l'image ont une ressemblance similaire** : le modèle switche entre elles → jitter, télé-portations gauche/droite. C'est ce switch qui trahit un training set incomplet.

### B.6 — Itérations d'amélioration (extraction d'outliers manuelle)

Après avoir regardé la vidéo annotée à `pcutoff=0.3` :

Recommandation Tony : **extraire les frames à re-labelliser MANUELLEMENT**, pas via l'auto-detect de DLC. Le vrai gain vient du fait que tu vois exactement où le modèle échoue et que tu choisis les 50-100 frames les plus informatives par situation problématique.

Workflow d'itération :

1. **Regarde toutes tes vidéos analysées** en priorité (pas seulement la pilote). Identifie les patterns d'échec : « les pattes ratent quand elle grimpe le long du mur », « L/R switch pendant les demi-tours rapides », « rearing avec deux pattes cachées mal résolu ».
2. **Pour chaque pattern d'échec**, décide de combien de frames tu vas dédier. Tony suggère **50-100 nouvelles frames par situation problématique** (au-delà, rendement décroissant).
3. **Sélectionne les frames à la main** dans les vidéos, extrais-les via DLC GUI (« Extract Frames » → « Manual »).
4. **Labellise-les** avec la même règle cohérente qu'à B.3.
5. **Relance l'entraînement** (`02_train.py`) — **important : depuis le snapshot précédent, pas from scratch**. DLC le fait par défaut si tu ne changes pas d'`iteration` dans la config.

Le script `05_refine_outliers.py` reste dispo comme béquille pour attraper les cas évidents (frames à basse likelihood avec `OUTLIER_ALGORITHM = "uncertain"`), mais **le vrai levier c'est la passe manuelle**. Utilise-le comme premier passage rapide, puis fais ta propre sélection en complément.

### B.7 — Étendre à plusieurs souris (couverture inter-individu)

Recommandation Tony : « prendre autant d'animaux différents que possible » dans le training set. L'objectif est de présenter au réseau la plus large variété de situations possibles.

Édite `ADDITIONAL_VIDEOS` dans `_config.py`, puis :

```cmd
python scripts\dlc_model-training\04_add_videos.py
```

Extrait 20 frames par nouvelle vidéo (k-means). Labellise dans la GUI, relance `02_train.py`. En viser 6-10 souris différentes dans le training set final si tu as ~40 animaux à analyser.

### B.8 — Vérifier les inversions gauche/droite

```cmd
python scripts\dlc_model-training\06_check_labels.py
```

Audit géométrique qui détecte les frames où left/right paws ont probablement été inversées par erreur pendant la labellisation manuelle. Utile après plusieurs rounds.

### B.9 — Enregistrer le modèle final

Une fois satisfait de la précision, ton modèle DLC est à `<PROJECT_DIR>\config.yaml`. Depuis là, **tu es à l'Étape 1 du parcours principal** : crée un projet EthoFlow avec `create_project.py --dlc-config <ce chemin>` et enchaîne les étapes 2 à 9.

Le même modèle DLC peut être pointé par plusieurs projets EthoFlow (batches différents, mois différents, etc.).

---

## Structure d'un projet

```
D:\ethoflow\projects\<nom_projet>\
├── configs\
│   └── pipeline_config.yaml       # pointeur DLC, coords arènes (topview)
│
└── data\
    ├── raw\
    │   ├── BV-970\
    │   │   └── metadata.yaml       # produit par sync_from_excel_*
    │   └── BV-971\...
    │
    ├── cropped\                    # (topview seulement) 4 vidéos single-animal
    │   └── <session>\<session>_A{1..4}.mp4
    │
    ├── dlc-output\
    │   ├── BV-970\
    │   │   ├── BV-970DLC_...h5             # sortie brute DLC
    │   │   ├── BV-970DLC_..._labeled.mp4   # vidéo annotée (QC)
    │   │   └── BV-970_clean.h5             # h5 nettoyé (prêt pour VAME)
    │   └── ...
    │
    ├── vame\                       # projet VAME (setup + entraînement)
    │   ├── config.yaml
    │   ├── model\
    │   ├── results\
    │   │   ├── <session>\<model>\hmm-15\   # motifs par frame
    │   │   └── community_videos\           # clips par motif
    │   ├── analysis\               # sorties d'analyze_vame.py
    │   ├── behavior_structure\     # sorties de behavior_structure_gif.py
    │   └── motif_labels.csv        # labels manuels des motifs
    │
    └── results\                    # exports figures/CSV finaux
```

`configs/pipeline_config.yaml` typique :

```yaml
# Bottomview
dlc_project_config: E:\dlc-projects\souris-bottomview-labo-2026-06-05\config.yaml

# Topview (en plus)
default_arenes_coords:
  A1: [599, 40, 495, 465]
  A2: [599, 506, 496, 503]
  A3: [106, 501, 490, 505]
  A4: [110, 39, 486, 460]
```

---

## Index des scripts

**Setup projet**
- `create_project.py` — Init un nouveau projet EthoFlow (dossiers vides + pipeline_config)

**Sync depuis Excel**
- `sync_from_excel.py` (schéma multi-animal / vidéo) / `sync_from_excel_bottomview.py` (schéma 1 animal / vidéo) — Excel maître → 1 metadata.yaml par session
- `patch_captopril.py` — Backfill le champ captopril sans re-syncer (schéma 1 animal / vidéo)

**Multi-animal par vidéo — préparation**
- `calibrate_arenes.py` — GUI pour tracer les rectangles d'arène
- `crop_arenes.py` — Split vidéo brute en N vidéos single-animal (voie B)
- `assign_arenas.py` — Split .h5 DLC multi-animal en N .h5 single-animal par frame (voie A)

**DLC training** (`scripts/dlc_model-training/`) — top-view ou bottom-view, contrôlé via `_config.py`
- `_config.py` — Config centralisée (à éditer une fois par projet DLC)
- `01_setup_project.py` → `06_check_labels.py` — Workflow d'entraînement (voir [Parcours B](#parcours-b--entraîner-un-nouveau-modèle-dlc))
- `create_labeled_video.py` — Régénère la vidéo annotée à un pcutoff différent

**DLC inférence**
- `run_dlc_inference.py` — Inférence DLC (SuperAnimal ou custom)

**DLC → VAME prep**
- `prepare_vame_input_custom.py` — filterpredictions + mask + interp → `<session>_clean.h5` (pour tout .h5 single-animal sorti par modèle custom)
- `filter_keypoints.py` — Vire les keypoints non fiables (queue distale, etc.)
- `fill_nan_h5.py` — Impute agressivement les NaN restants
- `rekey_h5.py` — Re-clé un h5 à la convention VAME (`df_with_missing`)
- `reencode_vame_videos.py` — Re-encode H.264/yuv420p pour compat OpenCV

**VAME**
- `run_vame.py` — Orchestre setup/align/trainset/train/evaluate/segment/motif-videos/all

**Analyses**
- `analyze_vame.py` — Croise motifs avec conditions, CSV + heatmaps + boxplots + stats
- `community_dendrogram.py` — Dendrogramme labellisé des motifs
- `inspect_session.py` — QC par session (couverture, gaps)
- `inspect_vame_project.py` — QC d'un projet VAME (.nc files)
- `prepare_dlc_feedback_kit.py` — Kit diagnostic à envoyer à une équipe partenaire

**Visualisations**
- `motif_gif.py` — GIF/MP4 vidéo + bande motif color-codée
- `behavior_structure_gif.py` — Manifold VAME style, side-by-side avec la vidéo, mode poolé

**Utilitaires transverses**
- `paths.py` — Résolution des chemins projet-aware (importé par tous les autres)
- `run_pipeline.py` — Orchestrateur DLC → VAME (utile pour du batch automatisé)
- `trim_empty_arena.py` — Tronque h5+mp4 des frames d'empty-arena en bord d'enregistrement

---

## Troubleshooting

### CUDA non détectée ou torch=cpu

Piège fréquent sur Windows : `pip install torch` récupère parfois la build CPU. Après création de l'env `dlc`, vérifie :

```cmd
conda activate dlc
python -c "import torch; print(torch.__version__)"
```

Si ça finit par `+cpu`, ou si tu as une GPU Blackwell (RTX 50xx) qu'aucune build stable ne supporte encore, installe la nightly CUDA 12.8 :

```cmd
pip uninstall torch torchvision torchaudio -y
pip install --pre torch torchvision torchaudio ^
    --index-url https://download.pytorch.org/whl/nightly/cu128
python -c "import torch; x = torch.randn(1024,1024,device='cuda'); print(torch.cuda.get_device_name(0))"
```

### VAME `motif_videos` : "Video capture could not be opened"

Sur Windows, le hardware decoder MSMF de cv2 rentre en conflit avec CUDA quand torch est chargé avant cv2. Le fix est déjà dans `run_vame.py` (env var au top du module). Si tu appelles VAME depuis un autre script, ajoute au tout début :

```python
import os
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
```

### `dlc.filterpredictions` ne produit pas de `_filtered.h5`

Vérifie que la vidéo source pointée dans metadata est bien celle qui a servi à l'inférence DLC (le .h5 doit être à côté de la vidéo). Si tu as bougé les vidéos entre inférence et prep VAME, `filterpredictions` cherche à côté de la nouvelle position et échoue silencieusement.

### `behavior_structure_gif.py --with-video` : vidéo pas trouvée

Le chemin `source_video` dans `metadata.yaml` pointe vers un fichier qui n'existe plus (drive débranché, fichier bougé). Override :

```cmd
python scripts\behavior_structure_gif.py ... --source-video "D:\autre\chemin\970.mp4"
```

### Rendu `behavior_structure_gif` bloqué en mode `--pool-all-sessions`

Sur 1M+ points UMAP fitté seul-threadé prend >30 min. Le script cape à `--pool-max-frames 300000` par défaut et UMAP tourne en parallèle. Si tu vois toujours des ralentissements, réduis `--background-max-points 30000` pour un rendu final plus rapide.

### VAME plante avec "no such file: cropped/<session>/<session>_A1.mp4"

Si tu es en multi-animal voie B, VAME veut des vidéos croppées. Lance `crop_arenes.py --all` avant `run_vame.py setup`. Si tu es en 1 animal / vidéo, VAME attend `<session>_clean.h5` dans `dlc-output/<session>/` — vérifie que `prepare_vame_input_custom.py` a bien tourné.

### Metadata avec chemins Windows sur machine Linux (ou inversement)

Les `source_video:` dans metadata.yaml sont des chemins absolus. Si tu migres un projet entre machines, patch-les avec un `find + replace` :

```powershell
Get-ChildItem -Recurse -Filter metadata.yaml | ForEach-Object {
    (Get-Content $_.FullName) -replace "E:\\data\\ancien_dossier", "D:\nouveau_chemin" | Set-Content $_.FullName
}
```

---

## Liens externes

- DeepLabCut : https://deeplabcut.github.io/DeepLabCut/
- VAME : https://github.com/LINCellularNeuroscience/VAME
- SuperAnimal Quadruped (transfer learning base) : https://deeplabcut.github.io/DeepLabCut/docs/ModelZoo.html
- Publication référence LIN (qualité vidéo cible) : https://www.sciencedirect.com/science/article/pii/S0896627323009753

---

## Licence

À définir avec le labo.
