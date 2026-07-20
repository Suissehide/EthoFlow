# EthoFlow

Pipeline d'analyse comportementale souris à partir de vidéos brutes, basé sur **DeepLabCut** (estimation de pose) et **VAME** (segmentation comportementale non-supervisée). Deux vues supportées :

- **Topview** — caméra au plafond, 4 souris dans 4 arènes physiquement séparées, 1 vidéo → 4 sessions.
- **Bottomview** — caméra sous le plancher transparent IR, 1 souris par vidéo, 1 vidéo → 1 session.

Le pipeline part d'une acquisition brute (vidéo + Excel des souris) et produit des CSV statistiques, des figures et des vidéos annotées, groupables par n'importe quelle variable expérimentale (génotype, traitement, sexe, etc.).

Ce README couvre deux parcours :

- **[Parcours A](#parcours-a--utiliser-un-modèle-dlc-existant)** — tu as déjà un modèle DLC entraîné (le cas typique en labo : le modèle a été entraîné une fois pour ton setup expérimental et sert pour tous les projets suivants).
- **[Parcours B](#parcours-b--entraîner-un-nouveau-modèle-dlc)** — tu montes un nouveau setup expérimental et il faut entraîner un modèle DLC from scratch (ou fine-tuner sur de nouvelles données).

---

## Table des matières

1. [Concepts](#concepts)
2. [Prérequis machine](#prérequis-machine)
3. [Installation](#installation-first-time)
4. [Environnements conda](#environnements-conda-cheat-sheet)
5. [Parcours A — modèle DLC existant](#parcours-a--utiliser-un-modèle-dlc-existant)
6. [Parcours B — nouveau modèle DLC](#parcours-b--entraîner-un-nouveau-modèle-dlc)
7. [Structure d'un projet](#structure-dun-projet)
8. [Index des scripts](#index-des-scripts)
9. [Troubleshooting](#troubleshooting)

---

## Concepts

**Un projet EthoFlow** = un dossier autonome qui contient les données brutes, les sorties DLC/VAME et la config, pour une expérience donnée. Chaque projet vit à un chemin absolu (ex : `D:\ethoflow\projects\bottomview-MCC-2026-06`) et tous les scripts prennent `--project-dir <chemin>`. Tu peux avoir autant de projets en parallèle que tu veux — ils sont indépendants.

**Une session** = une acquisition (une vidéo + les metadata associées : ID souris, groupe, traitement, date, etc.). Sur bottomview, 1 vidéo = 1 session. Sur topview, 1 vidéo = 4 sessions (une par arène).

**Un modèle DLC** = un réseau pré-entraîné qui détecte les points anatomiques (nez, oreilles, pattes, queue, etc.) sur chaque frame. Un modèle DLC vit hors du projet EthoFlow (dans `E:\LEO\dlc-projects\souris-bottomview-...` par exemple) et est **réutilisé** entre projets EthoFlow.

**Un modèle VAME** = un VAE entraîné à segmenter les séquences de pose en motifs comportementaux. Contrairement à DLC, VAME s'entraîne **une fois par projet** (parce que sa segmentation dépend des animaux dans le projet).

**Modèle existant vs nouveau modèle** : la partie coûteuse est l'entraînement DLC (labellisation manuelle de plusieurs centaines de frames + jour de calcul GPU). Une fois qu'un modèle DLC est bien entraîné pour ton setup imaging, tu le réutilises pour tous les projets futurs — tu es en Parcours A. Le Parcours B ne se refait que quand tu changes de setup (nouvelle caméra, nouvel angle, nouvelles souris très différentes visuellement).

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

## Environnements conda — cheat sheet

| Env | À quoi ça sert | Requis pour |
|---|---|---|
| `ethoflow` (env-pipeline) | Utilitaires légers (pandas, PyYAML, ffmpeg, streamlit, openpyxl) | Sync depuis Excel, orchestrateur `run_pipeline.py`, app Streamlit |
| `dlc` (env-dlc) | DeepLabCut 3.x + PyTorch | Inférence DLC, entraînement DLC, GUI de labellisation |
| `vame` (env-vame) | VAME + scipy/matplotlib/UMAP | Setup/train/segment VAME, analyses, visualisations |

**L'env `ethoflow` est-il indispensable ?** Oui pour deux usages :
- **Application Streamlit** (`streamlit run streamlit_app/app.py`) — c'est l'unique env qui contient streamlit.
- **`run_pipeline.py`** — l'orchestrateur enchaîne DLC → VAME en appelant `conda run -n <env>` entre les étapes ; il tourne lui-même depuis `ethoflow`.

Pour un usage purement CLI script-par-script (ce que fait ce README), tu peux techniquement t'en passer : `sync_from_excel*.py` marche depuis l'env `vame` puisqu'il n'utilise que pandas/pyyaml/openpyxl. Mais garder les 3 envs séparés reste la manière propre de bosser (pas de risque de collision de dépendances).

---

## Parcours A — utiliser un modèle DLC existant

**Contexte** : ton labo (ou toi précédemment) a déjà entraîné un modèle DLC pour ton setup imaging. Tu as un dossier de projet DLC quelque part (ex : `E:\LEO\dlc-projects\souris-bottomview-Leo-2026-06-05\`) qui contient un `config.yaml`. Tu veux traiter un nouveau batch d'acquisitions.

Ce parcours prend 30 min à 2h par session selon la longueur des vidéos.

### A.1 — Créer un nouveau projet EthoFlow

```cmd
conda activate ethoflow
python scripts\create_project.py ^
    --project-dir D:\ethoflow\projects\bottomview-MCC-2026-06 ^
    --kind bottomview ^
    --dlc-config "E:\LEO\dlc-projects\souris-bottomview-Leo-2026-06-05\config.yaml"
```

Options :
- `--kind bottomview` — 1 souris par vidéo, pas d'arena splitting
- `--kind topview` — 4 souris par vidéo, split arènes

Résultat : arborescence vide + `configs/pipeline_config.yaml` qui pointe vers ton config DLC.

### A.2 — Préparer ton Excel de sessions

Le pipeline lit un Excel maître qui décrit tes souris. Le schéma diffère topview / bottomview :

**Bottomview** (feuille `Sessions`) — 1 ligne par souris :

| mouse_id | sex | group | cage | tail_label | birth_date | animal_id | line | genotype_mcc | captopril |
|---|---|---|---|---|---|---|---|---|---|
| 970 | F | MCCf/f | CD329 | 1 | 2024-10-15 | 54310 | MCC\*Cdh5-cre | fl/fl | oui |
| 971 | F | MCCiECKO | CD330 | 2 | 2024-10-15 | 54311 | MCC\*Cdh5-cre | fl/fl | oui |

`mouse_id` = nom du fichier vidéo attendu (`970.mp4`, `971.mp4`). `group` = ta variable de comparaison principale.

**Topview** (feuilles `Trials_Videos` + `Subjects` + `Arena_Mapping`) — voir `configs/metadata_template.yaml` pour un exemple.

### A.3 — Sync des sessions depuis l'Excel

**Bottomview** :

```cmd
python scripts\sync_from_excel_bottomview.py ^
    --project-dir D:\ethoflow\projects\bottomview-MCC-2026-06 ^
    --excel D:\ethoflow\projects\bottomview-MCC-2026-06\bottomview_sessions.xlsx ^
    --videos-dir E:\data\bottom_view\08062026 ^
    --date 2026-06-08
```

Répète la commande pour chaque batch d'acquisition (`--videos-dir` change, l'Excel reste le même). Utilise `--overwrite` pour re-générer sur une metadata déjà existante.

**Topview** :

```cmd
python scripts\sync_from_excel.py ^
    --project-dir D:\ethoflow\projects\openfield-M1-2025-10 ^
    --excel D:\path\to\OpenField_trials.xlsx
```

Résultat dans les deux cas : un `metadata.yaml` par session dans `data/raw/<session_id>/`. Vérifie qu'au moins un fichier contient `source_video:` avec un chemin qui existe.

### A.4 — Inférence DLC

Sur bottomview (modèle custom déjà pointé dans pipeline_config) :

```cmd
conda activate dlc
python scripts\run_dlc_inference.py --project-dir D:\ethoflow\projects\bottomview-MCC-2026-06 --all --mode custom
```

Sur topview (modèle SuperAnimal multi-animal par défaut) :

```cmd
python scripts\run_dlc_inference.py --project-dir D:\ethoflow\projects\openfield-M1-2025-10 --all
```

Options utiles :
- `--all` — traite toutes les sessions non encore traitées
- `<session_id>` en argument positionnel — cible une session précise
- `--video-adapt` sur des vidéos assez différentes du training set → adapte le modèle sur les statistiques de tes vidéos (lent mais améliore la précision)
- `--video-adapt-batch-size 2` sur GPU 16 GB (défaut 8 déborde en VRAM sur RTX 4080/5080)

Sortie : `data/dlc-output/<session>/<hash>.h5` + éventuellement `_labeled.mp4`.

### A.5 — Préparer les fichiers pour VAME

VAME veut un h5 single-animal par session, sans NaN aggressifs, avec les mauvaises prédictions déjà masquées.

**Bottomview** — pipeline complet en une commande :

```cmd
python scripts\prepare_vame_input_custom.py ^
    --project-dir D:\ethoflow\projects\bottomview-MCC-2026-06
```

Ça fait pour chaque session : `dlc.filterpredictions` (median filter temporel) + masking des prédictions à likelihood < 0.3 + interpolation linéaire des trous ≤ 25 frames. Écrit `<session>_clean.h5` à côté du .h5 brut.

**Topview** — étape supplémentaire de split par arène :

```cmd
conda activate ethoflow
python scripts\assign_arenas.py --project-dir <...> --all
```

Puis éventuellement `fill_nan_h5.py` pour remplir les trous résiduels si VAME râle.

### A.6 — Setup + train + segment VAME

VAME s'entraîne **une fois par projet** (le VAE apprend la structure des poses de tes souris). Compte 3-8h sur GPU pour l'entraînement.

```cmd
conda activate vame
cd D:\EthoFlow

:: 1. Init du projet VAME dans <project>/data/vame/
python scripts\run_vame.py --project-dir D:\ethoflow\projects\bottomview-MCC-2026-06 setup

:: 2. Alignement égocentrique des poses (rotation/translation pour recentrer la souris)
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

### A.7 — Labelliser les motifs à la main

VAME te donne 15 motifs numérotés 0-14. Il faut les nommer et les catégoriser. Deux options :

- **Générer les vidéos par motif** — 30-60 clips de 10s pour chaque motif, tirés des sessions :
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

### A.8 — Analyses statistiques

```cmd
python scripts\analyze_vame.py --project-dir D:\ethoflow\projects\bottomview-MCC-2026-06

:: Analyses étendues (transitions, bouts, spatial, temporal quarters)
python scripts\analyze_vame.py --project-dir <...> --extended --extended-by group4
```

Sortie dans `data/vame/analysis/` :
- **CSV** : `motif_usage.csv`, `motif_usage_long.csv`, `stats_by_motif_*.csv`, `usage_by_category.csv`
- **Heatmaps groupées** : `heatmap_usage_by_condition.png`, `_by_captopril.png`, `_by_group4.png` (sessions triées par groupe, séparateurs visuels)
- **Barres + boxplots** : `mean_by_*.png`, `boxplots_top_by_*.png`, `boxplots_by_category_by_*.png`
- **Extended** : `bout_duration_by_*.png`, `thigmotaxis_by_*.png`, `temporal_by_motif_*.png`

Les stats utilisent Mann-Whitney (2 groupes) ou Kruskal-Wallis (≥3 groupes) avec correction Benjamini-Hochberg.

### A.9 — Visualisations (optionnel mais parlant pour figures/posters)

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

**Contexte** : tu changes de setup imaging (nouvel angle, nouvelle caméra, nouvelles souris visuellement différentes) et le modèle DLC actuel ne marche plus. Ou tu démarres depuis zéro.

Compte 1-2 semaines de travail réparti : labellisation manuelle (~1 jour effectif), entraînement (~1 nuit), itérations d'amélioration (~3-5 rounds étalés sur plusieurs jours).

Les scripts de ce parcours vivent dans `scripts/dlc_bottomview/` et sont **numérotés 01 → 06** dans l'ordre d'exécution. Ils utilisent un fichier de config centralisé (`_config.py`) que tu édites une fois pour toutes.

### B.1 — Configurer `_config.py`

Édite `scripts/dlc_bottomview/_config.py` :

```python
PROJECT_NAME = "souris-bottomview"        # nom du projet DLC (arbitraire)
EXPERIMENTER = "Leo"                       # ton prénom
WORKDIR = Path(r"E:\LEO\dlc-projects")     # où créer le projet
PILOT_VIDEO = Path(r"D:\path\to\une_video_representative.mp4")
```

`PROJECT_DIR` sera calculé automatiquement à partir de `WORKDIR + PROJECT_NAME + EXPERIMENTER + date`. Tu devras mettre à jour cette variable après le setup (l'étape 01 imprime la vraie valeur).

### B.2 — Setup du projet DLC + extraction de frames

```cmd
conda activate dlc
python scripts\dlc_bottomview\01_setup_project.py
```

Crée un projet DLC vierge + extrait 60 frames de la vidéo pilote via k-means (frames visuellement diverses). Sortie : `<WORKDIR>\<PROJECT_NAME>-<EXPERIMENTER>-<date>\labeled-data\<video_stem>\img*.png`.

**Mets à jour `PROJECT_DIR`** dans `_config.py` avec le vrai nom (avec la date).

### B.3 — Labellisation manuelle (GUI)

```cmd
python -c "import deeplabcut; deeplabcut.launch_dlc()"
```

Ouvre la GUI DLC → charge ton `config.yaml` → onglet "Label Frames". Pour chaque frame extraite, place les 12 keypoints définis dans `_config.py` (nose, ears, front paws L/R, hind paws L/R, tail base/mid/tip, center, left flank). Compte ~1 min par frame → **~1h pour 60 frames**.

Points d'attention :
- Sois cohérent avec toi-même sur la définition (« left front paw » = celle qui touche le sol si visible)
- Marque les points **non visibles** en cliquant droit → « invisible »
- Sauve régulièrement

### B.4 — Entraînement

```cmd
python scripts\dlc_bottomview\02_train.py
```

Fait le split train/test (95/5 par défaut), transfer learning depuis **SuperAnimal-Quadruped** (HRNet-w32 backbone), entraîne 50 epochs. Compte **~2-6h sur GPU 16GB**.

Détail important : `NET_TYPE = "hrnet_w32"` doit matcher `MODEL_NAME = "hrnet_w32"` sinon size mismatch au chargement des poids pré-entraînés.

### B.5 — Appliquer et QC visuel

```cmd
python scripts\dlc_bottomview\03_apply.py
```

Lance l'inférence sur la vidéo pilote + produit une vidéo annotée à `pcutoff=0.6`. Regarde `<PROJECT_DIR>\result-videos\<stem>\<stem>DLC*_labeled.mp4` — tu dois voir les 12 points suivre la souris correctement.

Pour voir toutes les prédictions même de basse confiance (utile pour diagnostiquer où le modèle échoue) :

```cmd
python scripts\dlc_bottomview\create_labeled_video.py --pcutoff 0.3
```

### B.6 — Itérations pour améliorer la précision

Trois scripts d'itération à lancer dans l'ordre selon les besoins :

**B.6.a — Ajouter des vidéos d'autres souris** (couvre la variance inter-individu) :

Édite `ADDITIONAL_VIDEOS` dans `_config.py`, puis :

```cmd
python scripts\dlc_bottomview\04_add_videos.py
```

Extrait 20 frames de chacune, tu les labellises dans la GUI, tu relances `02_train.py`.

**B.6.b — Extraire les outliers du modèle** (frames où le modèle échoue) :

Édite `TRAINING_VIDEOS_FOR_REFINE` dans `_config.py`, puis :

```cmd
python scripts\dlc_bottomview\05_refine_outliers.py
```

Utilise `OUTLIER_ALGORITHM = "jump"` (attrape les frames avec sauts inter-frame anormaux) ou `"uncertain"` (frames à low likelihood, idéal pour cibler les pattes occultées à leur émergence). Extrait 30 frames par vidéo. Re-labellise dans la GUI, relance `02_train.py`.

**B.6.c — Vérifier les inversions gauche/droite** :

```cmd
python scripts\dlc_bottomview\06_check_labels.py
```

Audit géométrique qui détecte les frames où left/right paws ont probablement été inversées par erreur. Utile après plusieurs rounds de labellisation manuelle.

### B.7 — Enregistrer le modèle final dans un projet EthoFlow

Une fois satisfait de la précision, ton modèle DLC est à `<PROJECT_DIR>\config.yaml`. Depuis là, **tu es en Parcours A** : crée un projet EthoFlow avec `create_project.py --dlc-config <ce chemin>` et enchaîne les étapes A.2 à A.9.

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
dlc_project_config: E:\LEO\dlc-projects\souris-bottomview-Leo-2026-06-05\config.yaml

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
- `sync_from_excel.py` (topview) / `sync_from_excel_bottomview.py` (bottomview) — Excel maître → 1 metadata.yaml par session
- `patch_captopril.py` (bottomview) — Backfill le champ captopril sans re-syncer

**Topview — préparation**
- `calibrate_arenes.py` — GUI pour tracer les 4 rectangles d'arène
- `crop_arenes.py` — Split vidéo brute en 4 vidéos single-animal
- `assign_arenas.py` — Split .h5 DLC multi-animal en 4 .h5 single-animal par frame

**DLC training bottomview** (`scripts/dlc_bottomview/`)
- `_config.py` — Config centralisée (à éditer une fois)
- `01_setup_project.py` → `06_check_labels.py` — Workflow d'entraînement (voir [Parcours B](#parcours-b--entraîner-un-nouveau-modèle-dlc))
- `create_labeled_video.py` — Régénère la vidéo annotée à un pcutoff différent

**DLC inférence**
- `run_dlc_inference.py` — Inférence DLC (SuperAnimal ou custom)

**DLC → VAME prep**
- `prepare_vame_input_custom.py` (bottomview) — filterpredictions + mask + interp → `<session>_clean.h5`
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

Sur 1M+ points UMAP fitté seul-threadé prend >30 min. Solution : le script cape à `--pool-max-frames 300000` par défaut et UMAP tourne en parallèle. Si tu vois toujours des ralentissements, réduis `--background-max-points 30000` pour un rendu final plus rapide.

### VAME plante avec "no such file: cropped/<session>/<session>_A1.mp4"

Sur topview, VAME veut des vidéos croppées. Lance `crop_arenes.py --all` avant `run_vame.py setup`. Sur bottomview, VAME attend `<session>_clean.h5` dans `dlc-output/<session>/` — vérifie que `prepare_vame_input_custom.py` a bien tourné.

### Metadata avec chemins Windows sur machine Linux (ou inversement)

Les `source_video:` dans metadata.yaml sont des chemins absolus. Si tu migres un projet entre machines, patchse-les avec un `find + replace` :

```powershell
Get-ChildItem -Recurse -Filter metadata.yaml | ForEach-Object {
    (Get-Content $_.FullName) -replace "E:\\data\\ancien_dossier", "D:\nouveau_chemin" | Set-Content $_.FullName
}
```

---

## Liens externes

- DeepLabCut : https://deeplabcut.github.io/DeepLabCut/
- VAME : https://github.com/LINCellularNeuroscience/VAME
- SuperAnimal Quadruped (transfer learning base pour bottomview) : https://deeplabcut.github.io/DeepLabCut/docs/ModelZoo.html

---

## Licence

À définir avec le labo.
