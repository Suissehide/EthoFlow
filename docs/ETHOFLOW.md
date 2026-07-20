# EthoFlow — pipeline d'analyse comportementale souris

> **Stack** : DeepLabCut (estimation de pose) + VAME (segmentation comportementale)
> **Plateforme** : Windows
> **Public** : chercheurs non techniques (utilisation) + développeur successeur (maintenance)
> **Volume cible** : ~100 vidéos / mois, 1024×1080 à 25 fps, 10–20 min, 4 arènes par vidéo

---

## 0. Comment lire cette doc

Cette doc est organisée en parcours. Lis-la dans l'ordre la première fois.

- **Partie 1** : architecture et décisions — *à lire avant tout le reste*
- **Partie 2** : préparation du poste — *à faire une seule fois par machine*
- **Partie 3** : organisation des données — *à figer dès le début, le plus tôt = le mieux*
- **Partie 4** : créer ton premier modèle DLC — *à faire une fois par setup expérimental*
- **Partie 5** : pipeline d'inférence quotidien — *à faire à chaque session*
- **Partie 6** : analyse VAME — *à faire après l'inférence DLC*
- **Partie 7** : interface web — *à construire en phase 2*
- **Partie 8** : maintenance et passation
- **Partie 9** : troubleshooting

Chaque partie commence par **« objectif »** (ce qu'on veut obtenir) et **« pré-requis »** (ce qui doit déjà être fait).

---

## 1. Architecture et décisions

### 1.1 Vue d'ensemble du pipeline

```
[Acquisition]
    │
    ▼
[Vidéo brute 1024×1080 à 25 fps, 4 animaux dans 4 arènes]
    │
    ▼
[Crop des 4 arènes]  ───►  4 vidéos single-animal
    │
    ▼
[Inférence DLC]    ───►  fichiers .h5 (coordonnées des keypoints)
    │
    ▼
[VAME]             ───►  segmentation des comportements (motifs)
    │
    ▼
[Figures + métriques]  ───►  livrables aux chercheurs
```

### 1.2 Décisions structurantes

**Une seule machine Windows** héberge le pipeline pour le moment. Le code est écrit en Python pur, sans dépendances spécifiques Windows, pour pouvoir être déménagé vers un serveur Linux le jour où le volume le justifie.

**Pas de DLC multi-animal**. Comme les 4 animaux sont dans 4 arènes physiquement séparées et que la caméra est fixe, on crop chaque arène en pré-traitement. Résultat : 4 vidéos single-animal, un seul modèle DLC à entraîner. Énorme gain de temps de labellisation et de fiabilité.

**Pas d'orchestrateur lourd** (pas de Snakemake, Nextflow, SLURM). À 100 vidéos/mois, un script Python + un dossier surveillé suffisent largement.

**Interface chercheur en deux temps** : d'abord CLI (un fichier `.bat` à double-cliquer), ensuite Streamlit web une fois que le pipeline est stable.

**La labellisation reste dans la GUI native de DeepLabCut**. On ne réinvente pas cette roue.

### 1.3 Spécifications matérielles attendues

À vérifier avant tout :

- **GPU NVIDIA** avec ≥ 8 GB VRAM (RTX 3060 12GB, 3070, 3080, 3090, 4070+ idéal)
- **RAM** ≥ 32 GB
- **SSD** ≥ 1 TB pour le travail courant
- **Stockage secondaire** (HDD ou NAS) ≥ 10 TB pour archives
- **Windows 10 ou 11** à jour

> ⚠️ Sans GPU NVIDIA correct, l'entraînement DLC prend des jours et l'inférence est inutilisable. C'est non négociable.

---

## 2. Préparation du poste de travail

**Objectif** : un poste prêt à exécuter DLC et VAME, isolé proprement par environnements conda.

**Pré-requis** : droits administrateur Windows.

### 2.1 Vérifier la GPU

Ouvre un PowerShell et tape :

```powershell
nvidia-smi
```

Tu dois voir le nom de ta GPU et la version du driver. Sinon :

1. Va sur https://www.nvidia.com/Download/index.aspx
2. Télécharge le driver « Studio » (plus stable que le « Game Ready ») correspondant à ta GPU
3. Installe, redémarre, retape `nvidia-smi`

> Pas besoin d'installer CUDA Toolkit séparément : conda installera la bonne version dans chaque environnement.

### 2.2 Installer Miniconda

1. Télécharger : https://docs.conda.io/projects/miniconda/en/latest/
2. Choisir « Miniconda3 Windows 64-bit »
3. À l'installation, **cocher** « Add Miniconda3 to my PATH environment variable » (oui, c'est marqué « not recommended » mais ça nous simplifie la vie ensuite)
4. Ouvre un nouveau PowerShell et vérifie :

```powershell
conda --version
```

### 2.3 Installer Git

1. Télécharger : https://git-scm.com/download/win
2. Installation par défaut, sauf : choisir « Git from the command line and also from 3rd-party software »
3. Vérifier dans PowerShell :

```powershell
git --version
```

### 2.4 Cloner le repo du projet

À adapter selon ton hébergement (GitLab institut, GitHub, etc.) :

```powershell
cd C:\
mkdir labo
cd labo
git clone <URL-DU-REPO> ethoflow
cd ethoflow
```

L'arborescence cible du repo est décrite en partie 3.

### 2.5 Créer les environnements conda

On utilise **deux environnements isolés** : un pour DLC, un pour VAME. Ils ont des dépendances incompatibles, ne tente surtout pas de les fusionner.

#### 2.5.1 Environnement DeepLabCut

```powershell
conda create -n dlc python=3.10 -y
conda activate dlc
pip install "deeplabcut[gui]"
```

Test :

```powershell
python -c "import deeplabcut; print(deeplabcut.__version__)"
```

Test GPU (très important) :

```powershell
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

Si `torch.cuda.is_available()` renvoie `False`, voir partie 9 (troubleshooting).

#### 2.5.2 Environnement VAME

```powershell
conda deactivate
conda create -n vame python=3.10 -y
conda activate vame
pip install vame-py
```

> Référence officielle : https://github.com/LINCellularNeuroscience/VAME — vérifier la procédure d'installation à jour, elle évolue.

Test :

```powershell
python -c "import vame; print(vame.__version__)"
```

#### 2.5.3 Environnement utilitaires

Pour les scripts du pipeline (crop vidéo, organisation fichiers, interface web) :

```powershell
conda deactivate
conda create -n pipeline python=3.10 -y
conda activate pipeline
pip install opencv-python pandas numpy streamlit pyyaml tqdm openpyxl
```

### 2.6 Récap des environnements

| Env | Quand l'activer | Contenu |
|---|---|---|
| `dlc` | Labellisation, entraînement, inférence DLC | DeepLabCut + PyTorch GPU |
| `vame` | Analyse comportementale post-DLC | VAME + dépendances |
| `pipeline` | Scripts utilitaires, crop, interface web | OpenCV, Streamlit, etc. |

---

## 3. Organisation des données et conventions

**Objectif** : une structure de fichiers et une convention de nommage figées, valables pour les 5 prochaines années.

> 🔑 **C'est la partie la plus importante de toute la doc.** Une convention bâclée maintenant = enfer permanent ensuite.

### 3.1 Arborescence cible

```
C:\labo\
├── ethoflow\                 ← repo Git (code + cette doc)
│   ├── scripts\
│   ├── streamlit_app\
│   ├── configs\
│   ├── docs\
│   ├── README.md
│   └── data\                 ← données de travail du pipeline
│       ├── raw\              ← un dossier par session, contient metadata.yaml
│       │   └── OF-M1-20251010-V01\
│       │       └── metadata.yaml
│       ├── cropped\          ← vidéos après crop des arènes (éphémère)
│       ├── dlc-output\       ← fichiers .h5 / .csv issus de DLC
│       ├── vame-output\      ← résultats VAME
│       └── results\          ← figures et métriques finales
│
├── data\                     ← acquisition brute (vidéos + Excel maître)
│   ├── OpenField_trials_*.xlsx
│   ├── OF-M1-20251010-V01.mp4
│   └── ...
│
├── dlc-projects\             ← projets DLC (modèles entraînés)
│   └── souris-openfield-2026-XX-XX\
│
└── vame-projects\
    └── souris-openfield-2026-XX-XX\
```

> Note : les vidéos brutes restent à plat dans `data/`. C'est le format dans lequel les chercheurs les déposent et c'est très bien comme ça — `ethoflow/data/raw/<session>/metadata.yaml` référence chaque vidéo via son chemin absolu (clé `source_video`).

### 3.2 Convention de nommage des sessions (Open Field)

Convention adoptée avec le labo, documentée dans le Codebook de l'Excel maître :

| Niveau | Format | Exemple |
|---|---|---|
| **TrialCode** | `OF-<Timepoint>-<YYYYMMDD>-V<##>` | `OF-M1-20251010-V01` |
| **ArenaCode** | `<TrialCode>_A<#>` | `OF-M1-20251010-V01_A3` |
| **MouseTrialCode** | `<ArenaCode>_M<##>` | `OF-M1-20251010-V01_A3_M17` |

**Règles** : zero-padding sur tous les compteurs (V01 et non V1), date sans séparateur, dashes entre composants, underscore pour la hiérarchie. Le tri alphabétique = tri chronologique + numérique.

Le `TrialCode` sert de `session_id` dans EthoFlow. Les vidéos croppées prennent le format `<TrialCode>_<ArenaCode>.mp4`, par exemple `OF-M1-20251010-V01_A3.mp4`.

### 3.3 Excel maître = source de vérité

Les chercheurs maintiennent un fichier `OpenField_trials_<équipe>.xlsx` qui contient :

- onglet **Codebook** : documentation du format
- onglet **Subjects** : table des souris (MouseID, groupe baseline M1, groupe ANGII M2, stress)
- onglet **Trials_Videos** : une ligne par vidéo (TrialCode, date, fps, dimensions, notes)
- onglet **Arena_Mapping** : une ligne par (TrialCode, Arène) avec le MouseID assigné

EthoFlow lit ce fichier et **ne le modifie jamais**. Le script `scripts/sync_from_excel_multi.py` génère un `metadata.yaml` par session à partir de cet Excel.

```bash
conda activate ethoflow
python scripts/sync_from_excel_multi.py
# ou via l'interface Streamlit, page "Sync depuis Excel"
```

### 3.4 Schéma de métadonnée généré

Chaque session a un `data/raw/<session_id>/metadata.yaml` produit par le sync :

```yaml
session_id: OF-M1-20251010-V01
project: OF
timepoint: M1                # M1 ou M2
date: '2025-10-10'
trial_no: 1
source_video: /chemin/absolu/vers/OF-M1-20251010-V01.mp4

camera:
  fps: 25
  width: 1024
  height: 1080

arenes:
  - id: A1
    coords: null               # à définir : [x, y, w, h] dans la vidéo source
    mouse_trial_code: OF-M1-20251010-V01_A1_M15
    mouse_id: 15
    condition: SHAM            # ou CUS, SHAM+ANGII, CUS+ANGII
    angii: false
    stress: false
  - id: A2
    coords: null
    mouse_trial_code: OF-M1-20251010-V01_A2_M16
    mouse_id: 16
    condition: SHAM
    angii: false
    stress: false
  # ... A3, A4

notes: 'MANIP 1'
```

Champs dérivés automatiquement par le sync :
- `condition`, `angii`, `stress` → joints depuis la table Subjects via le MouseID + timepoint
- `coords` → reste à `null` jusqu'à ce qu'on calibre la géométrie de la grille des 4 arènes (étape manuelle, une seule fois si la caméra ne bouge pas)
- arènes vides (MouseID absent) → `mouse_id: null`, le pipeline les saute

### 3.5 Coordonnées des arènes

Les vidéos sont en 1024×1080 avec une grille de 4 arènes. Une fois la géométrie déterminée (typiquement une grille 2×2 où chaque arène fait ~512×540), enregistrer les coords dans :

- soit directement dans chaque `metadata.yaml` (si elles peuvent varier session par session)
- soit dans `configs/pipeline_config.yaml` sous une clé `default_arenes_coords` qui pourra servir de fallback

Si la caméra est strictement fixe, les coords sont identiques pour toutes les sessions et ce sont des constantes du `configs/`.

### 3.6 Sauvegarde et archivage

- `data/` (acquisitions brutes + Excel) : **immuable**, à archiver sur un stockage redondé
- `ethoflow/data/dlc-output/`, `ethoflow/data/vame-output/` : sauvegarder, ce sont des données dérivées coûteuses à recalculer
- `ethoflow/data/cropped/` : éphémère, recalculable à partir de la source — pas besoin de sauvegarder

### 3.7 SOP enregistrement

> 💡 **Règle apprise dans la douleur sur la cohorte M1/M2 2026-05** : démarrer l'enregistrement caméra **après** placement des souris dans les arènes, jamais avant.

Sur cette cohorte, ~60 % des sessions (22/36 arènes) ont eu une période initiale d'arène vide allant de 11 à 86 secondes selon l'enregistrement. C'est un problème en aval :

- DLC ne détecte rien sur ces frames → NaN dans le `.h5`
- `fill_nan_h5.py` impute en remplissant avec la médiane de la session → segments « pose constante imputée »
- VAME forme un (ou plusieurs) motif autour de ces segments artificiels, ce qui pollue les statistiques par condition
- La correction a posteriori (`--mask-empty` ou `trim_empty_arena.py`, cf. §6.6) marche mais ajoute des étapes et complique l'interprétation des cluster_videos

**Bonne pratique opérationnelle :**

1. Placer les 4 souris dans leurs arènes respectives
2. Vérifier au moniteur que les 4 sont effectivement présentes (et qu'aucune n'a grimpé sur le bord caméra)
3. **Puis** appuyer sur Record
4. Arrêter l'enregistrement avant de retirer les souris (pas pendant)

Si oubli ponctuel (1-2 sessions), la pipeline sait nettoyer ; si systématique sur une cohorte entière, on passe plusieurs heures en post.

---

## 4. Créer ton premier modèle DeepLabCut

**Objectif** : un modèle DLC entraîné qui détecte de manière fiable les keypoints d'une souris dans une arène.

**Pré-requis** : partie 2 et 3 terminées, au moins 5–10 vidéos pilotes acquises et stockées dans `data/raw/`.

> ⏱ Ce processus prend **1 à 2 semaines la première fois**, principalement à cause de la labellisation manuelle. Après c'est rapide.

### 4.1 Définir les keypoints

Avec les chercheurs, choisir 8 à 12 points anatomiques. Suggestion classique pour une souris vue du dessus :

1. nez
2. oreille gauche
3. oreille droite
4. centre du cou
5. centre du dos
6. flanc gauche
7. flanc droit
8. base de la queue
9. milieu de la queue
10. extrémité de la queue

Plus de keypoints = plus de précision pour VAME, mais aussi plus de temps de labellisation. 8–10 est un bon compromis.

### 4.2 Crop des arènes (optionnel — uniquement pour labellisation)

> ⚠️ Le pipeline d'inférence par défaut **ne crope pas** : DLC tourne en multi-animal directement sur la vidéo source, et `assign_arenas.py` splitte la sortie par arène. Cette section ne sert qu'au cas où on veut **labelliser** ou **fine-tuner** un modèle custom — la GUI DLC est plus simple en single-animal.

`scripts/crop_arenes.py` :

1. Lit `metadata.yaml` de la session (clé `source_video` + `arenes[].coords`)
2. Pour chaque arène avec `mouse_id != null` et `coords` définies, extrait le rectangle avec ffmpeg
3. Sauvegarde dans `data/cropped/<session_id>/<session_id>_<arene_id>.mp4`
   ex : `data/cropped/OF-M1-20251010-V01/OF-M1-20251010-V01_A1.mp4`

```bash
conda activate ethoflow
python scripts/crop_arenes.py OF-M1-20251010-V01
```

> Astuce : caméra fixe = mêmes coords pour toutes les sessions. Définir une fois pour toutes dans `configs/pipeline_config.yaml` sous `default_arenes_coords`, et les copier dans chaque metadata.yaml.

### 4.3 Créer le projet DLC

Activer l'env, lancer la GUI :

```powershell
conda activate dlc
python -m deeplabcut
```

Dans la GUI :

1. **Create a new project**
2. Nom du projet : `souris-openfield-AAAA-MM-JJ`
3. Ton nom comme expérimentateur
4. Sélectionner ~5 vidéos croppées variées (animaux différents, conditions différentes)
5. Working directory : `C:\labo\dlc-projects\`
6. Cocher « copy videos »

Cela crée un dossier projet avec un `config.yaml`.

### 4.4 Configurer les keypoints

Ouvrir le `config.yaml` du projet, éditer la section `bodyparts:` avec la liste choisie en 4.1.

Définir aussi le `skeleton:` (paires de keypoints reliés visuellement, par exemple `[[nez, cou], [cou, dos], [dos, queue_base]]`).

### 4.5 Extraction des frames à labelliser

Toujours dans la GUI :

1. **Extract frames** → mode `automatic`, algorithme `kmeans`
2. ~20 frames par vidéo, soit ~100 frames au total pour le premier round

### 4.6 Labellisation

1. **Label frames**
2. Cliquer chaque keypoint sur chaque frame
3. Si un keypoint est invisible (occlusion), le laisser non placé — DLC gère ça correctement

> Conseil : faire labelliser **le même chercheur** toutes les frames du premier dataset, pour la cohérence. Les variations inter-labelleurs nuisent au modèle.

### 4.7 Création du training dataset

```powershell
deeplabcut.create_training_dataset(config_path)
```

Ou via la GUI : **Create Training Dataset**.

Sélectionner un backbone : `resnet_50` est un bon défaut. `hrnet_w32` est plus précis mais plus lent.

### 4.8 Entraînement

```powershell
deeplabcut.train_network(config_path, displayiters=100, saveiters=10000, maxiters=200000)
```

Compte ~6 à 24 heures selon la GPU. À lancer la nuit.

### 4.9 Évaluation

```powershell
deeplabcut.evaluate_network(config_path, plotting=True)
```

Cela produit des images annotées et une erreur en pixels. Cible : **erreur test < 5 pixels** sur les vidéos croppées (~512×540). Si > 10 pixels, voir 4.10.

### 4.10 Refinement (raffiner si besoin)

```powershell
deeplabcut.analyze_videos(config_path, ['video1.mp4'], save_as_csv=True)
deeplabcut.extract_outlier_frames(config_path, ['video1.mp4'])
```

Cela identifie les frames mal prédites par le modèle. Les labelliser :

```powershell
deeplabcut.refine_labels(config_path)
deeplabcut.merge_datasets(config_path)
deeplabcut.create_training_dataset(config_path)
deeplabcut.train_network(config_path)
```

Une à deux passes de refinement suffisent généralement.

---

## 5. Pipeline d'inférence quotidien

**Objectif** : à partir d'une vidéo brute, produire automatiquement les trajectoires DLC par arène, prêtes pour VAME.

**Pré-requis** : partie 3 terminée (sync Excel), modèle DLC dispo (SuperAnimal par défaut, ou modèle custom entraîné en partie 4).

### 5.1 Deux workflows possibles

EthoFlow supporte deux chemins équivalents qui aboutissent au même format de sortie (`data/vame-input/<session>/<session>_A{1..4}.h5`).

**Chemin A — Multi-animal sur vidéo entière** *(le plus rapide en pratique, recommandé pour la prod)* :

```
data/<TrialCode>.mp4 + ethoflow/data/raw/<TrialCode>/metadata.yaml
        ↓
run_dlc_inference.py --mode superanimal           (env: dlc)
        ↓  data/dlc-output/<TrialCode>/<...>.h5   (multi-animal, 4 tracks fragmentés)
assign_arenas.py                                  (env: ethoflow)
   par-frame voting → 4 tracks single-animal recomposés
        ↓  data/vame-input/<TrialCode>/<TrialCode>_A1..4.h5
run_vame.py                                       (env: vame)
        ↓  data/vame-output/<TrialCode>/
```

**Chemin B — Single-animal sur vidéos pré-croppées** *(plus simple si tu veux labelliser/fine-tuner)* :

```
data/<TrialCode>.mp4 + metadata.yaml
        ↓
crop_arenes.py                                    (env: ethoflow)
   ffmpeg crop des 4 rectangles d'arène
        ↓  data/cropped/<TrialCode>/<TrialCode>_A{1..4}.mp4
run_dlc_inference.py --mode single-animal         (env: dlc)
   SuperAnimal avec max_individuals=1, pas de tracker inter-animal,
   puis flatten + nettoyage automatique
        ↓  data/vame-input/<TrialCode>/<TrialCode>_A1..4.h5
run_vame.py                                       (env: vame)
        ↓  data/vame-output/<TrialCode>/
```

### 5.2 Quel chemin choisir ?

| Critère | Chemin A (multi) | Chemin B (cropped single) |
|---|---|---|
| Pré-traitement vidéo | aucun | crop ffmpeg (~2 min/session) |
| Tracker | multi-animal SuperAnimal | single-animal (max_individuals=1) |
| Gestion des fragmentations de tracker | requise (`assign_arenas` par-frame) | non-applicable |
| Temps inférence (RTX 5080) | ~54 min/vidéo | ~5-15 min/vidéo croppée × 4 ≈ 20-60 min/session |
| Adapté à la labellisation custom | non | oui, vidéos single-animal labellisables dans la GUI DLC |
| Inspection visuelle d'une arène | vidéo annotée multi-animal complète | vidéo annotée single-animal par arène |

En pratique, A est souvent un peu plus rapide bout-à-bout, et n'altère pas les vidéos sources. B est conceptuellement plus simple et obligatoire si tu veux entraîner ton propre modèle DLC.

Pour **comparer les deux** (utile pour le pilote), utilise `--output-dir` :

```bash
python scripts/run_dlc_inference.py --mode single-animal --all \
       --output-dir data/vame-input-single
```

Tu te retrouves avec `data/vame-input/` (chemin A) et `data/vame-input-single/` (chemin B) côte à côte.

### 5.3 Pourquoi pas de crop dans le chemin A ?

Avec DLC 3.x et SuperAnimal, le détecteur multi-animal trouve les 4 souris en une seule passe sur la vidéo source. Le tracker fragmente parfois les identités (animal0 piste M17 puis M18 puis M17…), mais comme les arènes sont physiquement séparées, `assign_arenas.py` recompose des tracks propres en votant par frame : à chaque instant, on prend la souris de meilleure likelihood qui est dans le rectangle de l'arène. Pas besoin de re-encoder 4 sous-vidéos.

### 5.3 Orchestrateur

`scripts/run_pipeline.py` enchaîne les étapes en utilisant `conda run -n <env>` pour gérer le passage entre environnements :

```bash
conda activate ethoflow
python scripts/run_pipeline.py OF-M1-20251010-V01
# ou pour traiter toutes les sessions non traitées :
python scripts/run_pipeline.py --all
```

Options utiles : `--skip-vame`, `--skip-assign`, `--crop-first` (bonus pour labellisation).

### 5.3 Lancement par les chercheurs

Créer un `lancer_pipeline.bat` à la racine du repo :

```bat
@echo off
call conda activate dlc
cd C:\labo\ethoflow
python scripts\run_pipeline.py
pause
```

Le chercheur double-clique, voit la console défiler, sait que c'est fini quand le terminal s'arrête.

> En partie 7 on remplace ce `.bat` par une interface Streamlit beaucoup plus agréable.

### 5.4 DLC video-adapt et VRAM limitée

`run_dlc_inference.py --video-adapt` active le fine-tuning court de SuperAnimal sur tes propres vidéos. C'est l'étape « Starting object detector training… » dans les logs, et c'est l'étape la plus VRAM-intensive de tout le pipeline.

**Sur GPU 16 Go (RTX 5080, 4080…) le défaut DLC `video_adapt_batch_size=8` déborde et provoque soit un OOM soit (pire) un freeze silencieux par paging WDDM** (cf. §9). On expose un paramètre dédié dans `run_dlc_inference.py` :

```bash
python scripts/run_dlc_inference.py <session> --mode single-animal \
    --video-adapt --video-adapt-batch-size 2
```

`2` est notre valeur de travail confirmée sur le 5080 / 16 Go. Si tu as un GPU 24 Go (RTX 4090, A6000…), tu peux probablement monter à 4 ou 8 sans rebondir.

Deux autres dispositifs déjà câblés dans le script pour minimiser les ratés VRAM :

- **`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`** est posé automatiquement en tête du module via `os.environ.setdefault`. Ça réduit la fragmentation de l'allocateur PyTorch — il y a souvent un GB+ de mémoire réservée-mais-non-allouée gaspillée sans ce flag.
- **Côté Windows** : NVIDIA Control Panel → Manage 3D Settings → « CUDA - Sysmem Fallback Policy » → **« Prefer No Sysmem Fallback »**. Sans ça, en cas d'OOM CUDA pagine sur la RAM système et l'entraînement tourne en boucle silencieuse au lieu de remonter une vraie erreur. À activer une fois pour toutes sur la machine de calcul.

**Reprise après crash** : `is_processed()` dans `run_dlc_inference.py` (mode single-animal) se base **exclusivement** sur `data/vame-input/<session>/<session>_A*.h5` pour décider si une session est traitée. Le scratch intermédiaire `data/dlc-output/<session>/cropped-raw/` n'est plus consulté (un crash en cours de `--video-adapt` y laisse un `.h5` partiel au nom canonique qui faussait l'ancienne logique). Donc relancer `--all --mode single-animal` après un crash reprend automatiquement les sessions incomplètes.

---

## 6. Analyse VAME

**Objectif** : à partir des fichiers .h5 de DLC, segmenter le comportement de la souris en motifs récurrents.

**Pré-requis** : partie 5 terminée, fichiers DLC disponibles dans `data/dlc-output/`.

### 6.1 Création du projet VAME

```powershell
conda activate vame
python
```

```python
import vame
config = vame.init_new_project(
    project='souris-openfield-AAAA-MM-JJ',
    videos=['list_of_cropped_videos'],
    poses_estimations=['list_of_dlc_h5_files'],
    working_directory='C:/labo/vame-projects/',
    videotype='.mp4'
)
```

### 6.2 Pré-traitement et alignement

```python
vame.egocentric_alignment(config, pose_ref_index=[0, 5])  # nez, base queue
vame.create_trainset(config)
```

### 6.3 Entraînement du modèle VAME

```python
vame.train_model(config)
vame.evaluate_model(config)
```

### 6.4 Segmentation comportementale

```python
vame.pose_segmentation(config)
vame.motif_videos(config, videoType='.mp4')
vame.community(config, show_umap=True)
```

> Référence détaillée : https://github.com/LINCellularNeuroscience/VAME/blob/master/examples/demo.py

### 6.5 Encapsuler dans le pipeline

Une fois validé manuellement sur quelques sessions, encapsuler ces commandes dans `scripts/run_vame.py` qui prend une liste de sessions DLC en entrée et produit les sorties dans `data/vame-output/`.

### 6.6 Détection et exclusion des artefacts empty-arena

**Le problème.** Si l'enregistrement caméra a démarré avant la mise en place des souris (cf. §3.7), les premières secondes/minutes de chaque session sont une arène vide. DLC ne détecte rien, `fill_nan_h5.py` impute avec la médiane de session, et VAME forme un motif sur ces patterns artificiels. Résultat : un (ou plusieurs) motif où, par session, parfois c'est du « vrai grooming » et parfois c'est de l'« arène vide imputée ». Mélange empoisonnant pour les statistiques par condition.

**Détection automatique.** `analyze_vame.py` sait identifier ces frames a posteriori, à partir des `.h5` **pré-fill** (avant que `fill_nan_h5` ne masque les NaN par imputation) :

```bash
python scripts/analyze_vame.py --validity-source data/vame-input/single-enhanced-2026-05
```

Heuristique implémentée : un bloc contigu de NaN qui démarre au frame 0 (ou termine au dernier frame), de longueur ≥ `--min-edge-frames` (défaut 25 = 1 s à 25 fps), est classé comme **empty-arena**. Les blocs internes (de quelques secondes à 1-2 min) restent dans l'analyse car ils correspondent à de la **vraie immobilité** de souris que DLC perd temporairement — comportement légitime à conserver.

Output diagnostique : `validity_per_session.csv` (frames empty par session) + colonnes `empty_arena_count` / `empty_arena_fraction` ajoutées à `motif_usage_long.csv` (combien de chaque motif × session tombe dans la zone empty).

**Deux modes d'action.**

1. **Voie pragmatique — masquage des fréquences** (`--mask-empty`) :

   ```bash
   python scripts/analyze_vame.py \
       --validity-source data/vame-input/single-enhanced-2026-05 \
       --mask-empty \
       --labels data/results/motif_labels_hmm15.yaml
   ```

   Les fréquences dans les CSVs et plots sont recalculées en excluant les frames empty-arena du dénominateur. Les counts originaux restent dans la colonne `count` pour audit. **Limitation** : les `cluster_videos` générés par `motif-videos` contiennent encore des extraits empty-arena (puisque générés AVANT le masquage), ce qui rend la labellisation visuelle ambiguë pour les motifs polluants. On contourne en notant explicitement dans le YAML : `5: grooming (contaminated empty-arena V01_A2, V02_*)`.

2. **Voie propre — trim des sources puis re-segment** (`trim_empty_arena.py`) :

   ```bash
   python scripts/trim_empty_arena.py \
       --validity-csv vame-projects/<projet>/analysis/validity_per_session.csv \
       --h5-input data/vame-input/<dataset>-clean \
       --h5-output data/vame-input/<dataset>-trimmed \
       --video-input data/cropped \
       --video-output data/cropped-trimmed
   ```

   Tronque les `.h5` ET les `.mp4` du même nombre de frames (lecture/réencodage opencv frame-accurate). Les sessions non affectées sont copiées telles quelles. Compte ~60-90 min de calcul pour 22 vidéos affectées.

   Ensuite tu crées un nouveau projet VAME sur les sorties trimées :

   ```bash
   python scripts/run_vame.py setup \
       --input-dir data/vame-input/<dataset>-trimmed \
       --cropped-dir data/cropped-trimmed \
       --project-name <nom>-trimmed
   ```

   ⚠️ **Caveat important découvert sur la cohorte M1/M2 2026-05** : le retrain VAME sur des données nettoyées de leur empty-arena overfit systématiquement, quel que soit le réglage de beta/dropout/zdims (cf. §6.8). Le test_loss minimum reste similaire avec ou sans empty-arena (~1100 en valeur absolue, plus haut que les ~540 du run original — qui était gonflé artificiellement par les frames triviales). En pratique sur cette cohorte, on est resté sur la voie 1 (masquage) plutôt que la voie 2 (trim+retrain).

**Recommandation par cohorte :**

- Si la SOP §3.7 a été respectée et qu'aucune session n'est affectée → rien à faire, `analyze_vame.py` tout seul suffit.
- Si quelques sessions sont affectées (< 30 %) → voie 1 (masquage) avec annotation YAML.
- Si majorité des sessions sont affectées (cas M1/M2 2026-05) → voie 1 par défaut. Voie 2 seulement si on veut absolument des cluster_videos visuellement propres pour la labellisation, et en acceptant 2-3 h de tuning VAME en plus.

### 6.7 Labellisation des motifs

Une fois `motif-videos` exécuté, chaque motif a un fichier `motif_<i>.mp4` (montage de clips où ce motif est actif à travers les sessions), dans :

```
vame-projects/<projet>/results/<une_session>/VAME/hmm-15/cluster_videos/
```

(le contenu est identique d'une session à l'autre, n'importe laquelle convient pour la labellisation).

**Workflow :**

1. Visionner 3-5 clips par motif (~1 min/motif, total ~15-20 min pour 15 motifs)
2. Attribuer un label éthologique au format YAML libre :

   ```yaml
   # data/results/motif_labels_hmm15.yaml
   0: immobility
   1: slow locomotion
   2: rearing (unsupported)
   3: grooming face
   4: grooming body
   5: fast locomotion
   6: sniffing wall
   7: thigmotaxis
   8: ambiguous          # OK de mettre ambigu si pas tranchable
   9: artifact           # OK pour les motifs non-comportementaux
   # ...
   ```

3. Relancer `analyze_vame.py --labels <ce_yaml>` pour propager les labels dans tous les CSV et plots.

**Vocabulaire éthologique standard pour open-field souris** (à piocher selon ce qu'on voit) :

| Catégorie | Labels |
|---|---|
| **Locomotion** | `locomotion`, `slow locomotion`, `fast locomotion`, `running`, `pivoting`, `turning` |
| **Stationary** | `immobility`, `freezing`, `resting`, `crouching` |
| **Vertical exploration** | `rearing supported`, `rearing unsupported`, `stretch-attend posture` (SAP) |
| **Sniffing** | `sniffing wall`, `sniffing floor`, `sniffing air` |
| **Grooming** | `grooming face`, `grooming body`, `grooming tail`, `grooming genital`, `scratching` |
| **Arène-specific** | `thigmotaxis`, `center exploration`, `corner` |
| **Autres** | `jumping`, `digging`, `wall climbing`, `transition`, `ambiguous`, `artifact` |

**Conseils pratiques :**

- Les motifs peu utilisés (< 1 % d'usage moyen sur la heatmap) sont souvent des artefacts ou des comportements rares — `ambiguous` ou `artifact` sont des labels honnêtes.
- Il est normal d'avoir deux motifs avec le même label (`slow locomotion 1`, `slow locomotion 2`) — VAME a séparé deux nuances que le clustering distingue mais qu'on ne distingue pas visuellement.
- Pour publication : double-labelliser indépendamment par deux personnes et reporter un Cohen's kappa de fiabilité inter-juges.
- **Motifs contaminés par empty-arena** (cf. §6.6) : labelliser le contenu majoritaire et noter explicitement le caveat : `5: grooming (note: contaminated empty-arena V01_A2 / V02_*)`. C'est défendable scientifiquement si l'analyse downstream utilise `--mask-empty`.

### 6.8 Notes hyperparamètres VAME (apprises dans la douleur)

Découvertes empiriques de la cohorte M1/M2 2026-05, à connaître avant de relancer un train VAME sur une nouvelle cohorte.

**Les défauts VAME peuvent masquer un overfit massif.** Avec le config par défaut (`beta: 1, dropout: 0 partout, noise: false, kl_start: 2, annealtime: 4`), VAME présente une `test_loss` apparemment stable parce que les frames imputées par `fill_nan_h5` (poses constantes triviales à reconstruire) gonflent artificiellement à la baisse la moyenne. Sur des données nettoyées (sans ces frames), l'overfit devient visible : train loss descend, test loss remonte.

**Le minimum atteignable de test_loss dépend essentiellement de la dataset, pas des hyperparamètres.** Sur la cohorte M1/M2 nettoyée, plage 1000-1200 quel que soit le réglage (beta 1, 4, 10 ; dropout 0, 0.2, 0.4 ; zdims 30, 15). Les ~540 obtenus sur les données non nettoyées étaient une illusion. **Conclusion pratique : on ne traque pas une « bonne » valeur absolue de test_loss, on évalue le modèle qualitativement sur la cohérence des motifs en aval.**

**VAME sauvegarde son `best_model.pkl` à la fin de l'entraînement**, pas au cours des epochs. Si tu Ctrl+C en cours de route, le fichier n'existe pas et `segment` plante. Pour avoir un modèle utilisable rapidement : **set `max_epochs: 30` dans `config.yaml`**, lance, laisse finir (~30 min). À la fin, `best_model.pkl` contient automatiquement le checkpoint du minimum de test_loss (typiquement epoch 5-10 sur des données nettoyées, plus tard sur des données avec frames triviales).

**Le critère `model_convergence` est basé sur la `train_loss`, pas la `test_loss`.** Le set par défaut (50) déclenche un auto-stop quand la train_loss stagne 50 epochs — ce qui peut arriver en plein overfit (train descend toujours mais test depuis longtemps remontée). **Set `model_convergence: 200` pour neutraliser ce critère** et te reposer uniquement sur `max_epochs`.

**Si tu veux régulariser plus** (pour cluster_videos plus propres, latent mieux structuré), les leviers efficaces dans l'ordre :

1. **`dropout_encoder: 0.2-0.4`** + **`dropout_pred: 0.2-0.4`**. (Note : `dropout_rec` ne fait rien quand `n_layers: 1`, c'est un warning PyTorch silencieux dans VAME — laisse à 0 ou ignore-le.)
2. **`noise: true`** — bruit gaussien sur les inputs d'entraînement, équivalent d'une augmentation de données.
3. **`zdims: 10-15`** au lieu de 30. Force vraiment le bottleneck à compresser. Réduit aussi `hidden_size_*` à 128 si tu veux taper plus fort.
4. **`kl_start: 5-10`** + **`annealtime: 20-30`** au lieu de `kl_start: 2`, `annealtime: 4`. Donne plus de temps au modèle pour s'installer avant que la pression KL ne kicke.
5. **`beta`** : entre 1 et 10. Sur cette cohorte la valeur de beta n'a pas eu d'effet majeur, parce que la contribution KL au loss total reste petite quoi qu'il arrive (KL ~5-8, MSE ~1000-2000, donc beta×KL << MSE même à beta=10).

**Combinaison « régularisée minimale » testée comme dernière itération sur la cohorte :**

```yaml
beta: 4
dropout_encoder: 0.4
dropout_rec: 0.4    # ignoré si n_layers=1 mais inoffensif
dropout_pred: 0.4
noise: true
zdims: 15
hidden_size_layer_1: 128
hidden_size_layer_2: 128
hidden_size_rec: 128
hidden_size_pred: 128
kl_start: 5
annealtime: 25
model_convergence: 200
max_epochs: 30
```

Ça n'a pas magiquement résolu l'overfit (test_loss min ~1145 quand même), mais le modèle au minimum est plus généralisable que celui en config par défaut (latent plus régulier, moins de mémorisation par exemple). C'est une bonne base de départ pour une nouvelle cohorte.

**Bug VAME à noter** : `vame.train_model()` fait `os.mkdir(model_losses)` sans `exist_ok=True`. Au deuxième run dans le même projet, ça plante avec `FileExistsError`. Workaround : `rmdir /S /Q "<projet>/model/model_losses"` avant chaque relance.

---

## 7. Interface web (phase 2)

**Objectif** : permettre aux chercheurs de lancer et suivre le pipeline depuis un navigateur, sans toucher à la ligne de commande.

**Pré-requis** : pipeline CLI (parties 5 et 6) stable et utilisé par au moins un chercheur depuis quelques semaines. **Ne pas démarrer cette partie avant.**

### 7.1 Stack

- **Streamlit** : framework Python qui transforme un script en app web
- **Hébergement** : sur le poste de calcul, accessible aux autres machines du LAN via `http://<ip-poste>:8501`

Pas besoin de serveur web séparé, pas de JavaScript, pas de frontend à maintenir.

### 7.2 Fonctionnalités cibles (v1 minimale)

1. **Page d'accueil** : liste des sessions présentes dans `data/raw/`, statut de chacune (brut / croppé / DLC ok / VAME ok)
2. **Page « nouvelle session »** : formulaire pour créer un `metadata.yaml` (animal IDs, conditions) et uploader la vidéo
3. **Page « lancer pipeline »** : sélectionner une ou plusieurs sessions, bouton « lancer », barre de progression
4. **Page « résultats »** : visualiser les vidéos annotées DLC, les motifs VAME, télécharger les fichiers

### 7.3 Squelette à écrire

`scripts/web_app.py` :

```python
import streamlit as st
import pandas as pd
from pathlib import Path

st.set_page_config(page_title="EthoFlow", layout="wide")
st.title("EthoFlow — analyse comportementale")

# Sidebar : navigation
page = st.sidebar.radio("Page", ["Sessions", "Nouvelle session", "Lancer pipeline", "Résultats"])

if page == "Sessions":
    sessions = list(Path("C:/labo/data/raw").iterdir())
    df = pd.DataFrame([
        {"session": s.name, "statut": "à traiter"}  # à enrichir
        for s in sessions
    ])
    st.dataframe(df)

# ... etc.
```

Lancement :

```powershell
conda activate pipeline
streamlit run scripts/web_app.py
```

Pour exposer sur le LAN, ajouter dans `.streamlit/config.toml` :

```toml
[server]
address = "0.0.0.0"
port = 8501
```

### 7.4 Ce qu'on ne met **pas** dans l'interface web

- la labellisation DLC → reste dans la GUI native de DLC, ne pas tenter de la réimplémenter
- l'entraînement de modèles → reste en CLI, c'est rare et nécessite des choix scientifiques
- l'authentification utilisateur → si le poste est sur le LAN du labo, suffisant ; pas de mots de passe à gérer

---

## 8. Maintenance et passation

### 8.1 Mises à jour

- **DeepLabCut** : tester les nouvelles versions dans un env dédié `dlc-test` avant de basculer l'env de prod. Une mise à jour majeure peut casser un modèle entraîné.
- **VAME** : idem, tester avant de migrer.
- **Drivers GPU** : mettre à jour ~tous les 6 mois, tester `nvidia-smi` et le pipeline juste après.

### 8.2 Réentraînement périodique

- Si un nouveau setup expérimental est introduit (nouvelle arène, nouvel angle de caméra), refaire un projet DLC dédié.
- Si le modèle existant donne des erreurs croissantes, ajouter ~50 frames de refinement et réentraîner.

### 8.3 Pour le successeur

À ton départ, prépare :

1. Une **session de transition** d'au moins une demi-journée avec ton successeur, parcourant cette doc point par point.
2. Une **vidéo screencast** (~30 min) où tu fais tourner un cycle complet du pipeline en commentant.
3. Un **document de décisions** (`docs/DECISIONS.md`) listant les choix techniques importants et leur justification.
4. Un **carnet de bord** (`docs/JOURNAL.md`) où tu as noté les incidents rencontrés et leur résolution.

### 8.4 Hébergement de cette doc

Une fois le repo Git stable, servir cette doc avec MkDocs :

```powershell
conda activate pipeline
pip install mkdocs-material
mkdocs new .
# éditer mkdocs.yml et déplacer ce .md dans docs/
mkdocs serve
```

Ouvre http://localhost:8000 et tu as une doc navigable. Avec un peu de config, hébergeable en interne du labo.

---

## 9. Troubleshooting

### `torch.cuda.is_available()` renvoie False

1. Vérifier `nvidia-smi` dans PowerShell. Si ça ne marche pas → driver à réinstaller.
2. Dans l'env `dlc`, vérifier la version de torch : `python -c "import torch; print(torch.__version__)"`. Si pas la version `+cuXXX`, réinstaller PyTorch via `pip install torch --index-url https://download.pytorch.org/whl/cu121`.
3. Si le problème persiste, recréer l'env conda from scratch.

### Inférence DLC très lente

- Vérifier que la GPU est bien utilisée avec `nvidia-smi` pendant l'inférence (l'utilisation doit monter à >50%).
- Réduire la batch_size si OOM, augmenter si la GPU est sous-utilisée.

### VAME plante à l'installation

- Souvent lié à des conflits de versions entre PyTorch et les dépendances VAME.
- Suivre exactement les versions Python recommandées dans le repo VAME (vérifier les Issues GitHub récentes).

### Les coordonnées de crop dérivent d'une session à l'autre

- Soit le setup caméra a bougé → réajuster les coords dans le metadata
- Soit prévoir une calibration automatique (détection des bordures d'arène avec OpenCV) — projet d'amélioration

### La doc est obsolète

- Quand tu modifies le code, **modifie la doc dans le même commit Git**. Sinon elle dérive en quelques semaines et devient pire qu'inutile.

### Crash Windows BSOD `HYPERVISOR_ERROR (0x20001)` pendant l'entraînement

Observé sur Dell Tower Plus EBT2250 (Z890 chipset, Arrow Lake-S) avec RTX 5080. Pattern : crash systématique à l'étape « Starting object detector training » de `--video-adapt`. Le minidump dans `C:\Windows\Minidump\` analysé via WinDbg (`!analyze -v`) pointe vers `intelppm.sys!HvRequestIdle` + Arg1=0x28 (« internal error in the I/O MMU module »).

C'est un bug d'interaction entre le power management Intel, Hyper-V (machine Secured-core) et le VT-d (IOMMU). **Pas un problème de PSU, RAM, ou GPU.**

**Fix** : F2 au démarrage → BIOS Dell → Virtualization Support → désactiver « VT for Direct I/O » (≠ VT-x qu'on laisse activé). Sauver, redémarrer. Le composant qui faute (IOMMU) est physiquement retiré du chemin, l'hyperviseur reste actif (Memory Integrity peut rester on) et le crash disparaît.

Aucun impact sur les perfs DLC/VAME. Tu perds juste la protection DMA, sans conséquence sur un poste de calcul en labo.

### CUDA out of memory pendant `--video-adapt`

Symptômes :
- Vrai cas : exception `torch.OutOfMemoryError` claire dans la console
- Pire cas (sans config NVIDIA) : freeze silencieux avec `nvidia-smi` qui montre 100 % GPU-util mais ~80 W de puissance et 38 °C — c'est du paging WDDM, pas du calcul

**Fix systémique** : NVIDIA Control Panel → Manage 3D Settings → « CUDA - Sysmem Fallback Policy » → **« Prefer No Sysmem Fallback »**. Ça transforme le freeze silencieux en vraie erreur OOM exploitable. À configurer une fois pour toutes.

**Fix par paramètre** : `--video-adapt-batch-size 2` (au lieu du défaut DLC 8). Voir §5.4.

**Aide complémentaire** : `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (déjà câblé en tête de `run_dlc_inference.py` via `os.environ.setdefault`).

### VAME plante au 2ᵉ run de `train` avec `FileExistsError: model_losses`

Bug VAME-py 0.13.0 : `os.mkdir(model_losses)` sans `exist_ok=True`. Workaround :

```cmd
rmdir /S /Q "<projet>/model/model_losses"
python scripts/run_vame.py train
```

Faire pareil pour `<projet>/model/best_model` et `<projet>/logs` si tu veux repartir d'une session TensorBoard propre.

⚠️ **Ne supprime PAS `<projet>/data/processed/`** entre deux runs : VAME y stocke les `.nc` initialisés au `setup`, et `align` les modifie en place. Si tu supprimes processed/, il faut refaire `setup --force` pour reconstruire les fichiers initiaux.

### TensorBoard affiche les courbes de deux runs mélangées

Si tu relances `train` plusieurs fois dans le même projet sans nettoyer, les events s'accumulent dans `logs/tensorboard/VAME/` et TB les concatène visuellement (le warning « Found more than one graph event per run » apparaît à l'ouverture). Pour un monitoring propre, supprime le dossier avant chaque relance :

```cmd
rmdir /S /Q "<projet>/logs/tensorboard"
```

---

## 10. Setup futur : autres vues / modèles DLC

**Pas de SuperAnimal pour la vue de dessous** (vérifié sur modelzoo officiel mai 2026). Les SuperAnimal disponibles sont `superanimal_topviewmouse` (le tien), `superanimal_quadruped` (quadrupèdes vus de profil) et `superanimal_humanbody`. Rien pour bottom-view (plaque de verre, gait analysis).

**Stratégie pragmatique pour ajouter une vue bottom-view** : **transfer learning** depuis `superanimal_quadruped`, qui partage des features visuelles bas niveau avec ton setup et qui voit naturellement les pattes (au contraire du topview qui ne les voit pas du tout).

Code template (à adapter quand tu y arriveras) :

```python
import deeplabcut
from deeplabcut.modelzoo import build_weight_init

config_path = "<chemin>/config.yaml"  # projet DLC custom avec tes propres keypoints bottom-view

weight_init = build_weight_init(
    cfg=config_path,
    super_animal="superanimal_quadruped",
    model_name="hrnet_w32",
    detector_name="fasterrcnn_resnet50_fpn_v2",
    with_decoder=False,  # nouveau decoder pour TES keypoints
)

deeplabcut.create_training_dataset(config_path, weight_init=weight_init)

deeplabcut.train_network(
    config_path,
    superanimal_name="superanimal_quadruped",
    superanimal_transfer_learning=True,
    epochs=50,
)
```

`with_decoder=False` + `superanimal_transfer_learning=True` = on charge le backbone Quadruped et on entraîne par-dessus un nouveau décodeur sur tes keypoints custom (qui peuvent être totalement différents de ceux de Quadruped — pattes/coussinets/ventre pour bottom-view au lieu de tête/oreilles).

**Budget labellisation estimé** : 100-150 frames bien réparties (différentes postures et timings) suffisent grâce au pré-entraînement. C'est 5-10× moins que de partir from-scratch.

Ensuite, le pipeline EthoFlow existant intègre le modèle custom via `--mode custom` dans `run_dlc_inference.py` (config dans `configs/pipeline_config.yaml` → clé `dlc_project_config`).

---

## Annexes

### A. Liens utiles

- DeepLabCut : https://deeplabcut.github.io/DeepLabCut/
- VAME : https://github.com/LINCellularNeuroscience/VAME
- Streamlit : https://streamlit.io/
- MkDocs Material : https://squidfunk.github.io/mkdocs-material/

### B. Glossaire

- **Keypoint** : point anatomique annoté (nez, oreille, etc.)
- **Inférence** : appliquer un modèle entraîné sur de nouvelles données
- **Refinement** : raffiner un modèle en corrigeant manuellement les frames mal prédites
- **Motif VAME** : segment de comportement automatiquement identifié par VAME
- **Backbone** : architecture de réseau neuronal sous-jacente (ResNet, HRNet…)

### C. Checklist de mise en route

- [ ] Spécifier la machine, vérifier GPU et stockage
- [ ] Installer drivers, conda, git
- [ ] Cloner le repo, créer les 3 envs conda
- [ ] Tester `torch.cuda.is_available()` → True
- [ ] Définir avec les chercheurs : convention de nommage, schéma metadata, keypoints
- [ ] Acquérir 5–10 vidéos pilotes
- [ ] Crop manuel d'une arène pour test
- [ ] Créer projet DLC, labelliser, entraîner, évaluer
- [ ] Faire tourner l'inférence sur une session complète
- [ ] Lancer VAME sur les sorties DLC
- [ ] Encapsuler dans `run_pipeline.py` + `lancer_pipeline.bat`
- [ ] Onboarder un chercheur pilote et itérer
- [ ] Construire l'interface Streamlit
- [ ] Servir la doc avec MkDocs
- [ ] Préparer la passation

### D. Journal de cohorte M1/M2 2026-05

Chronologie pour mémoire de ce qui s'est passé sur la première vraie cohorte avec ce pipeline. À jour : juin 2026.

**Phase 1 — Stabilisation hardware (Windows / RTX 5080)**

Plusieurs jours bloqués sur des crashs BSOD `HYPERVISOR_ERROR` à chaque tentative de `--video-adapt`. Diagnostic via minidumps (`C:\Windows\Minidump\` + WinDbg `!analyze -v`) : interaction Intel power management × Hyper-V (machine Secured-core Dell Tower Plus EBT2250) × VT-d. **Fix : désactiver VT-d dans le BIOS** (cf. §9). Pas de retombée sur les perfs.

**Phase 2 — VRAM video-adapt**

Avec VT-d désactivé, plus de crash mais freeze silencieux à « Starting object detector training… » : 100 % GPU-util, 78 W / 38 °C, 15.8 / 16.3 GB VRAM. Diagnostic : oversubscription VRAM + paging WDDM transparent (Sysmem Fallback Policy par défaut « auto »). **Fix : `Prefer No Sysmem Fallback` dans NVIDIA Control Panel + `--video-adapt-batch-size 2`** (au lieu du défaut DLC 8). Câblage `expandable_segments:True` dans `run_dlc_inference.py` (commit `c5a846d`).

**Phase 3 — DLC inference complète**

`--video-adapt-batch-size 2` ouvre la voie. Tous les 36 (session × arène) passent avec QC ≥ 92 % de couverture. Quelques bugs `is_processed()` corrigés au passage pour gérer les reprises après crash (commits `d2e5492`, `604dc8d`).

**Phase 4 — Empty-arena discovery**

À l'analyse VAME, on observe sur les `motif-videos` que le motif 5 (entre autres) montre **tantôt du grooming, tantôt une arène vide** selon la session. Investigation via comparaison des `.h5` pré-fill et post-fill : 22/36 arènes ont 11 à 86 secondes d'arène vide en tout début d'enregistrement (la caméra a été démarrée avant la mise en place des souris).

Fix amont pour les cohortes futures : SOP §3.7 « démarrer enregistrement après placement ».

Fix aval (cette cohorte) : `analyze_vame.py` étendu avec :
- `--validity-source <dir>` : détecte les blocs NaN edge des `.h5` pré-fill (commit `7d6afbd`)
- `--mask-empty` : recalcule les fréquences en excluant les frames empty
- `--labels <yaml>` : propage des labels comportementaux dans CSV+plots (commit `6fdabaa`)

Plus `trim_empty_arena.py` (nouveau script, commit `4c2a2cc`) qui tronque h5+vidéo en miroir pour repartir sur un dataset vraiment propre.

**Phase 5 — Retrain VAME sur données trimées (échec, info importante)**

Tentative de retrain VAME complet sur les données trimées pour avoir des cluster_videos propres et des fréquences sans avoir besoin de `--mask-empty`. 5 itérations d'hyperparamètres (beta 1, 4, 10 ; dropout 0, 0.2, 0.4 ; zdims 30 → 15 ; hidden 256 → 128 ; KL anneal accéléré → ralenti).

**Toutes overfittent.** Pattern systématique : `train_loss` descend, `test_loss` remonte rapidement (minimum entre epoch 4 et 10 selon le réglage). **Découverte importante** : les ~540 de test_loss du run original n'étaient pas un succès, c'était une illusion gonflée par les ~20 000 frames imputées avec pose constante (triviales à reconstruire). Sur du « vrai » comportement (données trimées), le plancher de `test_loss` atteignable se situe en réalité vers 1100-1200. Voir §6.8 pour la liste complète des leviers de régularisation et leurs limites.

**Décision finale pour cette cohorte** : on est restés sur le projet original `OF-single-enhanced-2026-05` avec `--mask-empty + --labels` dans `analyze_vame.py`. Les statistiques sont correctes, les cluster_videos sont parfois ambigus (motifs polluants notés explicitement dans le YAML).

**À retenir pour la cohorte suivante :**

1. Démarrer l'enregistrement APRÈS placement des souris (§3.7) — ça élimine tout le problème en amont.
2. Si exception (1-2 sessions), la voie « masquage » `--mask-empty` suffit largement.
3. Le retrain VAME sur données nettoyées est PROBABLEMENT possible mais demande plus de tuning qu'on n'a fait — peut-être tester `superanimal_topviewmouse` + transfer learning DLC (cf. §10 et la doc DLC à jour) plutôt qu'un VAE from-scratch sur si peu de données.
