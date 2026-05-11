# EthoFlow — pipeline d'analyse comportementale souris

Pipeline d'analyse comportementale de souris basé sur **DeepLabCut** (estimation de pose) et **VAME** (segmentation comportementale), avec une interface web **Streamlit** pour faciliter l'utilisation par les chercheurs non-techniques.

## Architecture

Deux chemins équivalents, à choisir selon le contexte. Le pipeline démarre toujours par `sync_from_excel.py` qui génère les `metadata.yaml`, et se termine par VAME qui consomme `data/vame-input/`.

**Chemin A — Multi-animal (rapide, sans pré-traitement vidéo)** :

```
[sync_from_excel.py]   →  metadata.yaml par session
        ↓
[run_dlc_inference.py --mode superanimal]   →  data/dlc-output/  (1 .h5 multi-animal par vidéo)
        ↓
[assign_arenas.py]   →  data/vame-input/  (4 .h5 single-animal par session, recomposés par position)
        ↓
[run_vame.py]   →  data/vame-output/
```

**Chemin B — Single-animal (vidéos pré-croppées, plus simple pour labellisation)** :

```
[sync_from_excel.py]   →  metadata.yaml par session
        ↓
[crop_arenes.py]   →  data/cropped/  (4 vidéos single-animal par session)
        ↓
[run_dlc_inference.py --mode single-animal]   →  data/vame-input/  (4 .h5 directs)
        ↓
[run_vame.py]   →  data/vame-output/
```

Différence : le chemin A garde les vidéos brutes intactes et utilise un tracker multi-animal (avec gestion des fragmentations par `assign_arenas`) ; le chemin B re-encode chaque arène en vidéo single-animal et lance DLC en mode `max_individuals=1` (pas de tracker inter-animal, sortie plus simple). Le chemin B est aussi celui à utiliser si tu veux **labelliser ou fine-tuner un modèle DLC custom** : les vidéos croppées single-animal sont infiniment plus simples à annoter.

Les chercheurs interagissent via l'interface Streamlit ; le code tourne en local sur le poste de calcul.

## Setup sur une nouvelle machine

### 1. Pré-requis système

- **macOS / Linux / Windows** (tout fonctionne, recommandation : Linux ou macOS pour le dev, Windows acceptable en prod)
- **GPU NVIDIA** fortement recommandé pour l'entraînement DLC et l'inférence VAME (CUDA 12+). Apple Silicon (MPS) marche pour l'inférence légère et le dev.
- **Miniconda** : https://docs.conda.io/projects/miniconda/
- **Git** : https://git-scm.com
- **ffmpeg** dans le PATH (vient avec l'env conda `ethoflow`)

### 2. Cloner les repos

```bash
mkdir -p ~/Inserm && cd ~/Inserm

# le repo de pipeline
git clone <URL_ETHOFLOW> ethoflow

# (optionnel) sources DeepLabCut et VAME pour pouvoir lire le code,
# voir les exemples, etc. — pas requis pour faire tourner le pipeline.
git clone --depth 1 https://github.com/DeepLabCut/DeepLabCut.git
git clone --depth 1 https://github.com/LINCellularNeuroscience/VAME.git
```

Tu te retrouves avec :

```
~/Inserm/
├── ethoflow/        ← le repo
├── DeepLabCut/      ← source DLC (pour référence)
└── VAME/            ← source VAME (pour référence)
```

### 3. Installer les environnements conda

Trois envs séparés (DLC, VAME et le reste ont des dépendances incompatibles) :

```bash
cd ~/Inserm/ethoflow

# env utilitaires (orchestration, Streamlit, scripts) — toujours nécessaire
conda env create -f environment-pipeline.yml

# env DeepLabCut — sur la machine où on fait l'inférence/entraînement
conda env create -f environment-dlc.yml

# env VAME — sur la machine d'analyse
conda env create -f environment-vame.yml
```

Sur Apple Silicon, vérifie que MPS est dispo pour DLC :

```bash
conda activate dlc
python -c "import torch; print('MPS:', torch.backends.mps.is_available())"
```

Sur GPU NVIDIA :

```bash
conda activate dlc
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

> ⚠️ **Si `cuda.is_available()` renvoie `False` ou si la torch installée finit par `+cpu`** (piège fréquent sur Windows), ou si tu as une **GPU Blackwell** (RTX 50xx, sm_120), réinstalle PyTorch nightly avec CUDA 12.8 :
>
> ```bash
> conda activate dlc
> pip uninstall torch torchvision torchaudio -y
> pip install --pre torch torchvision torchaudio \
>     --index-url https://download.pytorch.org/whl/nightly/cu128
> ```
>
> Test final (doit afficher le nom de ta GPU) :
>
> ```bash
> python -c "import torch; x = torch.randn(1024,1024,device='cuda'); print(torch.cuda.get_device_name(0))"
> ```

### 4. Récupérer les données brutes

EthoFlow ne stocke pas les vidéos ni l'Excel maître dans le repo Git (trop lourd). Il faut les copier à part dans `~/Inserm/data/` :

```
~/Inserm/data/
├── OpenField_trials_CDUPLAA.xlsx
├── OF-M1-20251010-V01.mp4
├── OF-M1-20251010-V02.mp4
└── ...
```

Sources possibles :
- NAS du labo (chemin à demander à C. Duplaa)
- Disque dur USB
- `scp` depuis l'ancienne machine : `scp -r user@machine:~/Documents/Inserm/data ~/Inserm/`

### 5. Synchroniser les metadata depuis l'Excel

Une fois les vidéos en place :

```bash
conda activate ethoflow
cd ~/Inserm/ethoflow
python scripts/sync_from_excel.py
```

Ça génère un `metadata.yaml` par session dans `data/raw/<TrialCode>/`. Vérifie qu'au moins un fichier a `source_video: /chemin/correct/...mp4`.

### 6. (Si calibration nécessaire) Calibrer les arènes

Les coordonnées des 4 arènes sont stockées dans `configs/pipeline_config.yaml` et sont **versionnées Git**. Si tu utilises le même setup caméra qu'avant, **tu n'as rien à faire**. Sinon :

```bash
python scripts/calibrate_arenes.py --session OF-M1-20251010-V01
```

Une fenêtre s'ouvre, tu dessines les 4 rectangles à la souris (dans l'ordre A1 → A2 → A3 → A4), ENTRÉE valide, et les nouvelles coords écrasent l'ancienne calibration dans le yaml.

### 7. Tester que tout marche

```bash
streamlit run streamlit_app/app.py
```

L'interface s'ouvre sur http://localhost:8501. Tu dois voir tes 10 sessions dans le tableau de bord avec leur timepoint M1/M2 et le statut « vidéo: OK ».

## Quick start après setup

### Chemin A — Multi-animal (recommandé pour la prod)

Sortie : `data/vame-input/<session>/<session>_A{1..4}.h5`

```bash
# Inférence DLC multi-animal sur la vidéo entière (env: dlc)
conda activate dlc
python scripts/run_dlc_inference.py OF-M1-20251010-V01
# ou en batch : python scripts/run_dlc_inference.py --all

# Split par arène + nettoyage (env: ethoflow)
conda activate ethoflow
python scripts/assign_arenas.py OF-M1-20251010-V01
# ou en batch : python scripts/assign_arenas.py --all-new
```

### Chemin B — Single-animal sur vidéos croppées

Sortie : `data/vame-input/<session>/<session>_A{1..4}.h5` (même structure)

```bash
# 1. Crop des 4 arènes par vidéo (env: ethoflow, calibration déjà dans pipeline_config.yaml)
conda activate ethoflow
python scripts/crop_arenes.py --all

# 2. DLC single-animal sur les vidéos croppées (env: dlc)
conda activate dlc
python scripts/run_dlc_inference.py --mode single-animal --all
```

### Comparer les deux chemins

Pour préserver les résultats des deux approches dans des dossiers séparés :

```bash
# Garde la sortie multi-animal d'abord, puis écris le single-animal ailleurs
python scripts/run_dlc_inference.py --mode single-animal --all \
       --output-dir data/vame-input-single
```

Tu te retrouves avec :
- `data/vame-input/` — résultats du chemin A (multi-animal + assign_arenas)
- `data/vame-input-single/` — résultats du chemin B (single-animal cropped)

Tu lances VAME en pointant sur l'un ou l'autre pour comparer les motifs comportementaux.

## Structure du repo

```
ethoflow/
├── README.md                          # ce fichier
├── docs/ETHOFLOW.md                   # doc complète, conventions, troubleshooting
│
├── environment-pipeline.yml           # env conda 'ethoflow' (orchestration)
├── environment-dlc.yml                # env conda 'dlc' (DeepLabCut 3.x)
├── environment-vame.yml               # env conda 'vame' (VAME)
├── requirements-pipeline.txt          # deps Docker / pip
│
├── streamlit_app/app.py               # interface web
│
├── scripts/
│   ├── sync_from_excel.py             # Excel → metadata.yaml
│   ├── calibrate_arenes.py            # GUI : tracer les 4 ROI d'arène
│   ├── crop_arenes.py                 # crop optionnel (pour labellisation)
│   ├── run_dlc_inference.py           # SuperAnimal multi-animal ou modèle custom
│   ├── assign_arenas.py               # split DLC multi-animal → 4 .h5 par arène
│   ├── run_vame.py                    # analyse VAME (squelette)
│   └── run_pipeline.py                # orchestrateur tout-en-un
│
├── configs/
│   ├── pipeline_config.yaml           # coords arènes + chemins modèles
│   ├── pipeline_config.yaml.example   # template
│   └── metadata_template.yaml         # exemple de schéma metadata
│
├── docker/                            # Dockerfile + docker-compose pour le Streamlit
├── tests/                             # tests basiques pytest
│
└── data/                              # GITIGNORED — données locales
    ├── raw/<TrialCode>/metadata.yaml  # généré par sync_from_excel.py
    ├── cropped/                       # éphémère
    ├── dlc-output/
    ├── vame-output/
    └── results/
```

## Ce qui n'est PAS dans le repo (gitignored)

- Les vidéos brutes (~100 MB chacune × 100/mois)
- Le fichier Excel maître (source de vérité, vit avec les données)
- Les sorties DLC/VAME (.h5, .pkl, etc.)
- Les modèles DLC entraînés (`dlc-projects/`, `vame-projects/` — chemins à configurer dans `configs/pipeline_config.yaml`)

Pour dupliquer un setup complet sur une nouvelle machine, il faut donc :
1. Cloner le repo (ce README)
2. Copier le dossier `data/` (acquisitions brutes + Excel) à part
3. Copier ou refaire le projet DLC entraîné (dossier `dlc-projects/`)

## Liens utiles

- DeepLabCut : https://deeplabcut.github.io/DeepLabCut/
- VAME : https://github.com/LINCellularNeuroscience/VAME
- Doc complète : [`docs/ETHOFLOW.md`](docs/ETHOFLOW.md)

## Licence

À définir avec le labo.
