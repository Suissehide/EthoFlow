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

EthoFlow lit ce fichier et **ne le modifie jamais**. Le script `scripts/sync_from_excel.py` génère un `metadata.yaml` par session à partir de cet Excel.

```bash
conda activate ethoflow
python scripts/sync_from_excel.py
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

### 5.1 Workflow

```
data/<TrialCode>.mp4 + ethoflow/data/raw/<TrialCode>/metadata.yaml
                          │
                          ▼
              run_dlc_inference.py  (env: dlc)
              SuperAnimal multi-animal sur la vidéo entière
                          │
                          ▼
              ethoflow/data/dlc-output/<TrialCode>/<une>.h5  (4 tracks)
                          │
                          ▼
              assign_arenas.py  (env: ethoflow)
              centroïde de chaque track → arène contenante
                          │
                          ▼
              <TrialCode>_A1.h5, <TrialCode>_A2.h5, ...   (single-animal)
                          │
                          ▼
              run_vame.py  (env: vame)
                          │
                          ▼
              ethoflow/data/vame-output/<TrialCode>/
```

### 5.2 Pourquoi pas de crop dans le pipeline ?

Avec DLC 3.x et SuperAnimal, le détecteur multi-animal trouve les 4 souris en une seule passe sur la vidéo source. Comme les arènes sont physiquement séparées, l'identité d'un track est triviale à résoudre par sa position : `assign_arenas.py` calcule le centroïde de chaque track sur toute la vidéo et le matche au rectangle de l'arène correspondante. Pas besoin de re-encoder 4 sous-vidéos.

Le crop reste disponible (`scripts/crop_arenes.py`) pour les phases de labellisation — voir partie 4.2.

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
