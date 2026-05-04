# Pipeline d'analyse comportementale souris — Documentation

> **Stack** : DeepLabCut (estimation de pose) + VAME (segmentation comportementale)
> **Plateforme** : Windows
> **Public** : chercheurs non techniques (utilisation) + développeur successeur (maintenance)
> **Volume cible** : ~100 vidéos / mois, 1080p 60fps, 10–20 min, 4 arènes par vidéo

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
[Vidéo brute 1080p/60fps, 4 animaux dans 4 arènes]
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
git clone <URL-DU-REPO> pipeline-souris
cd pipeline-souris
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
pip install opencv-python pandas numpy streamlit pyyaml tqdm
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

> 🔑 **C'est la partie la plus importante de toute la doc.** Une convention bâclée maintenant = enfer permanent ensuite. À discuter et valider avec les chercheurs avant la première session enregistrée.

### 3.1 Arborescence cible

```
C:\labo\
├── pipeline-souris\          ← repo Git (code + cette doc)
│   ├── scripts\
│   ├── configs\
│   ├── docs\
│   └── README.md
│
├── data\
│   ├── raw\                  ← vidéos brutes (jamais modifiées)
│   │   └── 2026-05-04_projet-X_session-001\
│   │       ├── video.mp4
│   │       └── metadata.yaml
│   │
│   ├── cropped\              ← vidéos après crop des 4 arènes
│   ├── dlc-output\           ← fichiers .h5 / .csv issus de DLC
│   ├── vame-output\          ← résultats VAME
│   └── results\              ← figures et métriques finales
│
├── dlc-projects\             ← projets DLC (modèles entraînés)
│   └── souris-openfield-2026-05-04\
│
└── vame-projects\
    └── souris-openfield-2026-05-04\
```

### 3.2 Convention de nommage des sessions

Chaque enregistrement est une **session** identifiée par un dossier nommé :

```
AAAA-MM-JJ_<projet>_session-NNN
```

Exemples :

```
2026-05-04_projet-X_session-001
2026-05-04_projet-X_session-002
2026-05-12_projet-Y_session-001
```

**Règles strictes** :

- date au format ISO (`AAAA-MM-JJ`), avec tirets
- pas d'espace, pas d'accent, pas de majuscule autres que celles déjà imposées
- pas de caractère spécial sauf `-` et `_`
- nom de projet court et stable (un projet = une équipe + un protocole)

### 3.3 Schéma de métadonnée

Chaque session contient un fichier `metadata.yaml` à la racine :

```yaml
session_id: 2026-05-04_projet-X_session-001
date: 2026-05-04
projet: projet-X
chercheur: nom.prenom
protocole: openfield-15min

camera:
  modele: <à remplir>
  resolution: 1920x1080
  fps: 60

arenes:
  - id: arene-1
    coords: [x, y, w, h]   # rectangle de crop dans la vidéo source
    animal_id: M001
    condition: control
  - id: arene-2
    coords: [x, y, w, h]
    animal_id: M002
    condition: traitement-A
  - id: arene-3
    coords: [x, y, w, h]
    animal_id: M003
    condition: control
  - id: arene-4
    coords: [x, y, w, h]
    animal_id: M004
    condition: traitement-A

notes: |
  Toute observation utile (incident, comportement particulier, etc.)
```

> Ce fichier est **rempli par le chercheur au moment de l'acquisition**. Il pilote toute la suite du pipeline. Sans lui, on ne sait pas quoi faire de la vidéo.

### 3.4 Sauvegarde et archivage

À discuter et formaliser avec l'IT du labo :

- les vidéos brutes (`data/raw/`) sont **immuables** et archivées sur un stockage redondé (NAS, serveur d'archivage du labo)
- les résultats DLC et VAME (`data/dlc-output/`, `data/vame-output/`) sont aussi sauvegardés (ce sont des données dérivées mais coûteuses à recalculer)
- les vidéos croppées (`data/cropped/`) sont **éphémères** — on peut les recalculer à tout moment, donc pas besoin de les sauvegarder

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

### 4.2 Crop des arènes

Avant d'entraîner DLC, on crop les 4 arènes pour avoir 4 vidéos single-animal.

Script à mettre dans `scripts/crop_arenes.py` (à écrire — on en fera un en phase 1 du projet). Logique :

1. Lit `metadata.yaml` de la session
2. Pour chaque arène, extrait le rectangle de la vidéo source avec `ffmpeg` ou OpenCV
3. Sauvegarde dans `data/cropped/<session_id>/arene-N.mp4`

> Astuce : si les coordonnées des arènes sont stables d'une session à l'autre (caméra fixe, même setup), créer un fichier `configs/setup_default.yaml` avec les coords par défaut pour ne pas avoir à les retaper.

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

Cela produit des images annotées et une erreur en pixels. Cible : **erreur test < 5 pixels** sur 1080p. Si > 10 pixels, voir 4.10.

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

**Objectif** : à partir d'une vidéo brute déposée par un chercheur, produire automatiquement les fichiers DLC.

**Pré-requis** : partie 4 terminée, modèle DLC entraîné et évalué.

### 5.1 Workflow

```
Chercheur dépose video.mp4 + metadata.yaml dans data/raw/<session_id>/
                          │
                          ▼
        Script de pipeline (à lancer manuellement ou via .bat)
                          │
              ┌───────────┴───────────┐
              ▼                       ▼
      crop_arenes.py            run_dlc_inference.py
              │                       │
              └───────────┬───────────┘
                          ▼
              Fichiers .h5 dans data/dlc-output/<session_id>/
```

### 5.2 Script de pipeline (à écrire)

À placer dans `scripts/run_pipeline.py`. Pseudo-code :

```python
# 1. Lire la liste des sessions non encore traitées
# 2. Pour chaque session :
#    a. Charger metadata.yaml
#    b. Crop des 4 arènes -> data/cropped/<session>/
#    c. Inférence DLC -> data/dlc-output/<session>/
#    d. Mettre à jour un journal pour ne pas retraiter
```

### 5.3 Lancement par les chercheurs

Créer un `lancer_pipeline.bat` à la racine du repo :

```bat
@echo off
call conda activate dlc
cd C:\labo\pipeline-souris
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

st.set_page_config(page_title="Pipeline souris", layout="wide")
st.title("Pipeline d'analyse comportementale")

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
