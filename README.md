# EthoFlow

Pipeline d'analyse comportementale souris à partir de vidéos brutes, basé sur **DeepLabCut** (estimation de pose) et **VAME** (segmentation comportementale non-supervisée).

Le pipeline gère **deux dimensions indépendantes** :

- **Angle de la caméra** — top-view (au plafond) ou bottom-view (sous plancher transparent IR). Ça détermine uniquement quelle SuperAnimal utiliser pour le transfer learning DLC (`superanimal_topviewmouse` vs `superanimal_quadruped`).
- **Nombre d'animaux par vidéo** — 1 seul animal (**1 vidéo = 1 session**) ou plusieurs animaux dans des arènes physiquement séparées (**1 vidéo = N sessions**, typiquement 4). Ça détermine si tu as besoin d'un split par arène ou pas.

Les deux se combinent librement : tu peux faire du bottom-view avec 4 souris dans 4 arènes séparées, ou du top-view avec une souris seule dans une arène ouverte. Le mode d'inférence DLC (`--mode superanimal` multi-animal + arena split, ou crop préalable + `--mode custom` single-animal) est **choisi à l'étape 5** en fonction du nombre d'animaux par vidéo, indépendamment de l'angle.

La CLI reflète cette distinction :

- `create_project.py --kind single` = 1 animal par vidéo, pas d'arena split
- `create_project.py --kind multi` = N animaux par vidéo, arena split activé + coords par défaut écrites
- `sync_from_excel.py` détecte automatiquement le schéma depuis les feuilles de l'Excel — un seul script pour les deux cas

Le pipeline part d'une acquisition brute (vidéo + Excel des souris) et produit des CSV statistiques, des figures et des vidéos annotées, groupables par n'importe quelle variable expérimentale (génotype, traitement, sexe, etc.).

---

## Table des matières

1. [Concepts](#concepts)
2. [Prérequis machine](#prérequis-machine)
3. [Installation](#installation-first-time)
4. [Environnements conda](#environnements-conda)
5. [Lancer l'interface](#lancer-linterface)
6. [Parcours pipeline](#parcours-pipeline)
   - [Étape 0 — bifurcation modèle DLC](#étape-0--bifurcation-modèle-dlc)
   - [Étapes 1-9 — pipeline principal](#étape-1--créer-un-projet-ethoflow)
7. [Structure d'un projet](#structure-dun-projet)
8. [Index des scripts](#index-des-scripts)
9. [Troubleshooting](#troubleshooting)

---

## Concepts

**Un projet EthoFlow** = un dossier autonome qui contient les données brutes, les sorties DLC/VAME et la config, pour une expérience donnée. Chaque projet vit à un chemin absolu (ex : `D:\ethoflow\projects\bottomview-MCC-2026-06`) et tous les scripts prennent `--project-dir <chemin>`. Tu peux avoir autant de projets en parallèle que tu veux — ils sont indépendants.

**Une session** = une acquisition (une vidéo + les metadata associées : ID souris, groupe, traitement, date, etc.). Le nombre de sessions par vidéo dépend uniquement du **nombre d'animaux dans la vidéo** :
- 1 animal par vidéo → 1 session
- N animaux dans N arènes physiquement séparées → N sessions (une par arène)

Cette distinction est indépendante de l'angle de la caméra : tu peux avoir du bottom-view mono-animal, du bottom-view multi-animal, du top-view mono-animal, du top-view multi-animal.

**Un modèle DLC** = un réseau pré-entraîné qui détecte les points anatomiques (nez, oreilles, pattes, queue, etc.). Un modèle DLC vit **hors** du projet EthoFlow (dans `D:\EthoFlow\models\...` par exemple) et est **réutilisé** entre projets. C'est la partie coûteuse à produire (labellisation manuelle + jour de calcul GPU) et la partie qu'on partage entre expérimentateurs.

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

**`CUDA: True` + le nom de ta carte** → rien à faire, passe à l'étape 4.

**`CUDA: False` / `GPU : N/A`** → torch a été installé en build CPU. Répare :

```cmd
python scripts\diagnose_gpu.py --fix
```

Le script détecte l'architecture de ta carte, réinstalle torch depuis l'index CUDA adapté (nightly `cu128` pour une RTX 50xx, stable sinon), puis tu relances la vérification ci-dessus.

<details>
<summary>Pourquoi ça arrive, et pourquoi pas systématiquement</summary>

`environment-dlc.yml` demande l'index CUDA de PyTorch **en plus** de PyPI. Les wheels CUDA sont versionnées `2.5.1+cu124` et, en PEP 440, `2.5.1+cu124 > 2.5.1` — à version de base égale, pip prend donc la CUDA. C'est pour ça que ça marche la plupart du temps.

Deux cas où ça échoue quand même :

- **PyPI publie une version plus récente que l'index CUDA** (2.6.0 vs 2.5.1) : pip prend la plus haute, donc la CPU. C'est du timing, d'où le caractère intermittent.
- **RTX 50xx (Blackwell, sm_120)** : aucune build stable ne couvre cette architecture, il faut la nightly `cu128`. Un fichier conda ne peut pas la demander.

Dans les deux cas `diagnose_gpu.py --fix` tranche. À refaire après chaque `conda env create` qui retombe sur `CUDA: False` — la vérification ne coûte rien, autant la faire.

</details>

### 4. Vérifier VAME

```cmd
conda activate vame
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

`CUDA: True` → passe à la suite. `CUDA: False` → `python scripts\diagnose_gpu.py --fix`.

---

## Environnements conda

| Env | À quoi ça sert |
|---|---|
| `ethoflow` (env-pipeline) | Sert principalement à faire tourner l'app Streamlit (voir ci-dessous) et l'orchestrateur `run_pipeline.py` |
| `dlc` (env-dlc) | DeepLabCut 3.x + PyTorch — inférence, entraînement, GUI de labellisation |
| `vame` (env-vame) | VAME + scipy/matplotlib/UMAP — setup/train/segment VAME, analyses, visualisations |

---

## Lancer l'interface

Depuis la racine du dépôt, **sans activer d'environnement au préalable** :

```cmd
:: Windows
ethoflow
```

```bash
# Linux / macOS
./ethoflow
```

Le lanceur trouve lui-même le `streamlit` de l'env `ethoflow` ; si l'env
n'existe pas, il affiche la commande pour le créer. Tout argument
supplémentaire est transmis à `streamlit run` :

```bash
./ethoflow --server.address=0.0.0.0   # accessible depuis le LAN
```

Sous Linux et macOS, le script peut être lié dans un dossier du `PATH` pour
être appelé de n'importe où — les liens symboliques sont résolus :

```bash
ln -s "$PWD/ethoflow" ~/.local/bin/ethoflow
```

L'interface couvre les étapes 1 à 9 ci-dessous. Le [Parcours B](#parcours-b--entraîner-un-nouveau-modèle-dlc)
(entraînement d'un modèle DLC) reste en ligne de commande.

---

## Parcours pipeline

Un seul parcours de bout en bout, avec **une seule bifurcation** au début : soit tu as déjà un modèle DLC utilisable, soit tu dois l'entraîner. Après cette bifurcation, la suite est identique.

> **Tous les scripts marchent sans argument.** Lance `python scripts\<script>.py` : ce qui manque est demandé à l'invite (menu des projets trouvés sous `D:\EthoFlow\projects`, chemins avec valeurs par défaut). Les arguments servent à aller plus vite quand tu sais déjà, ou à scripter. `--no-prompt` fait échouer au lieu de demander (CI, automatisation).
>
> Chaque étape ci-dessous montre les deux formes.

### Étape 0 — bifurcation modèle DLC

**As-tu déjà un modèle DLC entraîné pour ton setup imaging ?**

- **Oui** → tu as un dossier `D:\EthoFlow\models\<nom_projet>\` avec un `config.yaml` dedans, produit par ton labo ou une expérience précédente. Passe directement à l'**[Étape 1](#étape-1--créer-un-projet-ethoflow)**.
- **Non** → suis d'abord le [Parcours B — entraîner un nouveau modèle DLC](#parcours-b--entraîner-un-nouveau-modèle-dlc) tout en bas, qui produit ce fameux `config.yaml`. Puis reviens à l'**Étape 1**.

Le même modèle DLC entraîné une fois est réutilisé pour tous les projets EthoFlow futurs qui partagent le même setup imaging.

---

### Étape 1 — créer un projet EthoFlow

Deux façons de lancer — **interactif** (le script demande ce qui manque) ou **tout en arguments**.

**Mode interactif** (recommandé la première fois) :

```cmd
conda activate ethoflow
python scripts\create_project.py
```

Le script demande à l'invite : dossier racine des projets (défaut `D:\EthoFlow\projects`), nom du projet, type `single`/`multi`, et config DLC (laisse vide si tu ne sais pas encore).

**Mode arguments** :

```cmd
conda activate ethoflow
python scripts\create_project.py ^
    --projects-root D:\EthoFlow\projects ^
    --name bottomview-MCC-2026-06 ^
    --kind single ^
    --dlc-config "D:\EthoFlow\models\souris-bottomview\config.yaml"
```

Le projet est créé à `<projects-root>\<name>\`. Tu peux aussi passer le chemin complet d'un coup avec `--project-dir D:\EthoFlow\projects\bottomview-MCC-2026-06` (pratique pour les scripts). Ajoute `--no-prompt` pour échouer au lieu de demander si un argument manque (CI, automatisation).

#### `--kind` — obligatoire, détermine le nombre d'animaux par vidéo

| Valeur | Quand l'utiliser | Effet |
|---|---|---|
| `single` | **1 animal par vidéo** (1 vidéo = 1 session) | Pas d'arena splitting. Excel à 1 feuille `Sessions`. |
| `multi` | **N animaux par vidéo dans N arènes séparées** (1 vidéo = N sessions) | Arena splitting activé + `default_arenes_coords` écrites dans `pipeline_config.yaml`. Excel à 3 feuilles. |

Choisis **en fonction du nombre d'animaux par vidéo, pas de l'angle caméra**. Un projet bottom-view avec 4 souris dans 4 arènes séparées est un `--kind multi` ; un projet top-view avec une seule souris en arène ouverte est un `--kind single`.

Pour du `multi`, ajuste ensuite les coordonnées d'arène à ton setup avec `calibrate_arenes.py` (les valeurs par défaut correspondent à une grille 2×2 sur du 1024×1080).

#### `--dlc-config` — optionnel, pointeur vers le modèle DLC

Le modèle DLC et le projet EthoFlow sont **complètement indépendants** :

- Le modèle vit dans `D:\EthoFlow\models\<nom>\`, **jamais copié** dans le projet EthoFlow
- Un même modèle peut être utilisé par autant de projets EthoFlow que tu veux (batches différents, mois différents, expériences différentes)
- Supprimer un projet EthoFlow ne touche pas au modèle ; re-entraîner le modèle profite immédiatement à tous les projets qui pointent dessus

`--dlc-config` accepte **le dossier du modèle autant que son `config.yaml`** : `D:\EthoFlow\models\souris-bottomview` et `D:\EthoFlow\models\souris-bottomview\config.yaml` marchent tous les deux — le `config.yaml` à la racine est trouvé tout seul. Idem pour le champ « Modèle DLC » de l'app, pour `diagnose_dlc_model.py --model-dir`, et pour une valeur écrite à la main dans le YAML.

Ce qui est écrit dans `configs/pipeline_config.yaml` est toujours le fichier :

```yaml
dlc_project_config: D:\EthoFlow\models\souris-bottomview\config.yaml
```

Cette ligne est lue **uniquement par `run_dlc_inference.py --mode custom`** (étape 5) pour savoir quel modèle appliquer.

**Tu peux le sauter** si tu ne sais pas encore quel modèle utiliser (modèle en cours d'entraînement, projet préparé pour quelqu'un d'autre, comparaison de plusieurs modèles). Dans ce cas :

```cmd
python scripts\create_project.py ^
    --project-dir D:\ethoflow\projects\bottomview-MCC-2026-06 ^
    --kind single
```

Le script affiche un warning `⚠ dlc_project_config n'est pas renseigné` et tu édites `configs/pipeline_config.yaml` à la main avant l'étape 5. Tu peux aussi re-lancer `create_project.py --force --dlc-config <path>` pour re-générer le YAML.

#### Ce que le script produit

```
D:\ethoflow\projects\bottomview-MCC-2026-06\
├── configs\pipeline_config.yaml                   ← pointeur DLC + (multi) coords d'arène
├── data\{raw,cropped,dlc-output,vame,results}\    ← dossiers vides prêts
└── bottomview-MCC-2026-06_sessions.xlsx           ← starter Excel auto-généré
```

Le starter Excel contient un onglet **Instructions** avec le mode d'emploi + les onglets de données adaptés au kind (1 feuille `Sessions` pour `single`, 3 feuilles `Subjects` + `Trials_Videos` + `Arena_Mapping` pour `multi`). Ouvre-le, remplace les lignes d'exemple grisées par tes vraies souris, sauvegarde.

### Étape 2 — remplir l'Excel de sessions

Le pipeline lit un Excel maître qui décrit tes souris. **Le template a été généré à l'étape 1** à `<project>/<project>_sessions.xlsx` — ouvre-le directement, tu n'as rien d'autre à créer.

Deux schémas selon **le nombre d'animaux par vidéo** :

**1 animal / vidéo** — feuille `Sessions` — **1 ligne par vidéo** (= 1 session) :

| id | mouse_id | group | sex | cage | birth_date | genotype_mcc | captopril |
|---|---|---|---|---|---|---|---|
| 971 | 971 | MCCiECKO | F | CD330 | 2024-10-15 | fl/fl | oui |
| 970-M1 | 970 | MCCf/f | F | CD329 | 2024-10-15 | fl/fl | oui |
| 970-M2 | 970 | MCCf/f | F | CD329 | 2024-10-15 | fl/fl | oui |

**Une seule colonne est obligatoire** :

- **`id`** — nom du fichier vidéo **sans extension** (`970-M1` → cherche `970-M1.mp4`). C'est la **clé unique de la session** et le nom du dossier créé dans `data/raw/` (préfixé `BV-`). Deux lignes ne peuvent pas avoir le même `id`.

**Deux colonnes recommandées** :

- **`mouse_id`** — identifie l'**animal**. Se répète sur plusieurs lignes si la même souris est filmée à plusieurs timepoints (**design longitudinal**) : dans l'exemple, la souris 970 apparaît deux fois avec des `id` distincts → deux sessions séparées, regroupables par animal dans les analyses.
- **`group`** — ta variable de comparaison principale (génotype, traitement). C'est ce qui sépare tes groupes dans `analyze_vame.py`.

**Toutes les autres colonnes sont libres.** Renomme-les, supprime-les, ou ajoute les tiennes : chaque colonne remplie est recopiée telle quelle dans le `metadata.yaml` de la session et devient utilisable comme variable de groupement. Celles du template (`sex`, `cage`, `birth_date`, `genotype_*`, `captopril`…) sont des exemples du projet MCC — adapte-les à ton étude. Laisse une cellule vide si l'info n'existe pas.

> Sans colonne `id`, `mouse_id` sert d'identifiant de session — mode historique 1 vidéo/souris, sans support longitudinal.

**N animaux / vidéo** — 3 feuilles :

- `Subjects` — 1 ligne par MouseID avec attributs (groupe M1/M2, stress, notes)
- `Trials_Videos` — 1 ligne par vidéo (TrialCode conventionnel `OF-<M1|M2>-YYYYMMDD-V<##>`, date, FPS, dimensions)
- `Arena_Mapping` — 1 ligne par (vidéo × arène), reliant chaque arène de chaque vidéo à un MouseID

**Colonnes obligatoires** : `Trials_Videos.TrialCode`, `Arena_Mapping.TrialCode`, `Arena_Mapping.Arena`, `Arena_Mapping.MouseID`. Le reste est libre et adaptable à ton étude.

Un exemple pré-rempli est présent dans chaque feuille (lignes grisées, à supprimer). Voir aussi `configs/metadata_template.yaml` pour la structure de la metadata YAML produite en aval.

### Étape 3 — sync des sessions

Un seul script, qui **détecte le schéma automatiquement** depuis les feuilles de ton Excel (feuille `Sessions` → 1 animal/vidéo ; feuilles `Trials_Videos` + `Arena_Mapping` → N animaux/vidéo).

**Mode interactif** (le script demande ce qui manque) :

```cmd
python scripts\sync_from_excel.py
```

Il propose un menu des projets trouvés sous `D:\EthoFlow\projects`, auto-détecte l'Excel à la racine du projet, puis demande le dossier des vidéos et la date.

**Mode arguments** :

```cmd
python scripts\sync_from_excel.py ^
    --project-dir D:\EthoFlow\projects\bottomview-MCC-2026-06 ^
    --videos-dir E:\data\bottom_view\08062026
```

`--excel` est optionnel : le script prend le `*_sessions.xlsx` à la racine du projet. Répète la commande pour chaque batch d'acquisition (`--videos-dir` change, l'Excel reste le même). `--overwrite` pour re-générer une metadata existante, `--dry-run` pour prévisualiser.

Résultat : un `metadata.yaml` par session dans `data/raw/<session_id>/`.

### Étape 4 — (multi-animal seulement) crop optionnel des arènes

Si tu as **N animaux par vidéo** (peu importe l'angle caméra), tu as deux voies équivalentes pour arriver aux .h5 single-animal :

- **Voie A — DLC multi-animal + split** : lance DLC directement sur la vidéo entière, puis split la sortie multi-animal en N par arène (via `assign_arenas.py`). Plus rapide au global.
- **Voie B — crop puis DLC single-animal** : découpe d'abord la vidéo en N vidéos single-animal (via `crop_arenes.py`), puis lance DLC en mode single-animal sur chacune. Sortie plus propre, indispensable si tu veux **labelliser** des frames pour améliorer le modèle.

Pour la voie B (crop) :

```cmd
:: Interactif — demande le projet
python scripts\calibrate_arenes.py
python scripts\crop_arenes.py --all

:: Ou avec arguments
python scripts\calibrate_arenes.py --project-dir D:\EthoFlow\projects\mon-projet
python scripts\crop_arenes.py --project-dir D:\EthoFlow\projects\mon-projet --all
```

Si tu as **1 animal par vidéo**, cette étape n'existe pas — passe directement à l'étape 5.

### Étape 5 — inférence DLC

Trois modes possibles selon le combo (nombre d'animaux, modèle DLC dispo) :

**1 animal / vidéo, modèle DLC custom** (le cas typique bottom-view) :

```cmd
conda activate dlc

:: Interactif — demande le projet
python scripts\run_dlc_inference.py --all --mode custom

:: Ou avec arguments
python scripts\run_dlc_inference.py --project-dir D:\EthoFlow\projects\mon-projet --all --mode custom
```

> **Si tu n'as pas renseigné `--dlc-config` à l'étape 1**, le script te demande quel modèle utiliser au premier lancement, avec un menu des modèles trouvés sous `D:\EthoFlow\models` :
>
> ```
> ℹ  Aucun modèle DLC configuré pour ce projet.
>
> Modèles DLC trouvés dans D:\EthoFlow\models :
>   1. souris-bottomview
>   2. openfield-topview
>   3. (autre chemin)
> Modèle DLC [1] :
> ```
>
> Ton choix est écrit dans `configs/pipeline_config.yaml` — il ne te sera plus redemandé. Même comportement si le modèle référencé a été déplacé ou supprimé.

**N animaux / vidéo, DLC multi-animal SuperAnimal** (voie A, défaut sans training custom) :

```cmd
python scripts\run_dlc_inference.py --all
python scripts\run_dlc_inference.py --project-dir D:\EthoFlow\projects\mon-projet --all
```

**N animaux / vidéo, single-animal sur vidéos croppées** (voie B, quand tu as croppé à l'étape 4) :

```cmd
python scripts\run_dlc_inference.py --all --mode single-animal ^
    --video-adapt --video-adapt-batch-size 2
```

Options utiles pour tous les modes :
- `--all` — traite toutes les sessions non encore traitées
- `<session_id>` en argument positionnel — cible une session précise
- `--video-adapt` sur des vidéos assez différentes du training set → adapte le modèle sur les statistiques de tes vidéos (lent mais améliore la précision)
- `--video-adapt-batch-size 2` sur GPU 16 GB (défaut 8 déborde en VRAM sur RTX 4080/5080)

Sortie : `data/dlc-output/<session>/<hash>.h5` + éventuellement `_labeled.mp4`.

### Étape 6 — préparer les fichiers pour VAME

VAME veut un h5 single-animal par session, sans sauts de tracking aberrants. Cette étape fait **plus qu'un simple seuil de confiance** — recommandation Tony (VAME/LIN) :

> « Le cutoff n'est qu'un proxy qui marche en moyenne sur un nombre suffisant de frames, et devrait toujours être combiné à d'autres méthodes. J'utiliserais un cutoff autour de 70 % et j'appliquerais des méthodes qui détectent les frames individuelles au tracking cassé et essaient de les corriger. »

#### 6a — Calibrer l'échelle px/cm (une fois par setup caméra)

Nécessaire pour juger si un déplacement de label est physiquement plausible. Tony suggère de photographier une règle avec ton setup (plutôt que d'utiliser les dimensions de l'arène : plus l'objet est grand, plus la distorsion de lentille fausse la mesure).

```cmd
:: Interactif — liste les vidéos de tes sessions, tu en choisis une
python scripts\calibrate_scale.py

:: Sur une session précise (pas de chemin à taper)
python scripts\calibrate_scale.py --project-dir D:\EthoFlow\projects\mon-projet ^
    --session BV-970 --known-cm 10

:: Depuis une photo de règle
python scripts\calibrate_scale.py --project-dir D:\EthoFlow\projects\mon-projet ^
    --image D:\EthoFlow\calibration\regle.png --known-cm 10

:: Si tu connais déjà la valeur
python scripts\calibrate_scale.py --project-dir D:\EthoFlow\projects\mon-projet --set 12.5
```

Sans argument, le script propose les vidéos de tes sessions :

```
Sur quelle vidéo veux-tu calibrer ?
  1. BV-970  (970.mp4)
  2. BV-971  (971.mp4)
  3. BV-998  (998.mp4)
  4. Une photo de règle (image)
  5. Une autre vidéo (chemin libre)
Choix [1] :
```

Une fenêtre s'ouvre, tu cliques les deux extrémités de la distance connue, le script écrit `px_per_cm` dans `configs/pipeline_config.yaml`. L'étape 6b la lit automatiquement.

#### 6b — Nettoyage des poses

**À quoi ça sert.** DLC te rend une position pour chaque keypoint, sur chaque frame, quoi qu'il arrive — y compris quand la patte est cachée sous le corps ou qu'un reflet IR ressemble à une truffe. Le fichier brut contient donc des positions fausses, mêlées aux bonnes. Ce script les repère et les remplace par une position reconstruite à partir des frames voisines. VAME segmente des **trajectoires** : un keypoint qui téléporte à travers l'arène pendant 3 frames crée un faux motif comportemental que tu retrouveras dans tes stats.

**Ce n'est pas obligatoire.** Le pipeline tourne sans : tu peux enchaîner directement sur l'étape 7 avec les `.h5` bruts. Cette étape est de l'assurance qualité, pas une conversion de format. Elle vaut le coup si tes vidéos ont des occlusions (bottom-view avec pattes qui passent sous le corps), un contraste faible, ou si le modèle DLC est encore jeune. Si ton modèle est excellent et tes vidéos propres, elle ne changera presque rien — le résumé de fin (`% utilisables`, nombre de frames réparées) te le dira en une ligne.

Quatre passes successives par session :

1. **Filtre médian temporel** (`dlc.filterpredictions`, fenêtre 5 frames) — tue les jitters d'une ou deux frames.
2. **Cutoff de likelihood** (`--likelihood-threshold`, défaut 0.70) — le filet grossier.
3. **Détection de vitesse aberrante** (`--max-speed`, défaut 5 m/s) — la méthode que Tony privilégie. Convertit chaque déplacement inter-frame en m/s via `px_per_cm` et marque les frames physiquement impossibles. **Indépendant de la likelihood** : attrape aussi les labels *confiants mais faux*. Nécessite l'étape 6a, sinon la passe est silencieusement désactivée.
4. **Détection de points collants** — repère les coordonnées où un keypoint atterrit anormalement souvent (reflet IR fixe, coin d'arène). Tony : « parfois les labels bruités sautent toujours au même point que l'animal ne peut pas atteindre ». Le script distingue un artefact (frames dispersées dans le temps) d'une immobilité réelle (frames contiguës) et ne touche qu'aux premiers.

Les frames marquées par 2/3/4 sont **interpolées** depuis leurs voisines valides, pas jetées. Les trous > `--interp-limit` (défaut 25 frames ≈ 1 s) restent NaN.

**Qu'est-ce que la « likelihood » exactement ?** C'est un nombre entre 0 et 1 que DeepLabCut produit à côté de chaque coordonnée `(x, y)` : sa propre estimation de la probabilité que ce point soit au bon endroit. Elle vient de l'intensité du pic dans la carte de chaleur que le réseau produit pour ce keypoint — pic net et isolé → likelihood proche de 1 ; carte plate ou à deux pics → likelihood basse. C'est une auto-évaluation du modèle, pas une vérité : **un point peut être faux tout en étant confiant** (le modèle confond systématiquement patte avant droite et patte avant gauche, par exemple). D'où les passes 3 et 4 ci-dessous, qui ne regardent pas la likelihood du tout.

```cmd
:: Interactif — les seuils sont demandés avec leur valeur par défaut
python scripts\prepare_vame_input_custom.py

:: Ou avec arguments (rien n'est demandé)
python scripts\prepare_vame_input_custom.py --project-dir D:\EthoFlow\projects\mon-projet ^
    --likelihood-threshold 0.70 --max-speed 5
```

Sans argument, le script explique chaque seuil avant de le demander :

```
Seuil de likelihood — la confiance que DLC attribue à chaque
point, entre 0 (aucune) et 1 (certaine). En dessous du seuil,
la position est jugée non fiable, effacée, puis reconstruite
par interpolation depuis les frames voisines.
  · plus haut (0.9) = plus sévère, plus de points reconstruits
  · plus bas  (0.3) = plus permissif, on garde des points douteux
  · 0.7 = recommandation de l'équipe VAME/LIN
Seuil de likelihood [0.7] :
```

**Critère d'acceptation** — le script produit un graphe avant/après par session dans `data/dlc-output/_qc_trajectories/`. Le nom du fichier est `<session>_<keypoint>.png`, par exemple `BV-970_tail_base.png` : le graphe ne trace **qu'un seul keypoint**, celui passé à `--qc-bodypart`, et `tail_base` est le défaut. C'est le point le plus stable du corps — il ne disparaît jamais sous l'animal et bouge peu par rapport au centre de masse, donc un saut visible sur sa trajectoire est forcément une erreur de tracking, jamais un vrai mouvement. Le keypoint est dans le nom pour que tu puisses en tracer plusieurs sans écraser le précédent :

```cmd
python scripts\prepare_vame_input_custom.py --qc-bodypart front_paw_left
:: → data/dlc-output/_qc_trajectories/BV-970_front_paw_left.png, à côté du tail_base
```

Le nom doit être **exactement** celui du modèle DLC (`nose`, `left_ear`, `right_ear`, `front_paw_left/right`, `hind_paw_left/right`, `tail_base/mid/tip`, `center`, `left_flank` pour le modèle par défaut — c'est `front_paw_left`, pas `paw_front_left`). Un nom inconnu ne produit aucun graphe et le dit, en listant les keypoints disponibles :

```
  ⚠ graphe de contrôle non généré : keypoint « paw_front_left » absent des
    données. Keypoints disponibles : center, front_paw_left, front_paw_right, ...
```

Objectif de Tony :

> « Ce que tu veux voir au final, c'est que tracer la trajectoire de l'animal sur toute la vidéo ne montre aucun saut anormal de position dans l'arène, sans avoir à jeter complètement des points, ce qui pose beaucoup de problèmes en aval. »

Si des sauts subsistent sur le graphe, baisse `--max-speed` (4 m/s) ou monte `--likelihood-threshold`. Si au contraire trop de frames sont réparées (>10-15 %), c'est que le modèle DLC est encore faible — retourne au Parcours B plutôt que de compenser en post-processing.

**Options utiles** :

- `--px-per-cm 12.5` — override l'échelle sans passer par 6a
- `--max-speed 4` — plus strict sur les sauts
- `--no-sticky-detection` — désactive la passe 4
- `--qc-bodypart center` — trace un autre keypoint dans le graphe de contrôle
- `--no-qc-plot` — pas de graphes (gagne quelques secondes par session)
- `--no-prompt` — n'ouvre aucune invite, prend les défauts (pour du batch)

**Si tu as N animaux par vidéo** (voie A), fais le split par arène en amont :

```cmd
conda activate ethoflow
python scripts\assign_arenas.py --all
python scripts\assign_arenas.py --project-dir D:\EthoFlow\projects\mon-projet --all
```

Puis éventuellement `fill_nan_h5.py --root <project>/data/dlc-output` pour boucher les NaN résiduels si VAME râle.

### Étape 7 — setup + train + segment VAME

VAME s'entraîne **une fois par projet** (le VAE apprend la structure des poses de tes souris). Compte 3-8h sur GPU pour l'entraînement.

#### Ce que `setup` te demande

Avant de créer le projet, `setup` demande les quatre hyperparamètres qui changent le résultat ou la durée de l'entraînement. Chacun est expliqué à l'invite et arrive avec son défaut — entrée vide = défaut :

| Question | Flag | Défaut | Ce qu'il commande |
|---|---|---|---|
| Nombre de motifs | `--n-clusters` | 15 | combien de comportements distincts la segmentation produit, donc combien de clips tu auras à nommer à l'étape 8 |
| Fenêtre temporelle | `--time-window` | 30 frames | la durée que le VAE regarde d'un coup pour décider d'un motif (30 frames = 1 s à 30 fps) |
| Epochs max | `--max-epochs` | 500 | la durée d'entraînement — 3-8 h sur GPU ; mets 100 pour un premier essai de bout en bout en ~1 h |
| Seuil de confiance | `--pose-confidence` | 0.6 | en dessous, VAME masque le point. Son propre défaut (0.99) masquerait la majorité des points ; 0.6 s'aligne sur l'étape 6b |

Les valeurs sont écrites dans `data/vame/config.yaml`, où tu peux les rééditer avant `train`. Donner un flag évite la question correspondante ; `--no-prompt` prend tous les défauts sans rien demander (batch, CI).

**Combien de motifs ?** 15 est le défaut de VAME, pas une contrainte. C'est le nombre de clips que tu devras visionner et nommer à l'étape 8 : 15 motifs se labellisent en une petite heure, 30 en une demi-journée, et au-delà de ~40 sur un jeu de données modeste tu commences à découper le même comportement en sous-motifs quasi identiques. Les analyses en aval acceptent n'importe quelle valeur ; c'est ton temps de labellisation qui est la contrainte.

Rien n'est irréversible : `segment --n-clusters 25` réécrit la valeur et crée **son propre dossier de résultats** (`results/<session>/<model>/hmm-25/` à côté de `hmm-15/`), sans écraser les précédents — tu peux comparer deux granularités sans tout refaire. En revanche `motif_labels.csv` est unique par projet : si tu changes de `n_clusters` après avoir annoté, sauvegarde ton CSV (`motif_labels_hmm15.csv`) et repasse-le plus tard avec `analyze_vame.py --labels`, sinon `--regen-labels` te fabriquera le nouveau squelette par-dessus.

#### Les six commandes

```cmd
conda activate vame

python scripts\run_vame.py --project-dir D:\EthoFlow\projects\mon-projet setup      :: 1. Init du projet VAME
python scripts\run_vame.py --project-dir D:\EthoFlow\projects\mon-projet align      :: 2. Alignement égocentrique des poses
python scripts\run_vame.py --project-dir D:\EthoFlow\projects\mon-projet trainset   :: 3. Construction du trainset
python scripts\run_vame.py --project-dir D:\EthoFlow\projects\mon-projet train      :: 4. Entraînement du VAE (LONG)
python scripts\run_vame.py --project-dir D:\EthoFlow\projects\mon-projet evaluate   :: 5. Courbes de loss, KL divergence
python scripts\run_vame.py --project-dir D:\EthoFlow\projects\mon-projet segment    :: 6. Un motif par frame
```

Comme partout ailleurs, `--project-dir` est **optionnel** : sans lui, le script liste les projets trouvés et te demande lequel utiliser. Seule contrainte d'écriture : il doit précéder la sous-commande.

```cmd
python scripts\run_vame.py setup
python scripts\run_vame.py setup --n-clusters 25 --max-epochs 100   :: sans questions
```

Raccourci : `python scripts\run_vame.py all` enchaîne setup → segment d'une traite.

Sortie : `data/vame/results/<session>/<model>/hmm-15/15_hmm_label_<session>.npy` (1 label motif par frame ; `hmm-25/` si tu as demandé 25 motifs).

### Étape 8 — labelliser les motifs à la main

VAME te donne autant de motifs que le `n_clusters` choisi à l'étape 7, numérotés de 0 à N-1 (0-14 avec le défaut de 15). Il faut les nommer et les catégoriser.

**1. Générer les clips + le fichier de labels** — une seule commande :

```cmd
python scripts\run_vame.py motif-videos
python scripts\run_vame.py --project-dir D:\EthoFlow\projects\mon-projet motif-videos
```

Elle produit les clips (`data/vame/results/community_videos/motif_<N>.mp4`) **et** `data/vame/motif_labels.csv`, pré-rempli avec une ligne par motif :

```csv
motif_id;label;category;confidence;qc_inspected_sessions;notes;usage_pct;video
0;;;;;;18.42;results/community_videos/motif_0.mp4
1;;;;;;12.07;results/community_videos/motif_1.mp4
2;;;;;;9.85;results/community_videos/motif_2.mp4
```

`usage_pct` (fréquence du motif, toutes sessions confondues) et `video` (chemin du clip) sont remplis pour toi. `label` et `category` sont vides — c'est ton travail après visionnage. Trie par `usage_pct` décroissant et commence par le haut : les motifs à moins de 1 % ne pèsent presque rien dans les analyses.

Le fichier existant n'est **jamais écrasé** — ton travail d'annotation est préservé même si tu relances `motif-videos`. Pour repartir de zéro : `--regen-labels`. Pour regénérer le CSV seul, sans refaire les clips (long) :

```cmd
python scripts\run_vame.py motif-labels
```

**2. Remplir `label` et `category`.** Le `label` est un nom libre et spécifique (`grooming_face`, `walking_slow`) ; la `category` vient du référentiel ETHOGRAM : `Locomotion`, `Stationary`, `Vertical exploration`, `Sniffing`, `Grooming`, `Exploration`, `Specific behaviors`, `Transitions`. Les analyses groupent par `category`, donc reste dans cette liste.

```csv
motif_id;label;category;confidence;qc_inspected_sessions;notes;usage_pct;video
0;grooming_face;Grooming;high;BV-970;;18.42;results/community_videos/motif_0.mp4
1;walking;Locomotion;high;BV-970,BV-971;;12.07;results/community_videos/motif_1.mp4
```

Un motif ininterprétable (bruit de tracking, animal hors champ) : mets `artifact` dans **`confidence`**, `analyze_vame.py` l'exclura des stats au lieu de le compter comme un comportement.

> `artifact` est aussi accepté dans `category`, parce que cette section l'a longtemps indiqué là et que des CSV annotés ainsi existent. Les deux marchent, `confidence` est l'emplacement canonique — `category` est réservée aux huit valeurs ETHOGRAM, dont `artifact` ne fait pas partie.

Ce CSV est lu par toutes les analyses en aval. Sans lui, les figures affichent `motif_0`, `motif_1`, etc.

### Étape 9 — analyses + visualisations

```cmd
:: Interactif
python scripts\analyze_vame.py

:: Ou avec arguments
python scripts\analyze_vame.py --project-dir D:\EthoFlow\projects\mon-projet

:: Analyses étendues (bouts, spatial, temporal quarters)
python scripts\analyze_vame.py --project-dir D:\EthoFlow\projects\mon-projet --extended --extended-by group4
```

Sortie dans `data/vame/analysis/` :
- **CSV** : `motif_usage.csv`, `motif_usage_long.csv`, `stats_by_motif_*.csv`, `usage_by_category.csv`
- **Heatmaps groupées** : `heatmap_usage_by_<colonne>.png` (sessions triées par groupe, séparateurs visuels)
- **Barres + boxplots** : `mean_by_*.png`, `boxplots_top_by_*.png`, `boxplots_by_category_by_*.png`
- **Extended** : `bout_duration_by_*.png`, `thigmotaxis_by_*.png`, `temporal_by_motif_*.png`

Les stats utilisent Mann-Whitney (2 groupes) ou Kruskal-Wallis (≥3 groupes) avec correction Benjamini-Hochberg.

#### Comment les colonnes de ton Excel deviennent des graphes

C'est le point central de l'analyse, donc en détail. **Toute colonne de ton Excel peut servir d'axe de comparaison**, y compris celles que tu as inventées — le script ne connaît aucune colonne à l'avance.

**La chaîne, de bout en bout :**

```
mon_projet_sessions.xlsx          ton fichier, une ligne par vidéo
        │                          colonnes : id, mouse_id, group, sex, captopril, ...
        │  python scripts\sync_from_excel.py
        ▼
data/raw/<session>/metadata.yaml  chaque colonne recopiée telle quelle
        │
        │  python scripts\analyze_vame.py
        ▼
data/vame/analysis/*.png          une série de graphes par colonne exploitable
```

`sync_from_excel.py` recopie **toutes** les colonnes, pas seulement celles qu'il connaît. Une colonne `regime_alimentaire` que tu ajoutes dans ton Excel se retrouve dans les metadata, puis devient utilisable comme axe sans toucher au code.

**1. Voir ce qui est disponible.** À lancer en premier :

```cmd
python scripts\analyze_vame.py --list-columns
```

```
Colonnes exploitables comme axe de comparaison (6) :

  captopril                2 groupes : Captopril (8 sessions), Control (8 sessions)
  condition                2 groupes : MCCiECKO (8 sessions), MCCf/f (8 sessions)
  regime_alimentaire       2 groupes : standard (8 sessions), gras (8 sessions)
  sex                      2 groupes : M (8 sessions), F (8 sessions)
  cage                     4 groupes : C0 (4 sessions), C1 (4 sessions), ...
  group4                   4 groupes : MCCiECKO_Captopril (4 sessions), ...
```

Une colonne absente de cette liste a soit **une seule valeur** (rien à comparer — typiquement `operateur` si c'est toujours toi), soit **plus de 12** (c'est un identifiant comme `mouse_id`, pas un facteur expérimental). Les alias qui contiennent exactement la même information (`group` et `condition`) sont dédoublonnés pour ne pas produire deux fois les mêmes figures.

Si une colonne que tu attends n'apparaît pas : vérifie qu'elle est bien dans ton Excel **et** que tu as relancé `sync_from_excel.py` depuis que tu l'as ajoutée.

**2. Lancer sans rien préciser** — le script produit une série complète pour *chaque* colonne exploitable :

```cmd
python scripts\analyze_vame.py
```

```
  → axes de comparaison : captopril, condition, regime_alimentaire, sex, cage, group4
✓ Heatmap groupée par captopril : heatmap_usage_by_captopril.png
✓ Barres par captopril (2 groupes) : mean_by_captopril.png
  → boxplots top motifs : boxplots_top_by_captopril.png (motifs [14, 6, 8, 5, 2, 12])
✓ Stats Mann-Whitney (BH-corrected) : stats_by_motif_captopril.csv  (3/15 motifs significatifs à q<0.05)
... (idem pour condition, regime_alimentaire, sex, cage, group4)
```

Pour chaque axe tu obtiens : une heatmap groupée, un graphe en barres, des boxplots des 6 motifs les plus différenciants, un CSV de stats, et — si tu as rempli les catégories dans `motif_labels.csv` — les mêmes par catégorie ETHOGRAM.

**3. Restreindre à ce qui t'intéresse.** Sur 6 axes ça fait beaucoup de fichiers ; `--group-by` limite :

```cmd
:: Un seul axe
python scripts\analyze_vame.py --group-by captopril

:: Deux axes, chacun avec sa série de graphes
python scripts\analyze_vame.py --group-by condition captopril
```

**4. Croiser deux colonnes** (design factoriel) avec `--cross` :

```cmd
:: 4 groupes : MCCf/f_Control, MCCf/f_Captopril, MCCiECKO_Control, MCCiECKO_Captopril
python scripts\analyze_vame.py --cross condition captopril

:: Croisement + axes simples, et plusieurs croisements
python scripts\analyze_vame.py --group-by sex --cross condition captopril --cross sex cage
```

La colonne composite s'appelle `condition_x_captopril` et les fichiers suivent (`mean_by_condition_x_captopril.png`). Une session dont l'une des deux valeurs manque est exclue de ce croisement uniquement, pas des autres graphes.

`group4` (génotype × captopril) existe en dur pour rétrocompatibilité avec les analyses déjà produites — c'est exactement l'équivalent de `--cross condition captopril`.

**5. Analyses étendues sur l'axe de ton choix :**

```cmd
python scripts\analyze_vame.py --extended --extended-by regime_alimentaire
python scripts\analyze_vame.py --extended --cross condition captopril --extended-by condition_x_captopril
```

**Combien de groupes, combien de sessions ?** Le test statistique s'adapte : 2 groupes → Mann-Whitney, 3+ → Kruskal-Wallis. En dessous de 3 sessions par groupe le motif est ignoré (pas assez pour un test). Un axe à 6 groupes sur 16 sessions donnera des p-values inexploitables — regarde `--list-columns` et vérifie que chaque valeur a assez de sessions avant de conclure quoi que ce soit.

**Ajouter un facteur en cours de route :**

1. Ouvre `mon_projet_sessions.xlsx`, ajoute une colonne (nom sans espace ni accent, ex : `regime_alimentaire`), remplis-la
2. `python scripts\sync_from_excel.py` — les metadata sont mises à jour
3. `python scripts\analyze_vame.py --list-columns` — vérifie qu'elle apparaît
4. `python scripts\analyze_vame.py --group-by regime_alimentaire`

Aucune ré-inférence DLC ni ré-entraînement VAME nécessaire : les motifs sont déjà calculés, tu ne fais que les recouper autrement.

**Visualisations optionnelles** (parlant pour figures/posters) :

```cmd
:: GIF avec bande de motif color-codée sous la vidéo
python scripts\motif_gif.py --session BV-970 --duration 60
python scripts\motif_gif.py --project-dir D:\EthoFlow\projects\mon-projet --session BV-970 --duration 60

:: Manifold VAME style README, en pooled (référentiel commun toutes sessions)
python scripts\behavior_structure_gif.py --session BV-970 ^
    --pool-all-sessions --with-video --duration 30 --output-format mp4

:: Dendrogramme des communautés de motifs avec labels lisibles
python scripts\community_dendrogram.py
python scripts\community_dendrogram.py --project-dir D:\EthoFlow\projects\mon-projet --group MCCiECKO
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

### B.1 — Générer un dossier de config avec le wizard

Plutôt que d'éditer le `_config.py` versionné du repo, le wizard te crée un fichier custom dans un dossier dédié à ton projet DLC. Chaque projet DLC = son propre dossier autonome.

```cmd
conda activate dlc
python scripts\dlc_model-training\00_init_training_config.py
```

Le wizard te demande à l'invite, dans l'ordre :

1. **Dossier de travail** (défaut `D:\EthoFlow\models`) — la racine où tes projets DLC vont vivre
2. **Nom du projet DLC** (ex : `souris-bottomview`) — le dossier `<workdir>\<nom>\` sera créé automatiquement
3. **Identifiant expérimentateur** (défaut `labo`)
4. **Chemin de la vidéo pilote** (.mp4)
5. **SuperAnimal** : `quadruped` (bottom-view, voit les pattes) ou `topviewmouse` (top-view, pattes non visibles)
6. **Nombre de frames k-means** (défaut 120)

Résultat : un `_config.py` écrit dans `D:\EthoFlow\models\souris-bottomview\_config.py`.

### B.2 — Setup du projet DLC + auto-extraction de frames

```cmd
python scripts\dlc_model-training\01_setup_project.py
```

Le dossier de config est demandé à l'invite (menu des dossiers trouvés sous `D:\EthoFlow\models`), ou donné directement avec `--config-dir D:\EthoFlow\models\souris-bottomview`.

Cette commande fait **tout en une passe** :

1. Crée le projet DLC. `deeplabcut.create_new_project` nomme toujours son dossier `<nom>-<expérimentateur>-<date>` — le suffixe daté est câblé dans l'API DLC, on ne peut pas lui demander autre chose. Tu as donc, l'espace d'un instant, `D:\EthoFlow\models\souris-bottomview-labo-2026-08-26\` à côté du `souris-bottomview\` que le wizard a créé à l'étape B.1.
2. **Merge le dossier daté dans le dossier de config**, puis le supprime. Sans ça tu aurais deux dossiers frères pour un seul modèle — l'un avec ton `_config.py`, l'autre avec le projet DLC — et `--config-dir` ne désignerait pas le même endroit que le `config.yaml`. Le merge déplace le contenu, efface le dossier daté devenu vide, et réécrit les chemins absolus à l'intérieur du `config.yaml` (`project_path` et les clés de `video_sets`, qui pointaient encore vers l'ancien nom). Résultat : **un seul dossier par modèle, au nom que tu as choisi** — exactement celui que tu passeras plus tard en `--dlc-config` dans le Parcours A.
3. **Auto-patche** le `config.yaml` DLC :
   - Écrit les 12 bodyparts (`nose`, `left_ear`, `right_ear`, `front_paw_left/right`, `hind_paw_left/right`, `tail_base/mid/tip`, `center`, `left_flank`) — modifiable via `DEFAULT_BODYPARTS` dans `_config.py`
   - Écrit le skeleton anatomique (12 liaisons) — modifiable via `DEFAULT_SKELETON`
   - Règle `numframes2pick = 120` (ou la valeur `N_AUTO_FRAMES` de ton config)
4. Lance `dlc.extract_frames` en mode k-means automatique

**Aucune édition manuelle** de `_config.py` n'est nécessaire — `PROJECT_DIR = WORKDIR / PROJECT_NAME` est déterministe, ce que tu as tapé dans le wizard est exactement le chemin utilisé.

**Layout final** :

```
D:\EthoFlow\models\souris-bottomview\
├── _config.py              ← écrit par le wizard
├── config.yaml             ← DLC (auto-patché)
├── videos\
├── labeled-data\
├── training-datasets\
└── dlc-models\
```

### B.3 — Préparer le training set complet

C'est **la** phase qui détermine la qualité finale du modèle. Fais tout ce qui est ci-dessous **avant** le premier entraînement.

**Cible finale : 200-300 frames labellisées** pour la première passe complète (recommandation Tony/LIN), réparties sur **6-10 souris différentes**. Répartition :

- **100-150 frames par k-means** (extraites automatiquement par 01 + 04) — couverture visuelle générale
- **50-150 frames sélectionnées manuellement** dans les vidéos (situations difficiles)

Détail des étapes B.3.1 → B.3.4 ci-dessous.

#### B.3.1 — Étendre à plusieurs souris avant le premier train

Recommandation Tony : « prendre autant d'animaux différents que possible » **dès le premier entraînement**. C'est ce qui empêche le modèle d'apprendre des raccourcis liés à une souris ou un décor unique.

Passe les vidéos directement en CLI — le script écrit automatiquement dans `ADDITIONAL_VIDEOS` de ton `_config.py`, tu n'as pas besoin d'ouvrir le fichier :

```cmd
python scripts\dlc_model-training\04_add_videos.py ^
    --config-dir D:\EthoFlow\models\souris-bottomview ^
    --videos D:\data\bottom_view\2.mp4 D:\data\bottom_view\3.mp4 ^
             D:\data\bottom_view\4.mp4 D:\data\bottom_view\5.mp4
```

Extrait `NEW_VIDEO_FRAMES` (défaut 20) frames k-means par nouvelle vidéo. Par défaut le script **ajoute** aux vidéos déjà présentes dans `ADDITIONAL_VIDEOS` (dédup automatique), passe `--replace-videos` pour repartir de zéro.

Override du nombre de kmeans par vidéo directement en CLI si tu veux dévier du défaut :

```cmd
python scripts\dlc_model-training\04_add_videos.py ^
    --config-dir D:\EthoFlow\models\souris-bottomview ^
    --videos D:\data\... ^
    --new-video-frames 10
```

**Ajuste les quotas k-means selon le nombre de vidéos** pour rester dans le range 100-150 kmeans total recommandé par Tony :

| Vidéos totales | `N_AUTO_FRAMES` (pilote) | `NEW_VIDEO_FRAMES` (par vidéo) | Total kmeans |
|---|---|---|---|
| 5 (pilote + 4) | 60 | 20 | 60 + 80 = 140 |
| 6 (pilote + 5) | 50 | 15 | 50 + 75 = 125 |
| 8 (pilote + 7) | 30 | 15 | 30 + 105 = 135 |

Si tu as déjà lancé 01 avec 120 frames sur le pilote, tu peux compenser en baissant `NEW_VIDEO_FRAMES` à 5-10 avant de lancer 04.

#### B.3.2 — Extraction manuelle des frames difficiles

Sur les 200-300 frames cibles, il t'en manque 50-150 à sélectionner à la main. Lance l'extraction manuelle depuis Python :

```cmd
conda activate dlc
python
```

Puis :

```python
import deeplabcut
deeplabcut.extract_frames(
    r"D:\EthoFlow\models\souris-bottomview\config.yaml",
    mode="manual",   # ← clé : passe le mode automatique kmeans
    crop=False,
    userfeedback=False,
)
```

Une fenêtre s'ouvre par vidéo présente dans le projet, avec un lecteur + slider. Utilise les flèches gauche/droite pour du frame-par-frame, et le bouton **« Grab frames »** pour sauvegarder la frame courante.

**Comment répartir ton budget manuel** :

- **Divise par vidéo** : si tu vises 100 frames manuelles au total sur 5 vidéos, c'est 20/vidéo.
- **15-20 % faciles** : locomotion normale, pattes visibles. Frames floues attendues et OK.
- **80-85 % difficiles** : celles où toi-même hésites sur la position du keypoint. Rearing, occlusions, pattes qui sortent/rentrent sous le corps, ambiguïtés L/R.

**Workflow sur une vidéo, en pratique** :

1. **Passe une fois à vitesse normale** (10-15 min), note mentalement ou sur papier les timestamps qui te poseront problème :
   - Rearing (surtout côté gauche ET côté droit → capture la directionalité)
   - Demi-tours rapides avec L/R ambigu
   - Grooming où les pattes se superposent
   - Moments où la souris grimpe le long du mur
   - Blur de pattes en full-run

2. **Revient sur chaque timestamp** en frame-par-frame. Une action difficile de 2s fait 60 frames — tu peux en extraire 3-8 pour bien couvrir la transition.

3. **Frames faciles (15-20 %) en bonus** : quelques moments de locomotion normale avec les 12 keypoints bien visibles. Sert d'ancre pour le modèle.

#### B.3.3 — Stratégie de labellisation cohérente

Maintenant que tu as tes ~250 frames extraites (kmeans + manuelles), il faut les labelliser dans la GUI DLC. Compte ~1 min par frame → **3-5h pour 200-300 frames** en labellisation soigneuse.

```cmd
python -c "import deeplabcut; deeplabcut.launch_dlc()"
```

Charge ton `config.yaml` → onglet « Label Frames ».

**La règle absolue : cohérence**. Fixe-toi une règle claire pour les cas ambigus et applique-la partout, jusqu'au bout :

- « Patte floue en mouvement → je marque toujours le **centre du flou** »
- « Patte à moitié cachée sous le corps → je marque la **position estimée du carpe**, pas le contour visible »
- « Rearing avec 2 pattes invisibles → je pointe uniquement les visibles, clic droit → « invisible » sur les autres »

**Capture la directionalité** : si tu labellises un rearing vers la gauche avec la patte gauche visible, prends aussi un rearing vers la droite avec la patte droite visible. Sinon le modèle apprend un biais latéral.

**Attention** : une action difficile en temps réel se décompose en **plusieurs frames difficiles distinctes**. Un rearing de 2s = 60 frames, dont peut-être 15 vraiment ambiguës. Traite chacune indépendamment mais avec la même règle.

Note ta règle sur papier au début. **Change pas de règle à mi-parcours** — tu injecterais du bruit dans le training set.

#### B.3.4 — Audit gauche/droite avant le premier entraînement

```cmd
python scripts\dlc_model-training\06_check_labels.py ^
    --config-dir D:\EthoFlow\models\souris-bottomview
```

Audit géométrique qui détecte les frames où left/right paws ont probablement été inversées par erreur. Utile **avant** le premier entraînement (et à re-lancer après chaque round de labellisation). Corrige les inversions détectées dans la GUI DLC avant de passer à B.4.

### B.4 — Premier entraînement

> **`--config-dir` est optionnel partout dans le Parcours B** — ici comme en B.5 et B.6. Sans lui, le script affiche le menu des dossiers de config trouvés sous `D:\EthoFlow\models` et te demande lequel utiliser, comme `--project-dir` ailleurs. Il est écrit en toutes lettres dans les commandes ci-dessous pour qu'elles restent copiables telles quelles.

```cmd
python scripts\dlc_model-training\02_train.py ^
    --config-dir D:\EthoFlow\models\souris-bottomview
```

Fait le split train/test (95/5 par défaut), transfer learning depuis **SuperAnimal-Quadruped** (HRNet-w32 backbone), entraîne 50 epochs. Compte **~2-6h sur GPU 16 GB**.

**Objectif à valider après B.4** — regarde la RMSE de test que DLC imprime :

- **RMSE_pcutoff < 8 px** sur 1024×1080 (~0.8 % de la largeur) : c'est bon, passe à B.5
- **RMSE_pcutoff 8-15 px** : marge d'amélioration, mais le modèle est déjà utilisable pour un premier QC
- **RMSE_pcutoff > 15 px** : problème. Soit tu manques de frames sur des situations spécifiques (cf. B.6 outliers), soit tes labels sont incohérents (relance B.3.4 audit L/R)

**Notes techniques** :

- Recommandation Tony : **ne pas modifier les hyperparamètres**. La tâche (12 keypoints sur souris) n'est pas assez spécifique pour justifier un tuning au-delà des défauts.
- **50 epochs suffit** pour un premier passage : DLC démarre avec les poids pré-entraînés SuperAnimal-Quadruped, seule la tête décodeur pour tes 12 keypoints custom apprend vraiment.
- **Quand bumper** : si la loss train est encore clairement en décroissance à 50 epochs, passe à 100 dans `_config.py` (variable `EPOCHS`). Pour les passes de refinement après B.6, garde 20-30 epochs — c'est du fine-tuning.
- `NET_TYPE = "hrnet_w32"` doit matcher `MODEL_NAME = "hrnet_w32"` sinon size mismatch au chargement des poids pré-entraînés.

### B.5 — Appliquer et QC visuel

```cmd
python scripts\dlc_model-training\03_apply.py ^
    --config-dir D:\EthoFlow\models\souris-bottomview
```

Lance l'inférence sur la vidéo pilote + produit une vidéo annotée à `pcutoff=0.6`. Regarde `D:\EthoFlow\models\souris-bottomview\result-videos\<stem>\<stem>DLC*_labeled.mp4` — tu dois voir les 12 points suivre la souris correctement dans les cas normaux.

Pour voir toutes les prédictions même de basse confiance (utile pour diagnostiquer où le modèle échoue) :

```cmd
python scripts\dlc_model-training\create_labeled_video.py ^
    --config-dir D:\EthoFlow\models\souris-bottomview --pcutoff 0.3
```

Lecture clé — recommandation Tony : à `pcutoff=0.3`, un keypoint peut être **au bon endroit** même à basse confiance. La « confiance » exprime la ressemblance avec le training set : sur un modèle bien entraîné, une confiance à 30 % peut simplement signifier « la patte est floue mais c'est bien la patte, y'a rien d'autre qui lui ressemble dans l'image ». Le vrai problème apparaît quand **plusieurs zones de l'image ont une ressemblance similaire** : le modèle switche entre elles → jitter, télé-portations gauche/droite. C'est ce switch qui trahit un training set incomplet.

### B.6 — Itérations d'amélioration (extraction d'outliers)

Après avoir regardé la vidéo annotée à `pcutoff=0.3`, tu as identifié des situations où le modèle échoue. Il faut extraire de nouvelles frames dans ces situations pour les ajouter au training set.

Deux options — Tony recommande fortement l'**option A (manuelle)** comme levier principal, l'option B (auto DLC) est utile en complément rapide.

#### Option A — Extraction manuelle (recommandé)

Recommandation Tony : « le vrai gain vient du fait que tu vois exactement où le modèle échoue et que tu choisis les frames les plus informatives ».

Workflow :

1. **Regarde toutes tes vidéos analysées** en priorité (pas seulement la pilote). Identifie les patterns d'échec : « les pattes ratent quand elle grimpe le long du mur », « L/R switch pendant les demi-tours rapides », « rearing avec deux pattes cachées mal résolu ».
2. **Budget total : 50-100 nouvelles frames**, réparties entre les situations problématiques identifiées. Tony est explicite là-dessus : c'est un total, **pas un quota par situation**. Le nombre à extraire dépend de combien de situations distinctes posent problème — 3 patterns d'échec → ~20-30 frames chacun ; 8 patterns → ~10 frames chacun.
3. **Extrait les frames à la main** dans les vidéos. Depuis Python (env `dlc` actif) :

   ```python
   import deeplabcut
   deeplabcut.extract_frames(
       r"D:\EthoFlow\models\souris-bottomview\config.yaml",
       mode="manual",
       crop=False,
       userfeedback=False,
   )
   ```

   Une fenêtre s'ouvre par vidéo — utilise le slider + flèches gauche/droite pour le frame-par-frame, clic « Grab frames » sur chaque moment problématique. Même workflow qu'à B.3.2.

4. **Labellise-les** avec la même règle cohérente qu'à B.3.3.
5. **Ré-audit L/R** :
   ```cmd
   python scripts\dlc_model-training\06_check_labels.py ^
       --config-dir D:\EthoFlow\models\souris-bottomview
   ```
6. **Relance l'entraînement** — important : depuis le snapshot précédent, pas from scratch. DLC le fait par défaut si tu ne changes pas d'`iteration` dans la config. Pour un fine-tuning, baisse `EPOCHS` à 20-30 dans `_config.py` :
   ```cmd
   python scripts\dlc_model-training\02_train.py ^
       --config-dir D:\EthoFlow\models\souris-bottomview
   ```

#### Option B — Auto-detect via DLC (complément)

Utile en **premier passage rapide** pour attraper les cas évidents avant ta passe manuelle. Le script `05_refine_outliers.py` fait tourner `dlc.extract_outlier_frames()` sur toutes les vidéos analysées.

```cmd
python scripts\dlc_model-training\05_refine_outliers.py ^
    --config-dir D:\EthoFlow\models\souris-bottomview
```

**Configuration dans ton `_config.py`** — trois paramètres à comprendre :

- `TRAINING_VIDEOS_FOR_REFINE` : liste des vidéos sur lesquelles chercher des outliers (typiquement toutes celles de `ADDITIONAL_VIDEOS` + la pilote — celles déjà analysées par `03_apply.py`).
- `OUTLIER_ALGORITHM` :
  - `"uncertain"` (défaut) — extrait les frames à basse likelihood. Idéal pour cibler les pattes occultées à leur réémergence, les rearing, les moments où le modèle « perd » un keypoint.
  - `"jump"` — extrait les frames avec sauts inter-frame anormaux d'un keypoint. Cible les téléportations L/R et les jitter brutaux.
  - `"fitting"` — modèle ARIMA (plus coûteux, moins ciblé — utilise rarement).
- `OUTLIER_NUMFRAMES` : nombre max de frames extraites **par vidéo** (défaut 30, garde-fou pour éviter d'en extraire des centaines).

Le script :
1. Vérifie que chaque vidéo a bien été analysée par `03_apply.py` (`.h5` doit exister)
2. Met à jour `numframes2pick` dans le `config.yaml` DLC pour matcher `OUTLIER_NUMFRAMES`
3. Lance `dlc.extract_outlier_frames()` sur chaque vidéo, écrit les frames extraites dans `labeled-data\<video>\`
4. Ces frames apparaissent ensuite dans la GUI DLC pour labellisation, comme n'importe quelle autre

**Utilisation type combinée** : lance d'abord Option B pour attraper les 20-30 outliers évidents par vidéo, puis fais un passage manuel (Option A) sur les situations que l'auto-detect n'a pas attrapées mais que tu vois clairement dans les labeled videos.

**Après ta round de labellisation, dans les deux cas, relance 06_check_labels + 02_train** (étapes 5 et 6 de l'Option A).

### B.7 — Enregistrer le modèle final

Une fois satisfait de la précision, ton modèle DLC est à `D:\EthoFlow\models\souris-bottomview\config.yaml`. Depuis là, **tu es à l'Étape 1 du parcours principal** : crée un projet EthoFlow avec `create_project.py --dlc-config D:\EthoFlow\models\souris-bottomview\config.yaml` et enchaîne les étapes 2 à 9.

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
dlc_project_config: D:\EthoFlow\models\souris-bottomview\config.yaml

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
- `sync_from_excel.py` — Excel maître → 1 metadata.yaml par session. Détecte le schéma (1 animal/vidéo vs N animaux/vidéo) depuis les feuilles présentes.
- `patch_captopril.py` — Backfill le champ captopril sans re-syncer (schéma 1 animal / vidéo)

**Multi-animal par vidéo — préparation**
- `calibrate_arenes.py` — GUI pour tracer les rectangles d'arène
- `crop_arenes.py` — Split vidéo brute en N vidéos single-animal (voie B)
- `assign_arenas.py` — Split .h5 DLC multi-animal en N .h5 single-animal par frame (voie A)

**DLC training** (`scripts/dlc_model-training/`) — top-view ou bottom-view, contrôlé via `_config.py`
- `00_init_training_config.py` — Wizard interactif qui crée un `_config.py` custom dans un dossier hors du repo
- `_config.py` — Template versionné (défauts + `DEFAULT_BODYPARTS` + `DEFAULT_SKELETON`)
- `_load_config.py` — Helper commun pour `--config-dir` : flag, ou menu des dossiers de config trouvés (partagé par tous les scripts numérotés)
- `01_setup_project.py` → `06_check_labels.py` — Workflow d'entraînement, tous acceptent `--config-dir` et le demandent s'il manque (voir [Parcours B](#parcours-b--entraîner-un-nouveau-modèle-dlc))
- `create_labeled_video.py` — Régénère la vidéo annotée à un pcutoff différent

**DLC inférence**
- `run_dlc_inference.py` — Inférence DLC (SuperAnimal ou custom)

**DLC → VAME prep**
- `calibrate_scale.py` — Calibre l'échelle px/cm depuis une photo de règle (active la détection de vitesse)
- `pose_cleaning.py` — Module de nettoyage : cutoff + vitesse aberrante + points collants + interpolation + graphe QC
- `prepare_vame_input_custom.py` — Applique le nettoyage complet → `<session>_clean.h5`
- `filter_keypoints.py` — Vire les keypoints non fiables (queue distale, etc.)
- `fill_nan_h5.py` — Impute agressivement les NaN restants
- `rekey_h5.py` — Re-clé un h5 à la convention VAME (`df_with_missing`)
- `reencode_vame_videos.py` — Re-encode H.264/yuv420p pour compat OpenCV

**VAME**
- `run_vame.py` — Orchestre setup/align/trainset/train/evaluate/segment/motif-videos/motif-labels/community/all. `segment --n-clusters N` change le nombre de motifs ; `motif-videos` génère aussi `motif_labels.csv` pré-rempli

**Analyses**
- `analyze_vame.py` — Croise les motifs avec n'importe quelle colonne de ton Excel : CSV + heatmaps + boxplots + stats. `--list-columns` pour voir les axes disponibles, `--group-by` pour choisir, `--cross A B` pour un facteur composite
- `community_dendrogram.py` — Dendrogramme labellisé des motifs
- `inspect_session.py` — QC par session (couverture, gaps)
- `inspect_vame_project.py` — QC d'un projet VAME (.nc files)
- `diagnose_dlc_model.py` — Diagnostique un modèle DLC qui refuse de servir à l'inférence (projet déplacé, shuffle incohérent, jamais entraîné) et répare
- `diagnose_gpu.py` — Diagnostique `torch.cuda.is_available() == False` : build CPU, build CUDA trop ancienne pour la carte, driver absent — et donne la commande de réinstallation adaptée
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

### CUDA non détectée (`torch.cuda.is_available()` → False)

```cmd
conda activate dlc
python scripts\diagnose_gpu.py
```

Le script identifie laquelle des quatre causes s'applique et donne la commande exacte. Il vérifie le driver (`nvidia-smi`), la build torch, l'architecture de la carte, puis tente un vrai calcul GPU.

| Ce qu'il affiche | Cause | Correction |
|---|---|---|
| `torch 2.5.1+cpu` → build CPU | `pip install torch` sert la build CPU par défaut sur Windows. **Le piège n°1.** | réinstaller sur l'index CUDA (commande donnée par le script) |
| build CUDA trop ancienne pour la carte | RTX 50xx = Blackwell/sm_120, absent des builds stables | nightly `cu128` |
| `nvidia-smi` introuvable | driver absent, ou pas de GPU NVIDIA | installer le driver, ou accepter le CPU |
| build correcte mais GPU invisible | `CUDA_VISIBLE_DEVICES`, session RDP, GPU accaparée | vérifier l'env, puis réinstaller |

À la main, si tu préfères :

```cmd
python -c "import torch; print(torch.__version__)"
:: finit par +cpu ? alors :
pip uninstall torch torchvision torchaudio -y
pip install --pre torch torchvision torchaudio ^
    --index-url https://download.pytorch.org/whl/nightly/cu128
python -c "import torch; x = torch.randn(1024,1024,device='cuda'); print(torch.cuda.get_device_name(0))"
```

**Attention à l'environnement.** Chaque env conda a son propre torch : `dlc` peut voir la GPU pendant que `vame` ne la voit pas. Lance le diagnostic dans l'env où tu as le problème, pas dans `base`.

**Sans GPU, le pipeline tourne quand même** — DLC et VAME fonctionnent sur CPU, 10 à 50× plus lentement. Pour l'inférence sur quelques vidéos c'est tenable ; pour entraîner un modèle DLC ou le VAE de VAME, il faut une GPU.

### « Could not find a shuffle with trainingset fraction 0.95 and index 1 »

**Lance d'abord le diagnostic** — il vérifie tout et répare ce qui est réparable :

```cmd
:: Interactif — menu des modèles trouvés sous D:\EthoFlow\models
python scripts\diagnose_dlc_model.py

:: Sur un modèle précis, avec réparation automatique
python scripts\diagnose_dlc_model.py ^
    --model-dir D:\EthoFlow\models\souris-bottomview-Leo-2026-06-05 --fix

:: Ou depuis un projet EthoFlow (lit dlc_project_config tout seul)
python scripts\diagnose_dlc_model.py --project-dir D:\EthoFlow\projects\mon-projet
```

Trois causes possibles, toutes détectées en amont par le script.

**1. Le projet DLC a été déplacé.** C'est le cas le plus fréquent quand un modèle change de disque ou de dossier. DLC stocke un chemin absolu dans `project_path` du `config.yaml` ; s'il pointe vers l'ancien emplacement, DLC cherche `dlc-models-pytorch/` là-bas et ne trouve rien — alors que le modèle est bien entraîné. Le script détecte et **corrige automatiquement** :

```
⚠  Le projet DLC a été déplacé :
     config.yaml déclare : E:\LEO\dlc-projects\souris-bottomview-Leo-2026-06-05
     emplacement réel    : D:\EthoFlow\models\souris-bottomview-Leo-2026-06-05
✓ project_path corrigé automatiquement dans config.yaml
```

**2. `iteration` ou `TrainingFraction` ne correspondent à aucun shuffle existant.** Arrive après un refinement partiel (`iteration` incrémenté sans re-entraîner). Le script liste les shuffles réellement présents pour que tu corriges le `config.yaml`.

**3. Le modèle n'a jamais été entraîné.** Le script t'oriente selon l'état réel :

- **Aucune frame extraite** → reprends au [Parcours B](#parcours-b--entraîner-un-nouveau-modèle-dlc) depuis `01_setup_project.py`
- **Frames extraites mais pas labellisées** → labellise dans la GUI (`deeplabcut.launch_dlc()`), puis `02_train.py`
- **Frames labellisées, entraînement pas lancé** → `02_train.py --config-dir <dossier du modèle>`

Si tu voulais en fait utiliser un **autre** modèle déjà entraîné, corrige le pointeur :

```yaml
# <project>/configs/pipeline_config.yaml
dlc_project_config: D:\EthoFlow\models\<modele-entraine>\config.yaml
```

### `run_vame.py setup` : "Config file is not found"

Corrigé — si tu le vois encore, c'est que ta copie du repo est antérieure.

`create_project.py` crée le squelette du projet, `data/vame/` compris, vide. `vame.init_new_project` ne teste que l'existence du dossier :

```python
if project_path.exists():
    return projconfigfile, read_config(projconfigfile)
```

Il prenait donc ce dossier vide pour un projet déjà initialisé, sautait la création, et plantait en lisant un `config.yaml` jamais écrit. `setup` efface maintenant le dossier s'il est vide (rien à y perdre, VAME le recrée) ; un vrai projet VAME reste protégé et demande `--force`. Contournement sur une ancienne copie : supprime `<project>/data/vame/` à la main avant de relancer `setup`.

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
