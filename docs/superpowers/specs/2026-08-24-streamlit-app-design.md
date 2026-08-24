# Design — App Streamlit EthoFlow : front door du parcours principal

Date : 2026-08-24
Statut : approuvé pour implémentation

---

## 1. Contexte

`streamlit_app/` n'a pas été touché depuis le commit `881351a`. Les scripts,
eux, ont beaucoup bougé : passage projet-aware (`--project-dir`), fusion du
dossier VAME dans le projet, remplacement des labels YAML par un CSV, prompts
interactifs partout, nouveaux scripts (`prepare_vame_input_custom`,
`diagnose_dlc_model`, `calibrate_scale`). L'app est restée sur l'ancien modèle
et ne peut plus rien lancer.

Ce document décrit la remise à niveau : l'app devient la **porte d'entrée du
parcours principal du README (étapes 1 → 9)**, le terminal ne servant plus que
pour le Parcours B (entraînement d'un modèle DLC).

## 2. Objectifs

1. Piloter les étapes 1 à 9 du README depuis l'interface, sans jamais taper de
   commande.
2. Exécuter les scripts de façon fiable : jamais de blocage sur une invite,
   logs en direct, jobs qui survivent à la navigation.
3. Refléter le modèle de données actuel des scripts (projet auto-suffisant,
   VAME à plat dans le projet, `motif_labels.csv`).
4. Rendre le travail sur les vidéos concret : visionnage avant traitement, QC
   visuel du tracking, génération de visualisations à la demande.
5. Calibrer les arènes et l'échelle px/cm au clic, dans le navigateur.

## 3. Non-objectifs

- **Le Parcours B (entraînement d'un modèle DLC) reste entièrement hors de
  l'app.** Voir §7 : l'app *importe* un modèle produit ailleurs, elle n'en
  fabrique pas.
- Pas de refonte UX de la page de labellisation. Seule la migration de format
  est faite, parce qu'elle est nécessaire à la correction.
- Pas de file d'attente de jobs, pas de multi-utilisateur, pas
  d'authentification, pas d'exécution sur une machine distante.
- Pas de réécriture des scripts CLI. Ils restent utilisables tels quels et
  restent la seule autorité sur les formats de fichiers.

## 4. État des lieux — ce qui est cassé

Constats vérifiés dans le code, qui motivent chaque partie du design.

| # | Constat | Preuve |
|---|---|---|
| 1 | Aucun script ne reçoit `--project-dir`. Tous appellent `resolve_project()` qui tombe sur `input()`. Le subprocess se fige indéfiniment. | `lib/pipeline.py:33-44`, `scripts/paths.py:119-126` |
| 2 | `conda run --no-capture-output` combiné à `subprocess.run(capture_output=True)` : la sortie part au terminal, `result.stdout` est vide. | `lib/pipeline.py:34`, `lib/pipeline.py:39-44` |
| 3 | Les 3 scripts câblés (`filter_keypoints`, `fill_nan_h5`, `trim_empty_arena`) sont les 3 seuls non projet-aware et ne font plus partie du parcours du README. L'étape 6b réelle n'est pas exposée. | `lib/pipeline.py:74-148` vs README §Étape 6 |
| 4 | `analyze_vame` est lancé dans l'env `ethoflow`, qui n'a ni matplotlib ni scipy. | `lib/pipeline.py:189`, `analyze_vame.py:71-76`, `environment-pipeline.yml` |
| 5 | `prepare_vame_input_custom` importe `deeplabcut` : il lui faut l'env `dlc`. | `prepare_vame_input_custom.py:130` |
| 6 | L'app cherche les projets VAME sous `~/Inserm/vame-projects`. Le layout réel est plat : `<projet>/data/vame/` *est* le projet VAME. | `lib/config.py:37`, `paths.py:149-161`, `analyze_vame.py:108` |
| 7 | La labellisation écrit `analysis/motif_labels_<algo>.yaml`. Le format lu par les analyses est `<vame>/motif_labels.csv`, `;`-séparé, 8 colonnes. | `views/label_motifs.py`, `run_vame.py:634-637,712`, `analyze_vame.py:1230` |
| 8 | Le dashboard crée les projets par `mkdir` : pas de `pipeline_config.yaml`, pas de `--kind`, pas d'Excel starter. `create_project.py` n'est jamais appelé. | `lib/config.py:125-131` |
| 9 | `results.py` cherche `heatmap_usage.png` et `mean_by_condition*.png` ; les vrais noms sont `heatmap_usage_by_<colonne>.png`, `usage_by_category.csv`, `stats_by_motif_*.csv`. | `views/results.py:116-136` vs README §Étape 9 |
| 10 | `sessions.py` code en dur des colonnes `Stress`/`ANGII` alors que l'Excel est à colonnes libres ; le statut VAME teste `vame_dir()/<session>`, chemin qui n'existe pas. | `lib/sessions.py:95,109-110` |
| 11 | Racine des projets divergente : `Path.home()/"ethoflow"/"projects"` dans l'app, `D:\EthoFlow\projects` dans les scripts. | `lib/config.py:34`, `interactive.py:38` |
| 12 | Deux référentiels ETHOGRAM divergents : 9 catégories dans l'app, 8 dans les scripts. | `lib/config.py:149-185`, `run_vame.py:640-644` |
| 13 | Étapes absentes de l'app : `create_project`, `excel_templates`, `calibrate_arenes`, `crop_arenes`, `calibrate_scale`, `prepare_vame_input_custom`, `assign_arenas`, `diagnose_dlc_model`, `motif_gif`, `behavior_structure_gif`, `community_dendrogram`, et les options `--group-by` / `--cross` / `--extended` d'`analyze_vame`. | — |

## 5. Architecture d'exécution

### 5.1 Contrat de construction des commandes

Toute commande lancée par l'app respecte trois règles, sans exception :

1. **`--project-dir <chemin absolu du projet courant>` est toujours passé.**
   `resolve_project()` retourne alors immédiatement et aucune invite ne
   s'ouvre.
2. **`--no-prompt` est toujours passé.** `add_project_dir_arg()` l'injecte dans
   tous les scripts projet-aware. Un argument manquant produit une erreur
   franche et un code retour non nul, jamais une attente silencieuse.
3. **Tous les paramètres que le script demanderait à l'invite sont fournis
   explicitement.** En particulier les seuils de `prepare_vame_input_custom`
   (`--likelihood-threshold`, `--max-speed`, `--px-per-cm`) : l'interface les
   présente avec les mêmes explications et les mêmes défauts que le script.

Les trois scripts non projet-aware (`filter_keypoints`, `fill_nan_h5`,
`trim_empty_arena`) reçoivent leurs chemins explicitement, dérivés du projet
courant.

### 5.2 Table des environnements conda

Le mapping script → env est **une donnée déclarée en un seul endroit**, dans
`lib/pipeline.py`. Se tromper d'env produit un `ImportError` après plusieurs
minutes d'attente ; la table évite d'y penser à chaque appel. Elle couvre tous
les scripts du dépôt, y compris ceux que l'interface n'expose pas — ce qui est
exposé et où est défini en §8.

| Env | Scripts |
|---|---|
| `ethoflow` | `create_project`, `excel_templates`, `sync_from_excel`, `crop_arenes`, `assign_arenas`, `inspect_session`, `diagnose_dlc_model`, `motif_gif`, `post_process_cropped`, `filter_keypoints`, `fill_nan_h5`, `trim_empty_arena`, `rekey_h5` |
| `dlc` | `run_dlc_inference`, `prepare_vame_input_custom` |
| `vame` | `run_vame`, `analyze_vame`, `behavior_structure_gif`, `community_dendrogram`, `inspect_vame_project`, `reencode_vame_videos` |

`calibrate_arenes` et `calibrate_scale` ne sont pas lancés en subprocess : leurs
fonctions d'écriture sont importées directement (§9).

### 5.3 Le runner

Nouveau module `lib/runner.py`. Un *job* est l'exécution d'un script.

**Persistance.** L'état d'un job vit sur disque, sous
`<projet>/.ethoflow/jobs/` :

- `<job_id>.log` — stdout et stderr fusionnés, écrits au fil de l'eau, sans
  tampon de bloc.
- `<job_id>.json` — `{script, env, argv, started_at, ended_at, returncode, pid,
  state}`.

Conséquence recherchée : naviguer entre les pages, rafraîchir le navigateur ou
fermer l'onglet n'interrompt rien et ne perd aucun log. Un `session_state`
Streamlit ne survit à aucun de ces trois événements.

**États.** `running` → `succeeded` | `failed` | `cancelled`. Un job dont le
processus n'existe plus alors que le JSON dit `running` (app tuée pendant
l'exécution) est marqué `interrupted` à la relecture.

**Exécution.** `subprocess.Popen` avec `stdout=PIPE, stderr=STDOUT`, lu ligne à
ligne par un thread démon qui écrit dans le `.log`. `conda run -n <env>` **sans**
`--no-capture-output`, sinon rien ne remonte (constat n°2).

**Verrou.** Un seul job à la fois par projet, matérialisé par
`<projet>/.ethoflow/jobs/current.lock`. DLC et VAME veulent tous les deux le
GPU ; les laisser tourner ensemble est une source de plantage, pas un gain. Un
second lancement affiche quel job occupe la place, avec un lien vers ses logs.

**Annulation.** Terminaison de l'arbre de processus (le processus visible est
`conda run`, pas le Python qui travaille — il faut descendre aux enfants).

**Affichage.** Un composant partagé `views/_job.py` rend le job courant :
état, durée, les N dernières lignes du log en direct, log complet dépliable,
bouton Annuler, et le lien vers les artefacts produits une fois terminé.
Ce composant est présent sur toutes les pages qui lancent quelque chose.

### 5.4 `lib/pipeline.py` réduit à la construction d'argv

Une fonction par script, qui retourne `(env, script_name, argv)` — **sans rien
exécuter**. Le runner exécute. Bénéfice direct : la construction des commandes
devient testable sans conda, sans GPU et sans données (§12).

## 6. Modèle de données

### 6.1 Chemins

`lib/config.py` continue de déléguer à `scripts/paths.py`, qui reste la source
unique de vérité. Deux corrections :

- `DEFAULT_PROJECTS_ROOT` et `DEFAULT_MODELS_ROOT` sont importés de
  `scripts/interactive.py` au lieu d'être redéfinis (constat n°11). La page
  Configuration permet de les surcharger, et la surcharge est **persistée dans
  `~/.ethoflow/app_prefs.yaml`** — un `session_state` serait reperdu à chaque
  redémarrage de l'app. Ce fichier ne contient que des préférences d'interface
  (racines, dernier projet ouvert) ; il n'est jamais lu par les scripts.
- Les alias figés `DATA_ROOT`, `RAW_DIR`, `CROPPED_DIR`, `DLC_OUTPUT_DIR`,
  `VAME_DIR` (`lib/config.py:96-100`) sont supprimés. Ils pointent sur la racine
  du dépôt, jamais sur le projet courant : ce sont des pièges.

### 6.2 VAME à plat

`vame_projects_root()`, `discover_projects()`, `get_current_project_path()` et
toute lecture de `.vame_config_path` **disparaissent de l'app**. Il y a un
projet VAME par projet EthoFlow, à `vame_dir(project)`, et il n'y a donc rien à
sélectionner.

`lib/vame_projects.py` devient `lib/vame.py`. Chaque fonction prend le projet
EthoFlow et dérive `vame_dir(project)` elle-même. La logique de détection des
algos (`hmm-15`, `kmeans-25`) et d'agrégation des `motif_usage_*.npy` est
conservée : elle est correcte, seule sa racine change.

### 6.3 `motif_labels.csv`

Nouveau `lib/motif_labels.py`, seul point d'accès au fichier
`<projet>/data/vame/motif_labels.csv`.

- Format : séparateur `;`, encodage `utf-8-sig`, colonnes de
  `run_vame.MOTIF_LABELS_COLUMNS` (`motif_id`, `label`, `category`,
  `confidence`, `qc_inspected_sessions`, `notes`, `usage_pct`, `video`).
- **Les colonnes ajoutées à la main par l'utilisateur sont préservées** au
  round-trip. Quelqu'un qui ajoute une colonne `observateur` dans Excel ne doit
  pas la perdre parce qu'il a cliqué dans l'app.
- L'app ne fabrique pas le fichier à partir de rien : c'est
  `run_vame motif-videos` ou `run_vame motif-labels` qui le génère, avec
  `usage_pct` et `video` pré-remplis, et l'app propose le bouton correspondant
  s'il est absent. Les deux seules écritures faites par l'app sont la mise à
  jour de lignes existantes, et la reprise d'un ancien YAML décrite ci-dessous.
- La colonne `video` contient un chemin relatif au projet VAME, déjà résolu par
  le script. L'app le lit au lieu de deviner l'emplacement des clips — ce que
  fait aujourd'hui `find_any_motif_video()` avec trois motifs de glob successifs.

**Migration.** Si un `analysis/motif_labels_<algo>.yaml` de l'ancienne app
existe, la page Motifs propose une reprise en un clic vers le CSV, et ne
touche à rien sans confirmation.

### 6.4 Metadata génériques

`lib/sessions.py` cesse de connaître des colonnes à l'avance. `Stress` et
`ANGII` disparaissent (constat n°10). Le tableau des arènes et les détails de
session affichent **toutes** les clés présentes dans le `metadata.yaml`, dans
l'ordre du fichier. C'est le pendant nécessaire de « toutes les colonnes de ton
Excel sont recopiées telles quelles » (README §Étape 9) : une app qui n'affiche
que 6 colonnes connues casse la promesse.

Le statut VAME d'une session teste le vrai artefact
(`data/vame/results/<session>/…/*_label_*.npy`).

## 7. Modèle DLC : import, jamais entraînement

Le modèle DLC vit **hors** du projet EthoFlow, il est produit ailleurs (Parcours
B, au terminal, ou fourni par le labo) et **jamais copié**. L'app ne fait que le
désigner.

Une section **« Modèle DLC »** sur la page Projet :

1. Affiche le `dlc_project_config` courant lu dans
   `configs/pipeline_config.yaml`, ou l'absence de configuration.
2. Propose un menu des modèles trouvés sous `DEFAULT_MODELS_ROOT`
   (`D:\EthoFlow\models`), plus la saisie d'un chemin libre — même logique que
   le menu interactif de `run_dlc_inference.py`.
3. Écrit le choix dans `configs/pipeline_config.yaml` via
   `create_project.py --force --dlc-config <path>`, la voie documentée par le
   README.
4. Bouton **Diagnostiquer** qui lance `diagnose_dlc_model.py --project-dir …`,
   et **Réparer** qui ajoute `--fix`. C'est ce qui répond à l'erreur
   « Could not find a shuffle with trainingset fraction… » du Troubleshooting,
   typiquement causée par un modèle déplacé.

L'app n'expose **aucun** des scripts de `scripts/dlc_model-training/`. La page
À propos renvoie vers la section Parcours B du README pour qui doit entraîner un
modèle.

## 8. Navigation

La sidebar suit l'ordre du README. Un projet doit être ouvert pour que les
pages autres que Projet et Configuration soient accessibles.

| Page | Étapes | Contenu |
|---|---|---|
| **Projet** | 0-1 | Ouvrir / créer (via `create_project.py`, avec `--kind`), section Modèle DLC (§7), inventaire des sessions et de leur avancement |
| **Données** | 2-3 | Excel starter (téléchargement, ouverture, dépôt d'une version remplie), `sync_from_excel` avec aperçu `--dry-run` avant écriture |
| **Vidéos & calibration** | 4, 6a | Navigateur vidéo (§9.1), calibration des arènes au clic, `crop_arenes`, calibration px/cm au clic |
| **Pose (DLC)** | 5 | `run_dlc_inference` : mode, `--video-adapt`, `--skip-existing`, sélection de sessions |
| **Nettoyage** | 6b | `prepare_vame_input_custom` avec ses 4 passes expliquées, `assign_arenas` en multi-animal, galerie QC (§9.2), repli « Outils avancés » pour les 3 scripts legacy |
| **VAME** | 7 | `setup` → `align` → `trainset` → `train` → `evaluate` → `segment` en stepper montrant ce qui est déjà fait |
| **Motifs** | 8 | `motif-videos` / `motif-labels`, puis labellisation sur le vrai CSV |
| **Analyses** | 9 | `analyze_vame` : découverte des axes via `--list-columns`, `--group-by`, `--cross`, `--extended` ; puis explorateur de résultats |
| **Visualisations** | 9 | `motif_gif`, `behavior_structure_gif`, `community_dendrogram` |
| **Configuration** | — | Racines projets/modèles, vérification des 3 envs conda |
| **À propos** | — | Renvoi au README, dont le Parcours B |

**Anatomie commune à toute page d'étape**, pour que le chercheur sache toujours
où il en est :

1. *Ce que fait cette étape* — texte repris du README, jamais réinventé.
2. Le formulaire des paramètres, avec les mêmes défauts que le CLI.
3. Le bouton de lancement.
4. Le job en direct (`views/_job.py`).
5. Les artefacts produits, avec leur chemin.

Une page dont l'étape amont n'a rien produit grise ses actions et dit quoi
lancer avant. Un bouton actif qui échoue systématiquement est pire qu'un bouton
grisé qui explique.

### 8.1 Page Analyses — découverte des axes

`analyze_vame.py --list-columns` liste les colonnes exploitables avec leur
nombre de groupes et de sessions. L'app le lance et **transforme sa sortie en
sélecteurs** : cases à cocher pour `--group-by`, paires pour `--cross`, avec le
nombre de sessions par groupe affiché à côté de chaque option. C'est la partie
de l'étape 9 la plus laborieuse au terminal (relire une sortie texte, retaper
des noms de colonnes) et celle où une interface apporte le plus.

## 9. Interaction vidéo

### 9.1 Visionner avant traitement

Pour chaque session : vignette de la première frame, lecteur vidéo, et les
caractéristiques **réelles** lues dans le fichier avec OpenCV (fps, nombre de
frames, durée, dimensions), confrontées à ce que déclare le `metadata.yaml`. Un
écart fps réel / fps déclaré fausse toutes les conversions frames → secondes en
aval, et se voit ici avant de coûter une inférence.

Si `source_video` pointe dans le vide, l'app propose de **re-pointer le dossier
des vidéos** et de réécrire les metadata concernées, plutôt que d'afficher un
avertissement sans issue. Ce cas est documenté au Troubleshooting du README
(« Metadata avec chemins Windows sur machine Linux ») et se produit à chaque
changement de machine ou de disque.

### 9.2 QC du tracking

- Le `_labeled.mp4` produit par DLC et la vidéo brute côte à côte.
- Galerie des graphes `data/dlc-output/_qc_trajectories/<session>_<keypoint>.png`
  produits par l'étape 6b, avec sélecteur de keypoint et bouton pour en
  régénérer un autre (`--qc-bodypart`). Le critère est celui de Tony : aucun
  saut anormal sur la trajectoire.
- Rappel du résumé de l'étape 6b (% de frames utilisables, frames réparées) et
  du seuil d'alerte du README : au-delà de 10-15 % de frames réparées, le
  problème est le modèle DLC, pas le post-traitement.

### 9.3 Visualisations à la demande

Formulaires pour `motif_gif.py` (session, début, durée, format),
`behavior_structure_gif.py` (dont `--pool-all-sessions`, `--with-video`,
`--projection`) et `community_dendrogram.py` (`--group`, `--linkage`). Le rendu
s'affiche dans l'app dès la fin du job, et les rendus précédents restent listés
avec leurs paramètres.

## 10. Calibration dans le navigateur

Ajout de `streamlit-image-coordinates` à `environment-pipeline.yml` et
`requirements-pipeline.txt`.

**Arènes.** Extraction d'une frame de la vidéo choisie, clic sur deux coins
opposés par arène, aperçu qui redessine les rectangles nommés `A1`…`A4`
par-dessus la frame après chaque clic. Ajustement fin au pixel possible après
le clic, pour ne pas avoir à recliquer parfaitement.

**Échelle px/cm.** Clic sur les deux extrémités d'une distance connue, saisie de
la distance en cm, calcul de `px_per_cm`. La saisie directe d'une valeur déjà
connue reste possible (équivalent de `--set`).

**Contrainte de conception.** L'app n'invente aucun format de fichier. Elle
importe `calibrate_arenes.save_coords_default()` et
`calibrate_scale.write_scale()` et les appelle. Les scripts restent la seule
autorité sur ce qui est écrit dans `pipeline_config.yaml`, et les versions CLI
avec fenêtre OpenCV continuent de fonctionner sans modification. Deux
implémentations concurrentes du même format divergeraient au premier changement.

## 11. ETHOGRAM : catégories et vocabulaire

Deux choses distinctes qui étaient mélangées :

- **Les catégories** — liste fermée de 8 valeurs, écrite dans la colonne
  `category` du CSV et utilisée par les analyses pour grouper. Source de vérité
  unique : `run_vame.ETHOGRAM_CATEGORIES`. `lib/config.py` l'importe au lieu
  d'en garder une copie divergente (constat n°12). Présentée en liste
  déroulante : une valeur hors liste casse le groupement par catégorie.
- **Le vocabulaire suggéré** — le dictionnaire riche de `lib/config.py:149-185`,
  conservé, mais requalifié en **exemples pour aider à remplir le champ `label`**,
  qui est libre. Il aide à écrire `grooming_face` plutôt que `toilettage tête`
  et à rester cohérent d'un motif à l'autre, sans jamais contraindre.

## 12. Tests

`tests/` ne contient qu'un squelette. On ajoute des tests qui tournent sans
conda, sans GPU et sans données réelles :

- **Construction des argv**, pour chaque script exposé : `--project-dir` présent
  et absolu, `--no-prompt` présent, env correct selon la table §5.2, options
  correctement traduites depuis les paramètres de l'interface.
- **Round-trip `motif_labels.csv`** : séparateur, encodage, ordre des colonnes,
  et préservation d'une colonne inconnue ajoutée à la main.
- **Résolution des chemins** contre une arborescence de projet factice, y
  compris les cas dégradés (projet vide, VAME absent, sessions sans metadata).
- **Machine à états du runner** avec une commande bidon : succès, échec,
  annulation, et détection d'un job `running` dont le processus a disparu.

Corollaire architectural : **les vues restent minces**. Tout ce qui n'est pas du
placement de widgets vit dans `lib/` et est testable. Une logique enfouie dans
`views/` est une logique non testée.

## 13. Ce qui est supprimé

| Élément | Raison |
|---|---|
| `vame_projects_root`, `DEFAULT_VAME_PROJECTS_ROOT`, `discover_projects`, `get_current_project_path`, lecture de `.vame_config_path` | Le layout VAME est plat dans le projet ; il n'y a rien à découvrir ni à sélectionner |
| `lib/config.create_project()` (le `mkdir` maison) | Remplacé par un appel à `scripts/create_project.py` |
| Alias `DATA_ROOT`, `RAW_DIR`, `CROPPED_DIR`, `DLC_OUTPUT_DIR`, `VAME_DIR` | Figés sur la racine du dépôt, jamais sur le projet courant |
| `lib/labels.py` (YAML par algo) | Remplacé par `lib/motif_labels.py` (CSV), avec reprise proposée à l'utilisateur |
| `run_in_env()` et les wrappers exécutants de `lib/pipeline.py` | Remplacés par la construction d'argv + le runner |
| Colonnes `Stress` / `ANGII` en dur | L'Excel est à colonnes libres |

Le fichier `.vame_config_path` à la racine du dépôt n'est pas supprimé : les
scripts s'en servent encore. L'app cesse simplement de le lire.

## 14. Ordre d'implémentation

L'ordre n'est pas libre : tant que §5 et §6 ne sont pas en place, aucune page ne
peut fonctionner, et toute page écrite avant eux serait à réécrire.

1. **Socle** — runner, table des envs, construction d'argv, modules `lib/vame.py`
   et `lib/motif_labels.py`, correction des chemins. Rien de visible, mais tout
   en dépend.
2. **Parcours minimal vérifiable** — Projet (dont import du modèle DLC),
   Données, Pose. À ce stade on peut mener un projet de la création à
   l'inférence, ce qui valide le socle sur des jobs réels.
3. **Le reste du parcours** — Nettoyage, VAME, Motifs, Analyses.
4. **Vidéo et calibration** — navigateur vidéo, QC, calibration au clic,
   visualisations à la demande.

## 15. Risques

- **`environment-vame.yml` ne déclare que `vame-py`.** matplotlib, scipy, umap
  et scikit-learn arrivent en dépendances transitives. Si l'une manque,
  `analyze_vame` ou `behavior_structure_gif` échouent à l'import. La page
  Configuration sonde chaque env avec un import test et le signale **avant** un
  job de plusieurs heures, plutôt qu'après.
- **`conda run` et la remontée des logs.** Sans `--no-capture-output` la sortie
  est bien capturée, mais Python tamponne sa sortie quand elle n'est pas un
  terminal : les logs arriveraient par blocs. `PYTHONUNBUFFERED=1` est donc
  positionné dans l'environnement du subprocess. À vérifier sur un job réel.
- **Chemins Windows sur machine de développement macOS.** Les défauts des
  scripts sont des chemins `D:\`. Le développement et les tests se font sur des
  arborescences factices ; la vérification finale doit avoir lieu sur la machine
  Windows de production.
