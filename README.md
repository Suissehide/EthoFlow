# EthoFlow — pipeline d'analyse comportementale souris

Pipeline d'analyse comportementale de souris basé sur **DeepLabCut** (estimation de pose) et **VAME** (segmentation comportementale), avec une interface web **Streamlit** pour faciliter l'utilisation par les chercheurs.

## Architecture

```
Vidéo brute → Crop des 4 arènes → Inférence DLC → Analyse VAME → Résultats
                                       ↑
                                Modèle DLC entraîné
```

Les chercheurs interagissent avec le pipeline via l'interface Streamlit ; le code Python tourne en local sur le poste de calcul Windows.

## Quick start (local, sans GPU)

Pour tester l'interface et la structure du pipeline sans installer DLC/VAME (qui nécessitent une GPU sérieuse) :

```bash
# 1. Créer l'environnement utilitaires
conda env create -f environment-pipeline.yml
conda activate ethoflow

# 2. Lancer l'interface
streamlit run streamlit_app/app.py
```

L'app s'ouvre sur http://localhost:8501.

## Quick start (Docker)

Alternative sans avoir à installer conda :

```bash
docker compose -f docker/docker-compose.yml up --build
```

## Installation complète (poste de production)

Voir [`docs/ETHOFLOW.md`](docs/ETHOFLOW.md) pour la procédure complète : installation des drivers GPU, des trois environnements conda, configuration du modèle DLC et de VAME.

## Structure du repo

```
ethoflow/
├── README.md                       # ce fichier
├── environment-pipeline.yml        # env conda pour scripts + UI
├── environment-dlc.yml             # env conda pour DeepLabCut
├── environment-vame.yml            # env conda pour VAME
├── requirements-pipeline.txt       # deps pour Docker / pip
├── .gitignore
│
├── docs/
│   └── ETHOFLOW.md                 # doc utilisateur et technique complète
│
├── streamlit_app/
│   └── app.py                      # interface web
│
├── scripts/
│   ├── crop_arenes.py              # crop des 4 arènes par vidéo
│   ├── run_dlc_inference.py        # inférence DLC sur vidéos croppées
│   ├── run_vame.py                 # analyse VAME
│   └── run_pipeline.py             # orchestrateur
│
├── configs/
│   ├── pipeline_config.yaml.example
│   └── metadata_template.yaml      # template à copier pour chaque session
│
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
│
├── tests/
│   └── test_skeleton.py
│
└── data/                           # gitignored — données locales
    ├── raw/                        # vidéos brutes + metadata.yaml
    ├── cropped/                    # vidéos après crop des arènes
    ├── dlc-output/                 # sorties DLC (.h5, .csv)
    ├── vame-output/                # sorties VAME
    └── results/                    # figures et métriques
```

## Workflow type

1. Le chercheur enregistre une vidéo et la dépose dans `data/raw/<session_id>/`
2. Il remplit `metadata.yaml` (animal IDs, conditions, coords des arènes) — possible via l'UI Streamlit
3. Il lance le pipeline depuis l'UI
4. Les résultats apparaissent dans `data/dlc-output/` et `data/vame-output/`

## Liens utiles

- DeepLabCut : https://deeplabcut.github.io/DeepLabCut/
- VAME : https://github.com/LINCellularNeuroscience/VAME
- Streamlit : https://streamlit.io/

## Licence

À définir avec le labo.
