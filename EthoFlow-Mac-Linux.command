#!/usr/bin/env bash
# EthoFlow — double-clic pour installer (la première fois) et lancer l'app.
# macOS : double-clic dans le Finder. Linux : `./EthoFlow-Mac-Linux.command` une
# première fois dans un terminal ; EthoFlow apparaît ensuite dans le menu
# des applications. (Windows : EthoFlow-Windows.bat.)
#
# À chaque lancement :
#   1. trouve conda — et propose d'installer Miniforge s'il n'y en a pas ;
#   2. crée les environnements absents (ethoflow, dlc, vame), met à jour
#      ceux dont le fichier environment-*.yml a changé (git pull), et
#      complète l'env ethoflow s'il lui manque une dépendance ;
#   3. vérifie que torch voit la GPU NVIDIA dans dlc et vame, lance
#      `diagnose_gpu.py --fix` sinon (une fois par environnement) ;
#   4. lance l'app (scripts/lancer_app.sh) : elle s'ouvre dans le navigateur.
set -uo pipefail

ICI="$(cd -P "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ICI"

# Modules indispensables à l'app : s'il en manque un, l'env ethoflow est
# complété depuis environment-pipeline.yml avant le lancement.
IMPORTS_APP="streamlit, streamlit_image_coordinates, cv2, pandas, numpy, yaml, tqdm, openpyxl, tables"

# Présentation : une étape numérotée, puis une ligne par action avec un
# statut aligné en colonne (OK, INFO, ATTENTION, …).
ETAPE=0
etape() { ETAPE=$((ETAPE + 1)); printf '\n \033[1m[%d/4] %s\033[0m\n' "$ETAPE" "$1"; }
ligne() {
    local couleur=''
    case "$1" in OK) couleur='\033[32m' ;; ATTENTION) couleur='\033[33m' ;; esac
    printf "       ${couleur}%-10s\033[0m%s\n" "$1" "$2"
}
fin_erreur() {
    printf '\n \033[31mERREUR\033[0m  %s\n\n' "$1" >&2
    read -r -p "Appuie sur Entrée pour fermer." _
    exit 1
}

# ---------------------------------------------------------------- conda
trouver_conda() {
    local c
    for c in "${CONDA_EXE:-}" "$(command -v conda 2>/dev/null)" \
             "$HOME"/{miniforge3,mambaforge,miniconda3,anaconda3}/bin/conda \
             /opt/{miniforge3,miniconda3,anaconda3,conda}/bin/conda \
             /opt/homebrew/Caskroom/{miniforge,miniconda}/base/bin/conda \
             /usr/local/Caskroom/{miniforge,miniconda}/base/bin/conda; do
        if [ -n "$c" ] && [ -x "$c" ]; then echo "$c"; return 0; fi
    done
    return 1
}

printf '\n \033[1mEthoFlow\033[0m%42s%s\n' "" "$(date '+%d/%m/%Y %H:%M')"
printf ' %s\n' "=================================================================="

etape "Conda"
if ! CONDA="$(trouver_conda)"; then
    ligne INFO "conda introuvable - EthoFlow en a besoin pour ses environnements Python"
    read -r -p "       Installer Miniforge dans $HOME/miniforge3 (~100 Mo, sans droits admin) ? [O/n] " rep
    case "$rep" in [nN]*) fin_erreur "installation annulée." ;; esac
    installeur="$(mktemp -t miniforge.XXXXXX).sh"
    url="https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
    curl -fL -o "$installeur" "$url" || fin_erreur "téléchargement impossible ($url)."
    bash "$installeur" -b -p "$HOME/miniforge3" || fin_erreur "l'installation de Miniforge a échoué."
    rm -f "$installeur"
    CONDA="$HOME/miniforge3/bin/conda"
fi
export CONDA_EXE="$CONDA"   # scripts/lancer_app.sh s'en sert pour trouver l'env
BASE="$(dirname "$(dirname "$CONDA")")"
ligne OK "$CONDA"

# ------------------------------------------------------- environnements
# Une copie du .yml est gardée dans l'env : s'il diffère (après un git
# pull), l'env est mis à jour. Un env créé à la main, sans copie, est
# laissé tel quel — une mise à jour pourrait y remplacer un torch réparé.
preparer_env() {   # nom fichier.yml obligatoire(0/1)
    local nom="$1" yml="$2" obligatoire="$3" prefixe="$BASE/envs/$1"
    local copie="$prefixe/.ethoflow-env.yml"
    if [ ! -d "$prefixe" ]; then
        ligne ... "$nom : création (10 à 20 min, une seule fois)"
        if ! "$CONDA" env create -f "$yml" -q; then
            # Un env à moitié créé passerait pour installé au prochain lancement.
            "$CONDA" env remove -n "$nom" -y >/dev/null 2>&1
            [ "$obligatoire" = 1 ] && fin_erreur "impossible de créer l'environnement « $nom »."
            ligne ATTENTION "$nom : création impossible - les étapes qui en dépendent seront indisponibles"
            return
        fi
        cp "$yml" "$copie"
        ligne OK "$nom (créé)"
    elif [ ! -f "$copie" ]; then
        cp "$yml" "$copie"
        ligne OK "$nom"
    elif ! cmp -s "$yml" "$copie"; then
        ligne ... "$nom : mise à jour ($yml a changé)"
        if "$CONDA" env update -n "$nom" -f "$yml" --prune -q; then
            cp "$yml" "$copie"
            rm -f "$prefixe/.ethoflow-gpu-ok"   # torch a pu changer : revérifier
            ligne OK "$nom (mis à jour)"
        else
            ligne ATTENTION "$nom : mise à jour échouée - l'ancienne version est gardée"
        fi
    else
        ligne OK "$nom"
    fi
}

# L'env ethoflow peut dater d'avant l'ajout d'une dépendance au .yml : s'il
# lui manque un module, on le complète (sans --prune : on ne retire rien de
# ce que l'utilisateur aurait installé). Pas de risque torch ici.
completer_ethoflow() {
    local py="$BASE/envs/ethoflow/bin/python"
    [ -x "$py" ] || return 0
    "$py" -c "import $IMPORTS_APP" >/dev/null 2>&1 && return 0
    ligne ... "ethoflow : dépendances manquantes, installation"
    "$CONDA" env update -n ethoflow -f environment-pipeline.yml -q
    "$py" -c "import $IMPORTS_APP" >/dev/null 2>&1 \
        || fin_erreur "l'env ethoflow reste incomplet - détail : $py -c 'import $IMPORTS_APP'"
    ligne OK "ethoflow (complété)"
}

etape "Environnements"
preparer_env ethoflow environment-pipeline.yml 1
completer_ethoflow
preparer_env dlc      environment-dlc.yml      0
preparer_env vame     environment-vame.yml     0

# ------------------------------------------------------------------ GPU
etape "GPU"
if ! command -v nvidia-smi >/dev/null 2>&1; then
    ligne INFO "pas de GPU NVIDIA : DLC et VAME tourneront sur CPU (10 à 50× plus lent)"
else
    while IFS= read -r g; do ligne INFO "$g"; done \
        < <(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null)
    for nom in dlc vame; do
        prefixe="$BASE/envs/$nom"
        [ -x "$prefixe/bin/python" ] || continue
        [ -f "$prefixe/.ethoflow-gpu-ok" ] && { ligne OK "$nom : torch utilise la GPU"; continue; }
        if "$prefixe/bin/python" scripts/diagnose_gpu.py >/dev/null 2>&1 \
           || { ligne ... "$nom : torch ne voit pas la GPU, réparation"
                "$prefixe/bin/python" scripts/diagnose_gpu.py --fix \
                && "$prefixe/bin/python" scripts/diagnose_gpu.py >/dev/null 2>&1; }; then
            touch "$prefixe/.ethoflow-gpu-ok"
            ligne OK "$nom : torch utilise la GPU"
        else
            ligne ATTENTION "$nom : torch ne voit toujours pas la GPU - détail : $prefixe/bin/python scripts/diagnose_gpu.py"
        fi
    done
fi

# ------------------------------------------- raccourci (Linux seulement)
# macOS : ce fichier se double-clique déjà. Linux : entrée dans le menu
# des applications, pour ne plus passer par le terminal.
if [ "$(uname)" = Linux ]; then
    raccourci="$HOME/.local/share/applications/ethoflow.desktop"
    # Recréé aussi s'il pointe vers l'ancien nom du lanceur (EthoFlow.command).
    if [ ! -f "$raccourci" ] || ! grep -qF "$ICI/EthoFlow-Mac-Linux.command" "$raccourci"; then
        mkdir -p "$(dirname "$raccourci")"
        cat > "$raccourci" <<EOF
[Desktop Entry]
Type=Application
Name=EthoFlow
Comment=Analyse comportementale (DeepLabCut + VAME)
Exec="$ICI/EthoFlow-Mac-Linux.command"
Path=$ICI
Terminal=true
Categories=Science;
EOF
        ligne OK "EthoFlow ajouté au menu des applications"
    fi
fi

# ----------------------------------------------------------- lancement
etape "Application"
ligne INFO "http://localhost:8501  (s'ouvre dans le navigateur)"
ligne INFO "garde cette fenêtre ouverte ; ferme-la pour arrêter EthoFlow"
echo
exec "$ICI/scripts/lancer_app.sh"
