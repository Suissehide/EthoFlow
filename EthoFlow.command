#!/usr/bin/env bash
# EthoFlow — double-clic pour installer (la première fois) et lancer l'app.
# macOS : double-clic dans le Finder. Linux : `./EthoFlow.command` une
# première fois dans un terminal ; EthoFlow apparaît ensuite dans le menu
# des applications. (Windows : EthoFlow.bat.)
#
# À chaque lancement :
#   1. trouve conda — et propose d'installer Miniforge s'il n'y en a pas ;
#   2. crée les environnements absents (ethoflow, dlc, vame), met à jour
#      ceux dont le fichier environment-*.yml a changé (git pull) ;
#   3. vérifie que torch voit la GPU NVIDIA dans dlc et vame, lance
#      `diagnose_gpu.py --fix` sinon (une fois par environnement) ;
#   4. lance l'app (./ethoflow) : elle s'ouvre dans le navigateur.
set -uo pipefail

ICI="$(cd -P "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ICI"

titre() { printf '\n\033[1m== %s\033[0m\n' "$1"; }
fin_erreur() {
    printf '\n\033[31mErreur : %s\033[0m\n' "$1" >&2
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

titre "Conda"
if ! CONDA="$(trouver_conda)"; then
    echo "Conda est introuvable. EthoFlow en a besoin pour ses environnements Python."
    read -r -p "Installer Miniforge dans $HOME/miniforge3 (~100 Mo, sans droits admin) ? [O/n] " rep
    case "$rep" in [nN]*) fin_erreur "installation annulée." ;; esac
    installeur="$(mktemp -t miniforge.XXXXXX).sh"
    url="https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
    curl -fL -o "$installeur" "$url" || fin_erreur "téléchargement impossible ($url)."
    bash "$installeur" -b -p "$HOME/miniforge3" || fin_erreur "l'installation de Miniforge a échoué."
    rm -f "$installeur"
    CONDA="$HOME/miniforge3/bin/conda"
fi
export CONDA_EXE="$CONDA"   # le lanceur ./ethoflow s'en sert pour trouver l'env
BASE="$(dirname "$(dirname "$CONDA")")"
echo "conda : $CONDA"

# ------------------------------------------------------- environnements
# Une copie du .yml est gardée dans l'env : s'il diffère (après un git
# pull), l'env est mis à jour. Un env créé à la main, sans copie, est
# laissé tel quel — une mise à jour pourrait y remplacer un torch réparé.
preparer_env() {   # nom fichier.yml obligatoire(0/1)
    local nom="$1" yml="$2" obligatoire="$3" prefixe="$BASE/envs/$1"
    local copie="$prefixe/.ethoflow-env.yml"
    if [ ! -d "$prefixe" ]; then
        echo "Création de l'environnement « $nom » (10 à 20 min, une seule fois)…"
        if ! "$CONDA" env create -f "$yml"; then
            # Un env à moitié créé passerait pour installé au prochain lancement.
            "$CONDA" env remove -n "$nom" -y >/dev/null 2>&1
            [ "$obligatoire" = 1 ] && fin_erreur "impossible de créer l'environnement « $nom »."
            echo "⚠ « $nom » n'a pas pu être créé : les étapes qui en dépendent seront indisponibles."
            return
        fi
        cp "$yml" "$copie"
    elif [ ! -f "$copie" ]; then
        cp "$yml" "$copie"
        echo "✓ $nom (déjà là)"
    elif ! cmp -s "$yml" "$copie"; then
        echo "Mise à jour de l'environnement « $nom » ($yml a changé)…"
        if "$CONDA" env update -n "$nom" -f "$yml" --prune; then
            cp "$yml" "$copie"
            rm -f "$prefixe/.ethoflow-gpu-ok"   # torch a pu changer : revérifier
        else
            echo "⚠ mise à jour de « $nom » échouée : l'ancienne version est gardée."
        fi
    else
        echo "✓ $nom"
    fi
}

titre "Environnements"
preparer_env ethoflow environment-pipeline.yml 1
preparer_env dlc      environment-dlc.yml      0
preparer_env vame     environment-vame.yml     0

# ------------------------------------------------------------------ GPU
titre "GPU"
if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "Pas de GPU NVIDIA sur cette machine : DLC et VAME tourneront sur CPU (10 à 50× plus lent)."
else
    for nom in dlc vame; do
        prefixe="$BASE/envs/$nom"
        [ -x "$prefixe/bin/python" ] || continue
        [ -f "$prefixe/.ethoflow-gpu-ok" ] && { echo "✓ $nom voit la GPU"; continue; }
        if "$prefixe/bin/python" scripts/diagnose_gpu.py >/dev/null 2>&1 \
           || { echo "« $nom » ne voit pas la GPU : réparation de torch…"
                "$prefixe/bin/python" scripts/diagnose_gpu.py --fix \
                && "$prefixe/bin/python" scripts/diagnose_gpu.py >/dev/null 2>&1; }; then
            touch "$prefixe/.ethoflow-gpu-ok"
            echo "✓ $nom voit la GPU"
        else
            echo "⚠ « $nom » ne voit toujours pas la GPU. Détail : $prefixe/bin/python scripts/diagnose_gpu.py"
        fi
    done
fi

# ------------------------------------------- raccourci (Linux seulement)
# macOS : ce fichier se double-clique déjà. Linux : entrée dans le menu
# des applications, pour ne plus passer par le terminal.
if [ "$(uname)" = Linux ]; then
    raccourci="$HOME/.local/share/applications/ethoflow.desktop"
    if [ ! -f "$raccourci" ]; then
        mkdir -p "$(dirname "$raccourci")"
        cat > "$raccourci" <<EOF
[Desktop Entry]
Type=Application
Name=EthoFlow
Comment=Analyse comportementale (DeepLabCut + VAME)
Exec="$ICI/EthoFlow.command"
Path=$ICI
Terminal=true
Categories=Science;
EOF
        echo "✓ EthoFlow ajouté au menu des applications."
    fi
fi

# ----------------------------------------------------------- lancement
titre "Lancement"
echo "L'app s'ouvre dans le navigateur. Garde cette fenêtre ouverte ; ferme-la pour arrêter EthoFlow."
exec "$ICI/ethoflow"
