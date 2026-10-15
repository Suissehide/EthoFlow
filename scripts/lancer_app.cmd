@echo off
rem EthoFlow — lance l'interface Streamlit. Windows.
rem (Sous Linux et macOS : scripts/lancer_app.sh.)
rem
rem Appele par EthoFlow-Windows.bat, le lanceur a double-cliquer de la
rem racine - c'est lui qu'il faut utiliser. Ce script-ci demarre l'app
rem seule, sans verifier les environnements ni la GPU :
rem     scripts\lancer_app.cmd                            ouvre l'app
rem     scripts\lancer_app.cmd --server.address=0.0.0.0   ouverte au reseau local
rem
rem Tout argument supplementaire est transmis tel quel a `streamlit run`.
rem Pas besoin d'activer l'env conda au prealable : le script trouve le
rem binaire de l'env `ethoflow` lui-meme.
setlocal EnableDelayedExpansion

set "ENV_NAME=ethoflow"
rem Ce script vit dans scripts\ : la racine du depot est le dossier parent.
for %%R in ("%~dp0..") do set "ICI=%%~fR\"
set "APP=%ICI%streamlit_app\app.py"

if not exist "%APP%" (
    echo Erreur : "%APP%" introuvable. 1>&2
    echo Ce script doit rester dans scripts\ du depot EthoFlow. 1>&2
    exit /b 1
)

rem On appelle le binaire de l'env directement plutot que `conda run` : ce
rem dernier tamponne la sortie tant que le process tourne, donc l'URL du
rem serveur n'apparaitrait jamais, et il s'interpose sur Ctrl-C.
set "STREAMLIT="

rem 1. L'env est deja active dans ce shell.
if /I "%CONDA_DEFAULT_ENV%"=="%ENV_NAME%" (
    for /f "delims=" %%S in ('where streamlit 2^>nul') do (
        if not defined STREAMLIT set "STREAMLIT=%%S"
    )
)

rem 2. Sinon, deduire la racine des envs et viser l'env directement.
if not defined STREAMLIT (
    set "BASE="
    if defined CONDA_EXE (
        rem %CONDA_EXE% vaut <base>\Scripts\conda.exe : deux niveaux au-dessus.
        for %%A in ("%CONDA_EXE%") do set "DOSSIER_SCRIPTS=%%~dpA"
        for %%B in ("!DOSSIER_SCRIPTS!..") do set "BASE=%%~fB"
    )
    if not defined BASE if defined MAMBA_ROOT_PREFIX set "BASE=%MAMBA_ROOT_PREFIX%"
    if not defined BASE (
        for /f "delims=" %%C in ('conda info --base 2^>nul') do set "BASE=%%C"
    )
    if defined BASE (
        if exist "!BASE!\envs\%ENV_NAME%\Scripts\streamlit.exe" (
            set "STREAMLIT=!BASE!\envs\%ENV_NAME%\Scripts\streamlit.exe"
        )
    )
)

if not defined STREAMLIT (
    echo Erreur : l'environnement conda "%ENV_NAME%" est introuvable. 1>&2
    echo. 1>&2
    echo Cree-le depuis la racine du depot : 1>&2
    echo     conda env create -f environment-pipeline.yml 1>&2
    echo. 1>&2
    echo ^(voir README.md, section "Installation"^) 1>&2
    exit /b 1
)

rem Premier lancement : Streamlit demande une adresse e-mail et attend une
rem reponse au clavier, serveur bloque, navigateur jamais ouvert. Un fichier
rem d'identifiants vide supprime la question.
if not exist "%USERPROFILE%\.streamlit\credentials.toml" (
    if not exist "%USERPROFILE%\.streamlit" mkdir "%USERPROFILE%\.streamlit"
    > "%USERPROFILE%\.streamlit\credentials.toml" (
        echo [general]
        echo email = ""
    )
)

rem Depuis la racine du depot : Streamlit y lit .streamlit\config.toml (theme).
cd /d "%ICI%"
"%STREAMLIT%" run "%APP%" %*
exit /b %ERRORLEVEL%
