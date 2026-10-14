@echo off
rem EthoFlow - double-clic pour installer (la premiere fois) et lancer l'app.
rem Windows. (macOS / Linux : EthoFlow.command.)
rem
rem A chaque lancement :
rem   1. trouve conda - et propose d'installer Miniforge s'il n'y en a pas ;
rem   2. cree les environnements absents (ethoflow, dlc, vame), met a jour
rem      ceux dont le fichier environment-*.yml a change (git pull) ;
rem   3. verifie que torch voit la GPU NVIDIA dans dlc et vame, lance
rem      `diagnose_gpu.py --fix` sinon (une fois par environnement) ;
rem   4. lance l'app (ethoflow.cmd) : elle s'ouvre dans le navigateur.
setlocal EnableDelayedExpansion
chcp 65001 >nul
set "ICI=%~dp0"
cd /d "%ICI%"

rem ---------------------------------------------------------------- conda
echo.
echo == Conda
set "CONDA="
if defined CONDA_EXE if exist "%CONDA_EXE%" set "CONDA=%CONDA_EXE%"
if not defined CONDA (
    for %%D in ("%USERPROFILE%\miniforge3" "%USERPROFILE%\mambaforge" "%USERPROFILE%\miniconda3" ^
                "%USERPROFILE%\anaconda3" "%LOCALAPPDATA%\miniforge3" "%LOCALAPPDATA%\miniconda3" ^
                "%ProgramData%\miniforge3" "%ProgramData%\miniconda3" "%ProgramData%\anaconda3") do (
        if not defined CONDA if exist "%%~D\Scripts\conda.exe" set "CONDA=%%~D\Scripts\conda.exe"
    )
)
if not defined CONDA (
    for /f "delims=" %%C in ('where conda.exe 2^>nul') do if not defined CONDA set "CONDA=%%C"
)
if not defined CONDA (
    echo Conda est introuvable. EthoFlow en a besoin pour ses environnements Python.
    set /p "REP=Installer Miniforge dans %USERPROFILE%\miniforge3 (~100 Mo, sans droits admin) ? [O/n] "
    if /I "!REP!"=="n" goto :erreur_annule
    set "INSTALLEUR=%TEMP%\Miniforge3-Windows-x86_64.exe"
    curl.exe -fL -o "!INSTALLEUR!" https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Windows-x86_64.exe
    if errorlevel 1 (set "MSG=telechargement de Miniforge impossible." & goto :erreur)
    echo Installation de Miniforge...
    start /wait "" "!INSTALLEUR!" /InstallationType=JustMe /RegisterPython=0 /AddToPath=0 /S /D=%USERPROFILE%\miniforge3
    del "!INSTALLEUR!" >nul 2>&1
    set "CONDA=%USERPROFILE%\miniforge3\Scripts\conda.exe"
    if not exist "!CONDA!" (set "MSG=l'installation de Miniforge a echoue." & goto :erreur)
)
rem Le lanceur ethoflow.cmd s'en sert pour trouver l'env.
set "CONDA_EXE=%CONDA%"
for %%A in ("%CONDA%") do set "SCRIPTS=%%~dpA"
for %%B in ("%SCRIPTS%..") do set "BASE=%%~fB"
echo conda : %CONDA%

rem ------------------------------------------------------- environnements
rem Une copie du .yml est gardee dans l'env : si elle differe (apres un git
rem pull), l'env est mis a jour. Un env cree a la main, sans copie, est
rem laisse tel quel - une mise a jour pourrait y remplacer un torch repare.
echo.
echo == Environnements
call :preparer_env ethoflow environment-pipeline.yml 1 || goto :erreur
call :preparer_env dlc environment-dlc.yml 0
call :preparer_env vame environment-vame.yml 0

rem ------------------------------------------------------------------ GPU
echo.
echo == GPU
where nvidia-smi >nul 2>&1
if errorlevel 1 (
    echo Pas de GPU NVIDIA sur cette machine : DLC et VAME tourneront sur CPU ^(10 a 50x plus lent^).
) else (
    call :verifier_gpu dlc
    call :verifier_gpu vame
)

rem ----------------------------------------------------------- lancement
echo.
echo == Lancement
echo L'app s'ouvre dans le navigateur. Garde cette fenetre ouverte ; ferme-la pour arreter EthoFlow.
call "%ICI%ethoflow.cmd"
if errorlevel 1 (set "MSG=l'app s'est arretee sur une erreur (voir ci-dessus)." & goto :erreur)
exit /b 0


:preparer_env
rem %1 nom, %2 fichier .yml, %3 obligatoire (1/0)
set "PREFIXE=%BASE%\envs\%~1"
set "COPIE=%PREFIXE%\.ethoflow-env.yml"
if not exist "%PREFIXE%\" (
    echo Creation de l'environnement "%~1" ^(10 a 20 min, une seule fois^)...
    call "%CONDA%" env create -f "%~2"
    if errorlevel 1 (
        rem Un env a moitie cree passerait pour installe au prochain lancement.
        call "%CONDA%" env remove -n %~1 -y >nul 2>&1
        if "%~3"=="1" (set "MSG=impossible de creer l'environnement %~1." & exit /b 1)
        echo Attention : "%~1" n'a pas pu etre cree : les etapes qui en dependent seront indisponibles.
        exit /b 0
    )
    copy /y "%~2" "%COPIE%" >nul
    exit /b 0
)
if not exist "%COPIE%" (
    copy /y "%~2" "%COPIE%" >nul
    echo OK %~1 ^(deja la^)
    exit /b 0
)
fc /b "%~2" "%COPIE%" >nul 2>&1
if errorlevel 1 (
    echo Mise a jour de l'environnement "%~1" ^(%~2 a change^)...
    call "%CONDA%" env update -n %~1 -f "%~2" --prune
    if errorlevel 1 (
        echo Attention : mise a jour de "%~1" echouee : l'ancienne version est gardee.
    ) else (
        copy /y "%~2" "%COPIE%" >nul
        rem torch a pu changer : reverifier la GPU.
        del "%PREFIXE%\.ethoflow-gpu-ok" >nul 2>&1
    )
    exit /b 0
)
echo OK %~1
exit /b 0


:verifier_gpu
rem %1 nom de l'env
set "PREFIXE=%BASE%\envs\%~1"
if not exist "%PREFIXE%\python.exe" exit /b 0
if exist "%PREFIXE%\.ethoflow-gpu-ok" (echo OK %~1 voit la GPU & exit /b 0)
"%PREFIXE%\python.exe" scripts\diagnose_gpu.py >nul 2>&1
if not errorlevel 1 goto :gpu_ok
echo "%~1" ne voit pas la GPU : reparation de torch...
"%PREFIXE%\python.exe" scripts\diagnose_gpu.py --fix
"%PREFIXE%\python.exe" scripts\diagnose_gpu.py >nul 2>&1
if not errorlevel 1 goto :gpu_ok
echo Attention : "%~1" ne voit toujours pas la GPU. Detail : "%PREFIXE%\python.exe" scripts\diagnose_gpu.py
exit /b 0
:gpu_ok
type nul > "%PREFIXE%\.ethoflow-gpu-ok"
echo OK %~1 voit la GPU
exit /b 0


:erreur_annule
set "MSG=installation annulee."
:erreur
echo.
echo Erreur : %MSG%
pause
exit /b 1
