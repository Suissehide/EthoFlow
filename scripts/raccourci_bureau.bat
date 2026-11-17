@echo off
rem EthoFlow - double-clic pour poser un raccourci "EthoFlow" (avec icone)
rem sur le Bureau. Windows, optionnel : le lanceur reste EthoFlow-Windows.bat.
rem Relancer ce script met le raccourci a jour (depot deplace, par exemple).
rem Les chemins passent par des variables d'environnement : pas de souci de
rem guillemets, d'apostrophes ni de lettre de lecteur cote PowerShell.
for %%R in ("%~dp0..") do set "ICI=%%~fR\"
set "CIBLE=%ICI%EthoFlow-Windows.bat"
set "ICONE=%ICI%scripts\ethoflow.ico"
if not exist "%CIBLE%" (
    echo Erreur : "%CIBLE%" introuvable. Ce script doit rester dans scripts\ du depot EthoFlow.
    pause
    exit /b 1
)
powershell -NoProfile -ExecutionPolicy Bypass -Command "$lnk = [IO.Path]::Combine([Environment]::GetFolderPath('Desktop'), 'EthoFlow.lnk'); $r = (New-Object -ComObject WScript.Shell).CreateShortcut($lnk); $r.TargetPath = $env:CIBLE; $r.WorkingDirectory = $env:ICI; $r.IconLocation = $env:ICONE; $r.Description = 'EthoFlow - analyse comportementale (DeepLabCut + VAME)'; $r.Save(); Write-Host ('Raccourci cree : ' + $lnk); Write-Host ('Il lance      : ' + $env:CIBLE)"
if errorlevel 1 echo Erreur : le raccourci n'a pas pu etre cree (voir le message ci-dessus).
pause
