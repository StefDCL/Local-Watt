param(
    [switch]$Clean
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$venvPath = Join-Path $PSScriptRoot ".venv-build"
$venvPython = Join-Path $venvPath "Scripts\python.exe"

if ($Clean) {
    Remove-Item -Recurse -Force build, dist, LocalWatt.spec, $venvPath -ErrorAction SilentlyContinue
}

if (-not (Test-Path $venvPython)) {
    python -m venv $venvPath
}

& $venvPython -m pip install --upgrade pip pyinstaller bleak fit-tool
if ($LASTEXITCODE -ne 0) {
    throw "Dependency install failed."
}

$env:PYTHONNOUSERSITE = "1"
Get-Process -Name LocalWatt -ErrorAction SilentlyContinue | Stop-Process -Force
Start-Sleep -Milliseconds 300

& $venvPython -m PyInstaller `
    --noconfirm `
    --clean `
    --onefile `
    --windowed `
    --name LocalWatt `
    app.py
if ($LASTEXITCODE -ne 0) {
    throw "PyInstaller failed."
}

if (-not (Test-Path "dist\LocalWatt.exe")) {
    throw "Build failed: dist\LocalWatt.exe was not created."
}

Write-Host ""
Write-Host "Build complete: dist\LocalWatt.exe"
