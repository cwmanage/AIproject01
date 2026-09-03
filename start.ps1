# =====================================================================
#  One-click launcher for PowerShell (Windows)
#  Usage:  .\start.ps1   (or right-click -> Run with PowerShell)
# =====================================================================
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

# --- find a Python interpreter ------------------------------------------
$py = (Get-Command python -ErrorAction SilentlyContinue)
if ($py) {
    & python start.py
    exit $LASTEXITCODE
}
$py3 = (Get-Command py -ErrorAction SilentlyContinue)
if ($py3) {
    & py -3 start.py
    exit $LASTEXITCODE
}
Write-Host "[start.ps1] Python was not found on PATH."
Write-Host "Please install Python 3.9+ from https://www.python.org/downloads/"
exit 1
