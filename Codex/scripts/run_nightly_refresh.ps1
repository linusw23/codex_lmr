$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$pythonDir = Join-Path $root "Python"

Set-Location $pythonDir
python nightly_refresh.py
