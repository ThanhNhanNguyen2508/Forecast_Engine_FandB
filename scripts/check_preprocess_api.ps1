param([string]$ConfigPath)

$ErrorActionPreference = "Stop"
$SourceRoot = Split-Path -Parent $PSScriptRoot
$EngineRoot = Split-Path -Parent $SourceRoot
if (Test-Path -LiteralPath (Join-Path $SourceRoot 'shelfcash.config.json')) { $EngineRoot = $SourceRoot }
$Python = Join-Path $EngineRoot ".venv-preprocess\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $Python)) { $Python = Join-Path $EngineRoot '.venv\Scripts\python.exe' }

if ($ConfigPath) {
    & $Python -m shelfcash_preprocess doctor --live --config $ConfigPath
} else {
    & $Python -m shelfcash_preprocess doctor --live
}
exit $LASTEXITCODE
