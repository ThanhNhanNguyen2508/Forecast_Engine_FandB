param([string]$ConfigPath)

$ErrorActionPreference = "Stop"
$SourceRoot = Split-Path -Parent $PSScriptRoot
$EngineRoot = Split-Path -Parent $SourceRoot
$Python = Join-Path $EngineRoot ".venv-preprocess\Scripts\python.exe"

if ($ConfigPath) {
    & $Python -m shelfcash_preprocess doctor --live --config $ConfigPath
} else {
    & $Python -m shelfcash_preprocess doctor --live
}
exit $LASTEXITCODE
