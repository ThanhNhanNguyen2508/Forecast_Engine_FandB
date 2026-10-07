param(
    [string]$EngineRoot,
    [switch]$WithOcr,
    [switch]$WithAllFormats
)

$ErrorActionPreference = "Stop"
if (-not $EngineRoot) {
    $SourceRoot = Split-Path -Parent $PSScriptRoot
    $EngineRoot = $SourceRoot
    if (-not (Test-Path -LiteralPath (Join-Path $SourceRoot 'shelfcash.config.json'))) {
        $EngineRoot = Split-Path -Parent $SourceRoot
    }
}
$EngineRoot = [System.IO.Path]::GetFullPath($EngineRoot)
$SourceRoot = Join-Path $EngineRoot "source_code"
if (Test-Path -LiteralPath (Join-Path $EngineRoot 'pyproject.toml')) { $SourceRoot = $EngineRoot }
$Venv = Join-Path $EngineRoot ".venv-preprocess"
$Python = Join-Path $Venv "Scripts\python.exe"

if (-not (Test-Path -LiteralPath $Python)) {
    python -m venv $Venv
}

$Extras = "forecast,pdf,test"
if ($WithAllFormats) { $Extras = "all" }
elseif ($WithOcr) { $Extras = "forecast,pdf,test,ocr" }

& $Python -m pip install -e "${SourceRoot}[$Extras]"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $Python -m shelfcash_preprocess doctor --config (Join-Path $EngineRoot ".env.preprocess")
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "Setup complete. Configuration: $EngineRoot\.env.preprocess"
if ($WithOcr) {
    Write-Host "Python OCR adapter installed. Tesseract and Vietnamese language data must also be installed on Windows."
}
