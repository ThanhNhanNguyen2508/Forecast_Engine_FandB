[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][string]$Request,
    [Parameter(Mandatory=$true)][string]$Output,
    [string]$Python
)
$ErrorActionPreference = 'Stop'
$sourceRoot = Split-Path -Parent $PSScriptRoot
if (-not $Python) { $Python = Join-Path $sourceRoot '.venv\Scripts\python.exe' }
$requestPath = (Resolve-Path -LiteralPath $Request).Path
$outputPath = [System.IO.Path]::GetFullPath($Output)
$env:PYTHONDONTWRITEBYTECODE = '1'
$previousPythonPath = $env:PYTHONPATH
try {
    $env:PYTHONPATH = $sourceRoot
    & $Python -B -m shelfcash_pipeline.typed_runner --request $requestPath --output $outputPath
    if ($LASTEXITCODE -ne 0) { throw "Typed runner failed: exit $LASTEXITCODE" }
} finally { $env:PYTHONPATH = $previousPythonPath }
