<#
.SYNOPSIS
Run the offline ShelfCash pipeline, stopping at a chosen milestone.
.EXAMPLE
& .\source_code\shelfcash_pipeline\run.ps1 -StopAfter m1
.EXAMPLE
& .\source_code\shelfcash_pipeline\run.ps1 -BundlePath 'C:\path\to\bundle' -StopAfter m2
.NOTES
WRITES / OVERWRITES MANAGED OUTPUT. Loads fixed artifacts; never trains or promotes.
#>
[CmdletBinding()]
param(
    [ValidateSet('preprocess', 'm1', 'point', 'm2', 'm3', 'm4', 'm5', 'm6')]
    [string]$StopAfter = 'm2',
    [string]$InputPath,
    [string]$BundlePath,
    [string]$ArtifactsPath,
    [string]$OutputDir,
    [string]$CutoffDate = '2026-08-12',
    [int]$Horizon = 7,
    [ValidateSet('demo', 'backtest_replay', 'production')]
    [string]$ExecutionMode = 'demo',
    [string]$StoreId = 'STORE_A',
    [ValidateSet('DMY', 'MDY', 'YMD')]
    [string]$DateLocale = 'DMY',
    [string]$ContextMetadataFile,
    [string]$PlanningConfig,
    [ValidateRange(1, 2000)]
    [int]$ScenarioCount = 100,
    [int]$Seed = 42,
    [ValidateSet('deterministic', 'stochastic', 'compare')]
    [string]$OptimizationMode = 'compare',
    [string]$Python,
    [switch]$Help
)

$ErrorActionPreference = 'Stop'
$StopAfter = $StopAfter.ToLowerInvariant()
$ExecutionMode = $ExecutionMode.ToLowerInvariant()
$OptimizationMode = $OptimizationMode.ToLowerInvariant()
$DateLocale = $DateLocale.ToUpperInvariant()
$PipelineEngineRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$PipelineSourceRoot = Join-Path $PipelineEngineRoot 'source_code'
if (-not $Python) {
    $Python = Join-Path $PipelineEngineRoot '.venv-preprocess\Scripts\python.exe'
}
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Python executable not found: $Python"
}
if ($InputPath -and $BundlePath) {
    throw 'Use InputPath OR BundlePath, not both.'
}

$PipelineArgs = @('-B', '-m', 'shelfcash_pipeline')
if ($Help) {
    $PipelineArgs += '--help'
} else {
    $PipelineArgs += @('--stop-after', $StopAfter, '--cutoff-date', $CutoffDate,
        '--horizon', [string]$Horizon, '--execution-mode', $ExecutionMode,
        '--store-id', $StoreId, '--date-locale', $DateLocale,
        '--scenario-count', [string]$ScenarioCount, '--seed', [string]$Seed,
        '--optimization-mode', $OptimizationMode)
    if ($InputPath) { $PipelineArgs += @('--input', $InputPath) }
    if ($BundlePath) { $PipelineArgs += @('--bundle', $BundlePath) }
    if ($ArtifactsPath) { $PipelineArgs += @('--artifacts', $ArtifactsPath) }
    if ($OutputDir) { $PipelineArgs += @('--output-dir', $OutputDir) }
    if ($ContextMetadataFile) { $PipelineArgs += @('--context-metadata', $ContextMetadataFile) }
    if ($PlanningConfig) { $PipelineArgs += @('--planning-config', $PlanningConfig) }
}

$PreviousPythonPath = $env:PYTHONPATH
$PreviousPythonUtf8 = $env:PYTHONUTF8
$PreviousBytecode = $env:PYTHONDONTWRITEBYTECODE
try {
    $env:PYTHONPATH = $PipelineSourceRoot
    if ($PreviousPythonPath) {
        $env:PYTHONPATH += [System.IO.Path]::PathSeparator + $PreviousPythonPath
    }
    $env:PYTHONUTF8 = '1'
    $env:PYTHONDONTWRITEBYTECODE = '1'
    & $Python @PipelineArgs
    $PipelineExitCode = $LASTEXITCODE
    if ($PipelineExitCode -ne 0) {
        throw "ShelfCash pipeline failed (exit $PipelineExitCode). Inspect run_manifest.json in the milestone output folder."
    }
} finally {
    $env:PYTHONPATH = $PreviousPythonPath
    $env:PYTHONUTF8 = $PreviousPythonUtf8
    $env:PYTHONDONTWRITEBYTECODE = $PreviousBytecode
}
