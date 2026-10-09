<#
.SYNOPSIS
Run the offline ShelfCash pipeline, stopping at a chosen milestone.
.EXAMPLE
& .\source_code\run.ps1 -ConfigFile .\source_code\shelfcash.config.json -StopAfter m1
.EXAMPLE
& .\source_code\shelfcash_pipeline\run.ps1 -BundlePath 'C:\path\to\bundle' -ArtifactsPath 'C:\path\to\artifacts' -CutoffDate '2028-02-28' -StopAfter m2
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
    [string]$CutoffDate,
    [int]$Horizon = 7,
    [ValidateSet('demo', 'backtest_replay', 'production')]
    [string]$ExecutionMode = 'demo',
    [string]$StoreId,
    [ValidateSet('DMY', 'MDY', 'YMD')]
    [string]$DateLocale,
    [string]$ContextMetadataFile,
    [string]$PlanningConfig,
    [string]$ForecastOverrides,
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
if ($DateLocale) {$DateLocale = $DateLocale.ToUpperInvariant()}
$PipelineRepositoryRoot = Split-Path -Parent $PSScriptRoot
$PipelineEngineRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$PipelineSourceRoot = Join-Path $PipelineEngineRoot 'source_code'
if ((Test-Path -LiteralPath (Join-Path $PipelineRepositoryRoot 'shelfcash.config.json')) -and
    (Test-Path -LiteralPath (Join-Path $PipelineRepositoryRoot 'pyproject.toml'))) {
    $PipelineEngineRoot = $PipelineRepositoryRoot
    $PipelineSourceRoot = $PipelineRepositoryRoot
}
if (-not $Python) {
    $Python = Join-Path $PipelineEngineRoot '.venv\Scripts\python.exe'
    if (-not (Test-Path -LiteralPath $Python)) { $Python = Join-Path $PipelineEngineRoot '.venv-preprocess\Scripts\python.exe' }
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
    if (-not $CutoffDate) {throw 'PIPELINE_CUTOFF_REQUIRED: supply -CutoffDate (YYYY-MM-DD)'}
    if (-not $InputPath -and -not $BundlePath) {throw 'PIPELINE_INPUT_REQUIRED: supply -InputPath or -BundlePath'}
    $PipelineArgs += @('--stop-after', $StopAfter, '--cutoff-date', $CutoffDate,
        '--horizon', [string]$Horizon, '--execution-mode', $ExecutionMode,
        '--scenario-count', [string]$ScenarioCount, '--seed', [string]$Seed,
        '--optimization-mode', $OptimizationMode)
    if ($StoreId) {$PipelineArgs += @('--store-id', $StoreId)}
    if ($DateLocale) {$PipelineArgs += @('--date-locale', $DateLocale)}
    if ($InputPath) { $PipelineArgs += @('--input', $InputPath) }
    if ($BundlePath) { $PipelineArgs += @('--bundle', $BundlePath) }
    if ($ArtifactsPath) { $PipelineArgs += @('--artifacts', $ArtifactsPath) }
    if ($OutputDir) { $PipelineArgs += @('--output-dir', $OutputDir) }
    if ($ContextMetadataFile) { $PipelineArgs += @('--context-metadata', $ContextMetadataFile) }
    if ($PlanningConfig) { $PipelineArgs += @('--planning-config', $PlanningConfig) }
    if ($ForecastOverrides) { $PipelineArgs += @('--forecast-overrides', $ForecastOverrides) }
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
