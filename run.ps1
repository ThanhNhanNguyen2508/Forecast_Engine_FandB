<# Run from a checkout: .\run.ps1 -StopAfter m5 #>
[CmdletBinding(DefaultParameterSetName='Milestone')]
param(
    [Parameter(ParameterSetName='Milestone')]
    [ValidateSet('preprocess','m1','point','m2','m3','m4','m5','m6')]
    [string]$StopAfter = 'm2',
    [Parameter(ParameterSetName='WhatIf',Mandatory=$true)][switch]$WhatIf,
    [Parameter(ParameterSetName='WhatIf')][string]$BaselineDir,
    [string]$ConfigFile,
    [string]$Python,
    [switch]$ValidateOnly,
    [switch]$Help
)
$ErrorActionPreference = 'Stop'
if (-not $Python) {
    $Candidates = @((Join-Path $PSScriptRoot '.venv\Scripts\python.exe'),
                    (Join-Path $PSScriptRoot '.venv-preprocess\Scripts\python.exe'))
    if ($env:VIRTUAL_ENV) { $Candidates += Join-Path $env:VIRTUAL_ENV 'Scripts\python.exe' }
    foreach ($Candidate in $Candidates) {
        if (Test-Path -LiteralPath $Candidate -PathType Leaf) { $Python = $Candidate; break }
    }
}
if (-not $Python -or -not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw 'Python environment not found. Run .\scripts\setup.ps1 once, or pass -Python <python.exe>.'
}
$Module = 'shelfcash_pipeline.config_runner'
if ($WhatIf) { $Module = 'shelfcash_pipeline.what_if_runner' }
$Arguments = @('-B','-m',$Module)
if ($Help) { $Arguments += '--help' } else {
    if (-not $ConfigFile) {
        $ConfigFile = Join-Path $PSScriptRoot 'shelfcash.config.json'
        if ($WhatIf) { $ConfigFile = Join-Path $PSScriptRoot 'configs\what_if_budget.example.json' }
    }
    $Arguments += @('--config',([System.IO.Path]::GetFullPath($ConfigFile)))
    if ($WhatIf) {
        if (-not $BaselineDir) { $BaselineDir = Join-Path $PSScriptRoot 'outputs\pipeline_until_m6' }
        $Arguments += @('--baseline',([System.IO.Path]::GetFullPath($BaselineDir)))
    } else { $Arguments += @('--stop-after',$StopAfter.ToLowerInvariant()) }
    if ($ValidateOnly) { $Arguments += '--validate-only' }
}
$PreviousEnvironment = @{}
foreach ($Name in @('PYTHONPATH','PYTHONUTF8','PYTHONDONTWRITEBYTECODE','SHELFCASH_ENGINE_ROOT')) {
    $PreviousEnvironment[$Name] = [Environment]::GetEnvironmentVariable($Name,'Process')
}
try {
    $env:PYTHONPATH = $PSScriptRoot
    if ($PreviousEnvironment['PYTHONPATH']) { $env:PYTHONPATH += [System.IO.Path]::PathSeparator + $PreviousEnvironment['PYTHONPATH'] }
    $env:PYTHONUTF8 = '1'
    $env:PYTHONDONTWRITEBYTECODE = '1'
    $env:SHELFCASH_ENGINE_ROOT = $PSScriptRoot
    & $Python @Arguments
    $NativeExit = $LASTEXITCODE
    if ($NativeExit -ne 0) { throw "ShelfCash failed (exit $NativeExit). Inspect the reported configuration or output." }
} finally {
    foreach ($Name in $PreviousEnvironment.Keys) {
        [Environment]::SetEnvironmentVariable($Name,$PreviousEnvironment[$Name],'Process')
    }
}
