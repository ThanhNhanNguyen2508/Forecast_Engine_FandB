[CmdletBinding()]
param([string]$Python = 'python')
$ErrorActionPreference = 'Stop'
$RepositoryRoot = Split-Path -Parent $PSScriptRoot
$EnvironmentRoot = Join-Path $RepositoryRoot '.venv'
$EnvironmentPython = Join-Path $EnvironmentRoot 'Scripts\python.exe'
if (-not (Test-Path -LiteralPath $EnvironmentPython -PathType Leaf)) {
    & $Python -m venv $EnvironmentRoot
    if ($LASTEXITCODE -ne 0) { throw 'Failed to create Python environment. Python 3.11 or newer is required.' }
}
$VersionText = & $EnvironmentPython -c "import sys; print(str(sys.version_info.major) + '.' + str(sys.version_info.minor))"
if ($LASTEXITCODE -ne 0) { throw 'Could not read the Python version.' }
$PythonVersion = [version]($VersionText.Trim())
if ($PythonVersion -lt [version]'3.11') { throw 'Python 3.11 or newer is required.' }
if ($PythonVersion -ge [version]'3.12') {
    & $EnvironmentPython -m pip install -r (Join-Path $RepositoryRoot 'requirements-demo.lock.txt')
    if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
    & $EnvironmentPython -m pip install --no-deps -e $RepositoryRoot
} else {
    # The pinned numpy/scipy versions require 3.12+. Resolve declared compatible
    # versions on 3.11 rather than attempting an incompatible lock installation.
    & $EnvironmentPython -m pip install -e ($RepositoryRoot + '[demo,test]')
}
if ($LASTEXITCODE -ne 0) { throw 'ShelfCash installation failed.' }
& (Join-Path $RepositoryRoot 'run.ps1') -Python $EnvironmentPython -ValidateOnly
Write-Host 'Ready. Run .\run.ps1 -StopAfter m5 or -StopAfter m6 from the repository root.'
