param(
    [string]$Python = 'python',
    [string]$Git = 'git'
)

# Creates/updates only this project's optional checkout and optional environment.
# No full DB-GPT app, model weights, paid API call or changes to the old .venv.
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$taskRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$taskCheckout = Join-Path $taskRoot '.upstream/DB-GPT'
$taskPython = Join-Path $taskRoot '.venv-dbgpt/Scripts/python.exe'
$taskPin = '06a46eea2d248a0f2b3e9096035424df4f2dc2eb'

function Invoke-CheckedGit {
    param([string[]]$Arguments)
    & $Git @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Git failed with exit code $LASTEXITCODE" }
}

if (-not (Test-Path -LiteralPath $taskCheckout)) {
    New-Item -ItemType Directory -Path (Join-Path $taskRoot '.upstream') -Force | Out-Null
    Invoke-CheckedGit @('-c', 'core.longpaths=true', 'clone', '--filter=blob:none', '--no-checkout',
        'https://github.com/eosphoros-ai/DB-GPT.git', $taskCheckout)
}
$taskChanges = & $Git -C $taskCheckout status --porcelain
if ($LASTEXITCODE -ne 0) { throw 'The upstream path is not a Git checkout.' }
if ($taskChanges) { throw 'Upstream has local changes; preserve them before running setup.' }
Invoke-CheckedGit @('-C', $taskCheckout, 'sparse-checkout', 'set',
    'packages/dbgpt-core', 'packages/dbgpt-ext')
Invoke-CheckedGit @('-C', $taskCheckout, 'fetch', 'origin', $taskPin)
Invoke-CheckedGit @('-C', $taskCheckout, 'checkout', '--detach', $taskPin)

if (-not (Test-Path -LiteralPath $taskPython)) {
    & $Python -m venv --without-pip (Join-Path $taskRoot '.venv-dbgpt')
    if ($LASTEXITCODE -ne 0) { throw 'Failed to create the dedicated virtual environment.' }
}

$taskPipTemp = Join-Path $taskRoot 'tmp/pip-dbgpt'
New-Item -ItemType Directory -Path $taskPipTemp -Force | Out-Null
$taskPreviousTemp = $env:TEMP
$taskPreviousTmp = $env:TMP
$taskPreviousTmpdir = $env:TMPDIR
try {
    $env:TEMP = $taskPipTemp
    $env:TMP = $taskPipTemp
    $env:TMPDIR = $taskPipTemp
    & $Python -m pip --python $taskPython install --no-cache-dir -r (Join-Path $taskRoot 'requirements-dbgpt.txt') -r (Join-Path $taskRoot 'requirements-probes.txt')
    if ($LASTEXITCODE -ne 0) { throw 'Failed to install the minimal integration dependencies.' }
} finally {
    $env:TEMP = $taskPreviousTemp
    $env:TMP = $taskPreviousTmp
    $env:TMPDIR = $taskPreviousTmpdir
}

$taskPth = Join-Path $taskRoot '.venv-dbgpt/Lib/site-packages/data_verify_dbgpt.pth'
[IO.File]::WriteAllLines($taskPth, [string[]]@(
    $taskRoot,
    (Join-Path $taskCheckout 'packages/dbgpt-core/src'),
    (Join-Path $taskCheckout 'packages/dbgpt-ext/src')
), [Text.UTF8Encoding]::new($false))
& $taskPython -c 'from dataverify.dbgpt_bridge import verify_upstream_files; print(verify_upstream_files())'
if ($LASTEXITCODE -ne 0) { throw 'Pinned source import validation failed.' }
Write-Host 'Ready. Run .venv-dbgpt/Scripts/python.exe tools/validate_dbgpt.py'
