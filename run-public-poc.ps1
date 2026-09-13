[CmdletBinding()]
param(
    [string]$ConfigPath = (Join-Path $PSScriptRoot "config.public.example.toml"),
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$PocArguments
)

$ErrorActionPreference = "Stop"
$projectDirectory = (Resolve-Path $PSScriptRoot).Path
$configCandidate = if ([System.IO.Path]::IsPathRooted($ConfigPath)) {
    $ConfigPath
} else {
    Join-Path $projectDirectory $ConfigPath
}
if (-not (Test-Path -LiteralPath $configCandidate -PathType Leaf)) {
    throw "POC configuration not found: $configCandidate"
}
$configPath = (Resolve-Path $configCandidate).Path
$venvDirectory = Join-Path $projectDirectory ".venv"
$venvPython = Join-Path $venvDirectory "Scripts\python.exe"

# Prefer the Python launcher so the required 3.12 interpreter is explicit.
$pythonCommand = Get-Command py.exe -ErrorAction SilentlyContinue
if ($null -ne $pythonCommand) {
    $pythonArguments = @("-3.12")
} else {
    $pythonCommand = Get-Command python.exe -ErrorAction SilentlyContinue
    $pythonArguments = @()
}
if ($null -eq $pythonCommand) {
    throw "Python 3.12 is required. Install it from python.org or the Microsoft Store."
}

$pythonVersion = (& $pythonCommand.Source @pythonArguments -c "import sys; print(f'{sys.version_info[0]}.{sys.version_info[1]}')").Trim()
if ($pythonVersion -ne "3.12") {
    throw "Python 3.12 is required; found $pythonVersion."
}

if (-not (Test-Path -LiteralPath $venvPython -PathType Leaf)) {
    & $pythonCommand.Source @pythonArguments -m venv $venvDirectory
}
$venvVersion = (& $venvPython -c "import sys; print(f'{sys.version_info[0]}.{sys.version_info[1]}')").Trim()
if ($venvVersion -ne "3.12") {
    throw "Existing .venv is Python $venvVersion, not Python 3.12. Remove .venv and run again."
}

Push-Location $projectDirectory
try {
    & $venvPython -m pip install --no-build-isolation .
    & $venvPython -m asr_pipeline.model_download --config $configPath

    # These wheels provide the CUDA runtime used by Faster-Whisper on Windows.
    # CPU configurations do not need them.
    $configText = Get-Content -LiteralPath $configPath -Raw
    $cublasBin = Join-Path $venvDirectory "Lib\site-packages\nvidia\cublas\bin"
    if ($configText -match '(?m)^\s*device\s*=\s*"cuda"' -and -not (Test-Path -LiteralPath $cublasBin)) {
        & $venvPython -m pip install "nvidia-cublas-cu12>=12,<13" "nvidia-cudnn-cu12>=9,<10" "nvidia-cuda-nvrtc-cu12>=12,<13"
    }

    & $venvPython -m asr_pipeline.public_poc_cli --config $configPath @PocArguments
    exit $LASTEXITCODE
} finally {
    Pop-Location
}
