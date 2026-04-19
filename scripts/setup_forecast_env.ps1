param(
  [Parameter(Mandatory = $true)]
  [string]$PythonExe,

  [string]$VenvPath = ".venv312"
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path -LiteralPath $PythonExe)) {
  throw "Python executable not found: $PythonExe"
}

$versionText = & $PythonExe -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')"
$majorMinor = & $PythonExe -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"

if ($majorMinor -ne "3.12") {
  throw "SeaPredictor forecast dependencies should use Python 3.12 on Windows. Got Python $versionText at $PythonExe"
}

Write-Host "Creating $VenvPath with Python $versionText"
& $PythonExe -m venv $VenvPath

$venvPython = Join-Path $VenvPath "Scripts\python.exe"
Write-Host "Upgrading pip/setuptools/wheel"
& $venvPython -m pip install --upgrade pip setuptools wheel

Write-Host "Installing SeaPredictor requirements"
& $venvPython -m pip install -r requirements.txt

Write-Host "Verifying forecast imports"
& $venvPython -c "import cartopy, opendrift; print('forecast stack ok')"

Write-Host ""
Write-Host "Activate with:"
Write-Host "  .\$VenvPath\Scripts\Activate.ps1"
Write-Host "Run with:"
Write-Host "  python -m uvicorn src.api.server:app --host 0.0.0.0 --port 8000 --reload"
