# Build the Windows release of transitphot.
#
#   powershell -ExecutionPolicy Bypass -File packaging\build.ps1
#
# Uses the project's .venv (create it and `pip install -e ".[ld]"` first;
# with batman installed the build ships the reference limb-darkened model).
# Output: dist\transitphot\ and dist\transitphot-<version>-windows.zip
$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
$py = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { throw "No .venv found at $root\.venv" }

& $py -m pip install --quiet "pyinstaller>=6.16"
if ($LASTEXITCODE -ne 0) { throw "pip install pyinstaller failed" }

Push-Location $PSScriptRoot
try {
    & $py -m PyInstaller transitphot.spec --noconfirm `
        --distpath (Join-Path $root "dist") --workpath (Join-Path $root "build")
    if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed" }
} finally { Pop-Location }

$app = Join-Path $root "dist\transitphot"
Copy-Item (Join-Path $PSScriptRoot "README-windows.md") (Join-Path $app "READ ME FIRST.txt")

# Smoke test: the console exe must import the whole pipeline.
& (Join-Path $app "transitphot-cli.exe") --help | Out-Null
if ($LASTEXITCODE -ne 0) { throw "transitphot-cli.exe failed to start" }

$version = (& $py -c "import importlib.metadata as m; print(m.version('transitphot'))").Trim()
$zip = Join-Path $root "dist\transitphot-$version-windows.zip"
if (Test-Path $zip) { Remove-Item $zip }
Compress-Archive -Path $app -DestinationPath $zip
Write-Host "Built $zip"
