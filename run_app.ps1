param(
    [int]$Port = 8501
)

$ErrorActionPreference = 'Stop'

$repositoryRoot = $PSScriptRoot
$sitePackages = Join-Path $repositoryRoot '.venv\Lib\site-packages'
$previousPythonPath = $env:PYTHONPATH

Push-Location $repositoryRoot
try {
    $pythonLauncher = Get-Command py.exe -ErrorAction SilentlyContinue
    if (-not $pythonLauncher) {
        throw 'Python Launcher (py.exe) was not found. Install or enable an approved Python 3.12 runtime.'
    }

    $pythonEntries = & $pythonLauncher.Source -0p
    if ($LASTEXITCODE -ne 0) {
        throw 'Could not list Python runtimes with "py -0p".'
    }

    $python = $null
    foreach ($entry in $pythonEntries) {
        if ($entry -match '3\.12' -and $entry -match '(?<path>[A-Za-z]:\\.+\\python\.exe)\s*$') {
            $python = $Matches.path
            break
        }
    }
    if (-not $python) {
        throw 'No installed Python 3.12 runtime was found by "py -0p".'
    }

    if (-not (Test-Path -LiteralPath $sitePackages -PathType Container)) {
        throw "Project dependencies were not found at '$sitePackages'. Install them with uv before launching."
    }

    if ($previousPythonPath) {
        $env:PYTHONPATH = "$sitePackages;$previousPythonPath"
    }
    else {
        $env:PYTHONPATH = $sitePackages
    }

    Write-Host "Checking Streamlit with approved Python 3.12..."
    & $python -c 'import streamlit; import app.ui.streamlit_app; print(f"Streamlit {streamlit.__version__} is ready.")'
    if ($LASTEXITCODE -ne 0) {
        throw 'The approved Python interpreter could not load Streamlit or the application. Check the installed dependencies.'
    }

    Write-Host "Starting Streamlit at http://127.0.0.1:$Port ..."
    & $python -m streamlit run (Join-Path $repositoryRoot 'streamlit_app.py') `
        --server.headless true `
        --server.address 127.0.0.1 `
        --server.port $Port
    if ($LASTEXITCODE -ne 0) {
        throw "Streamlit exited with code $LASTEXITCODE."
    }
}
finally {
    $env:PYTHONPATH = $previousPythonPath
    Pop-Location
}