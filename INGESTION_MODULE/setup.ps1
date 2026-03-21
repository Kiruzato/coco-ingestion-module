#
# CoCo Ingestion Module - Windows Setup
# =======================================
# Sets up the ingestion environment for building vector store packages.
#
# Usage: .\INGESTION_MODULE\setup.ps1
#

$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "==============================================" -ForegroundColor Cyan
Write-Host "  CoCo Ingestion Module - Setup" -ForegroundColor Cyan
Write-Host "==============================================" -ForegroundColor Cyan
Write-Host ""

# Get paths
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir

Write-Host "Ingestion module: $ScriptDir"
Write-Host "Project root:     $ProjectRoot"
Write-Host ""

# Step 1: Check Python
Write-Host "[1/3] Checking Python..." -ForegroundColor Yellow
$PythonCmd = $null
try {
    $pyVer = & python --version 2>&1
    if ($pyVer -match "Python (\d+)\.(\d+)") {
        $major = [int]$Matches[1]
        $minor = [int]$Matches[2]
        if ($major -ge 3 -and $minor -ge 8) {
            $PythonCmd = "python"
            Write-Host "$pyVer found" -ForegroundColor Green
        }
    }
} catch {}

if (-not $PythonCmd) {
    Write-Host "Error: Python 3.8+ required but not found." -ForegroundColor Red
    exit 1
}
Write-Host ""

# Step 2: Create virtual environment and install dependencies
Write-Host "[2/3] Setting up virtual environment..." -ForegroundColor Yellow
$VenvDir = Join-Path $ScriptDir ".venv"

if (-not (Test-Path $VenvDir)) {
    & $PythonCmd -m venv $VenvDir
    Write-Host "Virtual environment created" -ForegroundColor Green
} else {
    Write-Host "Virtual environment already exists"
}

# Activate and install
$VenvPython = Join-Path $VenvDir "Scripts\python.exe"
$VenvPip = Join-Path $VenvDir "Scripts\pip.exe"
& $VenvPython -m pip install --upgrade pip
& $VenvPip install -r (Join-Path $ScriptDir "requirements.txt")
Write-Host "Dependencies installed" -ForegroundColor Green
Write-Host ""

# Step 3: Verify all dependencies
Write-Host "[3/3] Verifying installation..." -ForegroundColor Yellow

# Run verification from the module directory so 'from modules.*' resolves correctly
Push-Location $ScriptDir
try {
    & $VenvPython -c "import langchain_core; print('  langchain-core: OK')"
    & $VenvPython -c "import langchain_text_splitters; print('  langchain-text-splitters: OK')"
    & $VenvPython -c "import langchain_openai; print('  langchain-openai: OK')"
    & $VenvPython -c "import faiss; print('  faiss-cpu: OK')"
    & $VenvPython -c "from pypdf import PdfReader; print('  pypdf: OK')"
    & $VenvPython -c "from unstructured.partition.pdf import partition_pdf; print('  unstructured[pdf]: OK')"
    & $VenvPython -c "from docx import Document; print('  python-docx: OK')"
    & $VenvPython -c "from modules.document_manager import DocumentManager; print('  DocumentManager: OK')"
} catch {
    Pop-Location
    Write-Host ""
    Write-Host "ERROR: Dependency verification failed." -ForegroundColor Red
    Write-Host "  $_" -ForegroundColor Red
    Write-Host ""
    Write-Host "Setup cannot continue. Try deleting .venv and running setup again:" -ForegroundColor Yellow
    Write-Host "  Remove-Item -Recurse -Force $VenvDir" -ForegroundColor Yellow
    Write-Host "  .\setup.ps1" -ForegroundColor Yellow
    exit 1
}
Pop-Location
Write-Host "All dependencies verified" -ForegroundColor Green
Write-Host ""

Write-Host "==============================================" -ForegroundColor Green
Write-Host "  Setup Complete!" -ForegroundColor Green
Write-Host "==============================================" -ForegroundColor Green
Write-Host ""
Write-Host "To run ingestion:"
Write-Host "  cd $ScriptDir"
Write-Host "  .\.venv\Scripts\Activate.ps1"
Write-Host '  python ingest.py <documents_folder>'
Write-Host ""
