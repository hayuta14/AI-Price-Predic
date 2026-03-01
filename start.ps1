# BTCUSDT Trading System Optimization Platform - PowerShell Startup Script
# Script to start the project on Windows PowerShell

$ErrorActionPreference = "Stop"

# Set console encoding to UTF-8
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "BTCUSDT Trading System Optimization Platform" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Function to check if command exists
function Test-Command {
    param($Command)
    $null = Get-Command $Command -ErrorAction SilentlyContinue
    return $?
}

# 1. Check Python
Write-Host "[1/6] Checking Python..." -ForegroundColor Yellow
$pythonCmd = $null
$pythonFound = $false

# Try different Python commands
$pythonCommands = @("python", "python3", "py")
foreach ($cmd in $pythonCommands) {
    if (Test-Command $cmd) {
        try {
            $version = & $cmd --version 2>&1
            if ($LASTEXITCODE -eq 0 -or $?) {
                $pythonCmd = $cmd
                $pythonFound = $true
                Write-Host "[OK] Python found: $version" -ForegroundColor Green
                break
            }
        } catch {
            continue
        }
    }
}

# Try to find Python in common locations
if (-not $pythonFound) {
    $pythonPaths = @(
        "C:\Program Files\Python*",
        "$env:LOCALAPPDATA\Programs\Python",
        "$env:ProgramFiles\Python*"
    )
    
    foreach ($path in $pythonPaths) {
        $found = Get-ChildItem -Path $path -Filter "python.exe" -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($found) {
            $pythonCmd = $found.FullName
            $pythonFound = $true
            $version = & $pythonCmd --version 2>&1
            Write-Host "[OK] Python found: $version" -ForegroundColor Green
            break
        }
    }
}

if (-not $pythonFound) {
    Write-Host "[ERROR] Python is not installed!" -ForegroundColor Red
    Write-Host ""
    Write-Host "Please install Python 3.11+ from one of these options:" -ForegroundColor Yellow
    Write-Host "1. Download from: https://www.python.org/downloads/" -ForegroundColor White
    Write-Host "   - Select Python 3.11 or newer" -ForegroundColor Gray
    Write-Host "   - IMPORTANT: Check 'Add Python to PATH' during installation" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "2. Or install via Microsoft Store:" -ForegroundColor White
    Write-Host "   - Open Microsoft Store" -ForegroundColor Gray
    Write-Host "   - Search for 'Python 3.11' or 'Python 3.12'" -ForegroundColor Gray
    Write-Host "   - Click Install" -ForegroundColor Gray
    Write-Host ""
    Write-Host "3. Or install via winget (if available):" -ForegroundColor White
    Write-Host "   winget install Python.Python.3.11" -ForegroundColor Gray
    Write-Host ""
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host ""

# 2. Check pip
Write-Host "[2/6] Checking pip..." -ForegroundColor Yellow
try {
    & $pythonCmd -m pip --version | Out-Null
    if ($LASTEXITCODE -ne 0) { throw }
    Write-Host "[OK] pip is ready" -ForegroundColor Green
} catch {
    Write-Host "[WARNING] pip not found, attempting to install..." -ForegroundColor Yellow
    & $pythonCmd -m ensurepip --upgrade
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[ERROR] Failed to install pip!" -ForegroundColor Red
        Read-Host "Press Enter to exit"
        exit 1
    }
    Write-Host "[OK] pip installed successfully" -ForegroundColor Green
}
Write-Host ""

# 3. Install dependencies
Write-Host "[3/6] Installing dependencies from requirements.txt..." -ForegroundColor Yellow
if (-not (Test-Path "requirements.txt")) {
    Write-Host "[ERROR] requirements.txt not found!" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

& $pythonCmd -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Failed to install dependencies!" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}
Write-Host "[OK] Dependencies installed successfully" -ForegroundColor Green
Write-Host ""

# 4. Check Docker
Write-Host "[4/6] Checking Docker..." -ForegroundColor Yellow
$dockerAvailable = $false
if (Test-Command "docker") {
    try {
        docker --version | Out-Null
        if ($LASTEXITCODE -eq 0) {
            Write-Host "[OK] Docker is installed" -ForegroundColor Green
            $dockerAvailable = $true
        }
    } catch {
        $dockerAvailable = $false
    }
}

if (-not $dockerAvailable) {
    Write-Host "[WARNING] Docker is not installed or not running" -ForegroundColor Yellow
    Write-Host "Program will use SQLite in-memory (for testing only)" -ForegroundColor Yellow
}
Write-Host ""

# 5. Start Docker containers
if ($dockerAvailable) {
    Write-Host "[5/6] Starting Docker containers (PostgreSQL & MongoDB)..." -ForegroundColor Yellow
    
    # Check docker-compose
    $composeCmd = $null
    if (Test-Command "docker-compose") {
        $composeCmd = "docker-compose"
    } elseif (Test-Command "docker") {
        try {
            docker compose version | Out-Null
            if ($LASTEXITCODE -eq 0) {
                $composeCmd = "docker compose"
            }
        } catch {}
    }
    
    if ($composeCmd) {
        # Check if containers are already running
        $running = docker ps --format "{{.Names}}" | Select-String -Pattern "trading_postgres|trading_mongodb"
        if ($running) {
            Write-Host "[OK] Docker containers are already running" -ForegroundColor Green
        } else {
            Invoke-Expression "$composeCmd up -d"
            if ($LASTEXITCODE -eq 0) {
                Write-Host "[OK] Docker containers started successfully" -ForegroundColor Green
                Write-Host "Waiting for PostgreSQL to be ready..." -ForegroundColor Yellow
                Start-Sleep -Seconds 5
            } else {
                Write-Host "[WARNING] Failed to start Docker containers" -ForegroundColor Yellow
                $dockerAvailable = $false
            }
        }
    } else {
        Write-Host "[WARNING] docker-compose is not available" -ForegroundColor Yellow
        $dockerAvailable = $false
    }
} else {
    Write-Host "[5/6] Skipping Docker (not available)" -ForegroundColor Yellow
}
Write-Host ""

# 6. Create necessary directories
Write-Host "[6/6] Creating necessary directories..." -ForegroundColor Yellow
@("data", "results") | ForEach-Object {
    if (-not (Test-Path $_)) {
        New-Item -ItemType Directory -Path $_ | Out-Null
    }
}
Write-Host "[OK] Directories created" -ForegroundColor Green
Write-Host ""

# 7. Run main program
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Starting main program..." -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Ensure we're in the project root directory
$scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptPath

# Set PYTHONPATH to include current directory
$env:PYTHONPATH = "$scriptPath;$env:PYTHONPATH"

& $pythonCmd backend/main.py

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "Completed!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Read-Host "Press Enter to exit"
