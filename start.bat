@echo off
chcp 65001 >nul 2>&1
REM BTCUSDT Trading System Optimization Platform - Windows Startup Script

REM Change to script directory to ensure we're in the project root
cd /d "%~dp0"

echo ========================================
echo BTCUSDT Trading System Optimization Platform
echo ========================================
echo.

REM 1. Check Python
echo [1/6] Checking Python...
set PYTHON_CMD=
set PYTHON_FOUND=0

REM Try python command
python --version >nul 2>&1
if errorlevel 1 goto try_python3
set PYTHON_CMD=python
set PYTHON_FOUND=1
goto python_verified

:try_python3
python3 --version >nul 2>&1
if errorlevel 1 goto try_py
set PYTHON_CMD=python3
set PYTHON_FOUND=1
goto python_verified

:try_py
py --version >nul 2>&1
if errorlevel 1 goto python_not_found
set PYTHON_CMD=py
set PYTHON_FOUND=1
goto python_verified

:python_not_found
echo [ERROR] Python is not installed!
echo.
echo Please install Python 3.11+ from one of these options:
echo 1. Download from: https://www.python.org/downloads/
echo    - Select Python 3.11 or newer
echo    - IMPORTANT: Check "Add Python to PATH" during installation
echo.
echo 2. Or install via Microsoft Store:
echo    - Open Microsoft Store
echo    - Search for "Python 3.11" or "Python 3.12"
echo    - Click Install
echo.
echo 3. Or install via winget (if available):
echo    winget install Python.Python.3.11
echo.
pause
exit /b 1

:python_verified
REM Verify Python version
for /f "tokens=2" %%i in ('%PYTHON_CMD% --version 2^>^&1') do set PYTHON_VERSION=%%i
echo [OK] Python found: %PYTHON_VERSION%

echo.

REM 2. Check pip
echo [2/6] Checking pip...
%PYTHON_CMD% -m pip --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] pip is not installed!
    echo Attempting to install pip...
    %PYTHON_CMD% -m ensurepip --upgrade
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to install pip!
        pause
        exit /b 1
    )
)
echo [OK] pip is ready
echo.

REM 3. Install dependencies
echo [3/6] Installing dependencies from requirements.txt...
if not exist requirements.txt (
    echo [ERROR] requirements.txt not found!
    pause
    exit /b 1
)
%PYTHON_CMD% -m pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo [ERROR] Failed to install dependencies!
    pause
    exit /b 1
)
echo [OK] Dependencies installed successfully
echo.

REM 4. Check Docker
echo [4/6] Checking Docker...
docker --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [WARNING] Docker is not installed or not running
    echo Program will use SQLite in-memory (for testing only)
    set DOCKER_AVAILABLE=false
) else (
    echo [OK] Docker is installed
    set DOCKER_AVAILABLE=true
)
echo.

REM 5. Start Docker containers
if "%DOCKER_AVAILABLE%"=="true" goto start_docker_containers
echo [5/6] Skipping Docker - not available
goto docker_section_done

:start_docker_containers
echo [5/6] Starting Docker containers PostgreSQL and MongoDB...

REM Check docker-compose
set COMPOSE_CMD=
docker-compose --version >nul 2>&1
if errorlevel 1 goto check_compose_v2
set COMPOSE_CMD=docker-compose
goto compose_found

:check_compose_v2
docker compose version >nul 2>&1
if errorlevel 1 goto compose_not_found
set COMPOSE_CMD=docker compose
goto compose_found

:compose_not_found
echo [WARNING] docker-compose is not available
set DOCKER_AVAILABLE=false
goto docker_section_done

:compose_found
if "%DOCKER_AVAILABLE%" neq "true" goto docker_section_done
REM Check if containers are already running
docker ps | findstr "trading_postgres trading_mongodb" >nul
if errorlevel 1 goto start_docker
echo [OK] Docker containers are already running
goto docker_section_done

:start_docker
if "%COMPOSE_CMD%"=="docker-compose" (
    docker-compose up -d
) else (
    docker compose up -d
)
if errorlevel 1 (
    echo [WARNING] Failed to start Docker containers
    set DOCKER_AVAILABLE=false
) else (
    echo [OK] Docker containers started successfully
    echo Waiting for PostgreSQL to be ready...
    timeout /t 5 /nobreak >nul
)

:docker_section_done
echo.

REM 6. Create necessary directories
echo [6/6] Creating necessary directories...
if not exist data mkdir data
if not exist results mkdir results
echo [OK] Directories created
echo.

REM 7. Run main program
echo ========================================
echo Starting main program...
echo ========================================
echo.

REM Set PYTHONPATH to include current directory (we're already in project root)
set PYTHONPATH=%CD%;%PYTHONPATH%

REM Run the main program
%PYTHON_CMD% backend/main.py

echo.
echo ========================================
echo Completed!
echo ========================================
pause
