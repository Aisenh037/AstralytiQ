@echo off
REM Docker Build and Management Script for Enterprise SaaS Platform (Windows)
setlocal enabledelayedexpansion

REM Configuration
set COMPOSE_PROJECT_NAME=enterprise-saas
set REGISTRY_URL=%DOCKER_REGISTRY%
if "%REGISTRY_URL%"=="" set REGISTRY_URL=localhost:5000
if "%VERSION%"=="" set VERSION=latest

REM Function to check if Docker is running
docker info >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Docker is not running. Please start Docker and try again.
    exit /b 1
)

REM Main script logic
if "%1"=="" goto help
if "%1"=="help" goto help
if "%1"=="build" goto build
if "%1"=="start-dev" goto start_dev
if "%1"=="start-prod" goto start_prod
if "%1"=="stop" goto stop_all
if "%1"=="restart" goto restart
if "%1"=="logs" goto show_logs
if "%1"=="health" goto health_check
if "%1"=="test" goto run_tests
if "%1"=="cleanup" goto cleanup
if "%1"=="push" goto push_images
goto help

:build
if "%2"=="" (
    echo [INFO] Building all services...
    docker build -f Dockerfile.gateway -t %COMPOSE_PROJECT_NAME%-gateway:%VERSION% .
    docker build -f Dockerfile.user -t %COMPOSE_PROJECT_NAME%-user:%VERSION% .
    docker build -f Dockerfile.tenant -t %COMPOSE_PROJECT_NAME%-tenant:%VERSION% .
    docker build -f Dockerfile.data -t %COMPOSE_PROJECT_NAME%-data:%VERSION% .
    docker build -f Dockerfile.ml -t %COMPOSE_PROJECT_NAME%-ml:%VERSION% .
    echo [SUCCESS] All services built successfully!
) else (
    echo [INFO] Building %2 service...
    if not exist "Dockerfile.%2" (
        echo [ERROR] Dockerfile.%2 not found
        exit /b 1
    )
    docker build -f Dockerfile.%2 -t %COMPOSE_PROJECT_NAME%-%2:%VERSION% .
    echo [SUCCESS] %2 service built successfully!
)
goto end

:start_dev
echo [INFO] Starting development environment...
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up -d
echo [SUCCESS] Development environment started!
echo [INFO] Services available at:
echo   - API Gateway: http://localhost:8000
echo   - User Service: http://localhost:8001
echo   - Tenant Service: http://localhost:8002
echo   - Data Service: http://localhost:8003
echo   - ML Service: http://localhost:8004
goto end

:start_prod
echo [INFO] Starting production environment...
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
echo [SUCCESS] Production environment started!
echo [INFO] Services available at:
echo   - Load Balancer: http://localhost:80
echo   - HTTPS: https://localhost:443
goto end

:stop_all
echo [INFO] Stopping all services...
docker-compose -f docker-compose.yml -f docker-compose.dev.yml -f docker-compose.prod.yml down
echo [SUCCESS] All services stopped!
goto end

:restart
call :stop_all
timeout /t 2 /nobreak >nul
call :start_dev
goto end

:show_logs
if "%2"=="" (
    echo [INFO] Showing logs for all services...
    docker-compose logs -f
) else (
    echo [INFO] Showing logs for %2 service...
    docker-compose logs -f %2
)
goto end

:health_check
echo [INFO] Checking service health...
curl -f -s "http://localhost:8000/health" >nul 2>&1 && echo [SUCCESS] api-gateway is healthy || echo [ERROR] api-gateway is not responding
curl -f -s "http://localhost:8001/health" >nul 2>&1 && echo [SUCCESS] user-service is healthy || echo [ERROR] user-service is not responding
curl -f -s "http://localhost:8002/health" >nul 2>&1 && echo [SUCCESS] tenant-service is healthy || echo [ERROR] tenant-service is not responding
curl -f -s "http://localhost:8003/health" >nul 2>&1 && echo [SUCCESS] data-service is healthy || echo [ERROR] data-service is not responding
curl -f -s "http://localhost:8004/health" >nul 2>&1 && echo [SUCCESS] ml-service is healthy || echo [ERROR] ml-service is not responding
goto end

:run_tests
echo [INFO] Running tests...
docker-compose -f docker-compose.yml -f docker-compose.dev.yml exec api-gateway python -m pytest tests/ -v
echo [SUCCESS] Tests completed!
goto end

:cleanup
echo [INFO] Cleaning up Docker resources...
docker-compose -f docker-compose.yml -f docker-compose.dev.yml -f docker-compose.prod.yml down --remove-orphans
docker image prune -f
set /p cleanup_volumes="Do you want to remove unused volumes? This will delete data! (y/N): "
if /i "%cleanup_volumes%"=="y" (
    docker volume prune -f
    echo [WARNING] Volumes removed. Data may be lost!
)
echo [SUCCESS] Cleanup completed!
goto end

:push_images
if "%DOCKER_REGISTRY%"=="" (
    echo [ERROR] DOCKER_REGISTRY environment variable is not set
    exit /b 1
)
echo [INFO] Pushing images to registry: %REGISTRY_URL%
for %%s in (gateway user tenant data ml) do (
    echo [INFO] Tagging and pushing %%s...
    docker tag %COMPOSE_PROJECT_NAME%-%%s:%VERSION% %REGISTRY_URL%/%COMPOSE_PROJECT_NAME%-%%s:%VERSION%
    docker push %REGISTRY_URL%/%COMPOSE_PROJECT_NAME%-%%s:%VERSION%
    echo [SUCCESS] %%s pushed successfully
)
echo [SUCCESS] All images pushed to registry!
goto end

:help
echo Docker Build and Management Script for Enterprise SaaS Platform
echo.
echo Usage: %0 [COMMAND] [OPTIONS]
echo.
echo Commands:
echo   build [SERVICE]     Build all services or a specific service
echo   start-dev          Start development environment
echo   start-prod         Start production environment
echo   stop               Stop all services
echo   restart            Restart all services
echo   logs [SERVICE]     Show logs for all services or a specific service
echo   health             Check health of all services
echo   test               Run tests
echo   cleanup            Clean up Docker resources
echo   push               Push images to registry
echo   help               Show this help message
echo.
echo Environment Variables:
echo   VERSION            Image version tag (default: latest)
echo   DOCKER_REGISTRY    Docker registry URL for pushing images
echo.
echo Examples:
echo   %0 build                    # Build all services
echo   %0 build gateway            # Build only gateway service
echo   %0 start-dev               # Start development environment
echo   %0 logs api-gateway        # Show logs for API gateway
echo   set VERSION=v1.0.0 ^& %0 build    # Build with specific version

:end
endlocal