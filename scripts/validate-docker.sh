#!/bin/bash

# Docker Setup Validation Script
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_status "Validating Docker containerization setup..."

# Check if all Dockerfiles exist
dockerfiles=("Dockerfile.gateway" "Dockerfile.user" "Dockerfile.tenant" "Dockerfile.data" "Dockerfile.ml")
missing_dockerfiles=()

for dockerfile in "${dockerfiles[@]}"; do
    if [ -f "$dockerfile" ]; then
        print_success "$dockerfile exists"
    else
        print_error "$dockerfile is missing"
        missing_dockerfiles+=("$dockerfile")
    fi
done

# Check if docker-compose files exist
compose_files=("docker-compose.yml" "docker-compose.dev.yml" "docker-compose.prod.yml")
missing_compose=()

for compose_file in "${compose_files[@]}"; do
    if [ -f "$compose_file" ]; then
        print_success "$compose_file exists"
    else
        print_error "$compose_file is missing"
        missing_compose+=("$compose_file")
    fi
done

# Check if requirements file exists
if [ -f "requirements-enterprise.txt" ]; then
    print_success "requirements-enterprise.txt exists"
else
    print_error "requirements-enterprise.txt is missing"
fi

# Check if build scripts exist
if [ -f "scripts/docker-build.sh" ]; then
    print_success "scripts/docker-build.sh exists"
else
    print_error "scripts/docker-build.sh is missing"
fi

if [ -f "scripts/docker-build.bat" ]; then
    print_success "scripts/docker-build.bat exists"
else
    print_error "scripts/docker-build.bat is missing"
fi

# Check if .dockerignore exists
if [ -f ".dockerignore" ]; then
    print_success ".dockerignore exists"
else
    print_warning ".dockerignore is missing (recommended for build optimization)"
fi

# Check if nginx.conf exists
if [ -f "nginx.conf" ]; then
    print_success "nginx.conf exists"
else
    print_warning "nginx.conf is missing (needed for production load balancing)"
fi

# Validate docker-compose.yml syntax
if command -v docker-compose >/dev/null 2>&1; then
    print_status "Validating docker-compose.yml syntax..."
    if docker-compose config >/dev/null 2>&1; then
        print_success "docker-compose.yml syntax is valid"
    else
        print_error "docker-compose.yml has syntax errors"
    fi
else
    print_warning "docker-compose not available for syntax validation"
fi

# Summary
echo ""
print_status "Validation Summary:"

if [ ${#missing_dockerfiles[@]} -eq 0 ] && [ ${#missing_compose[@]} -eq 0 ]; then
    print_success "All required Docker files are present!"
    print_status "You can now:"
    echo "  1. Start Docker Desktop (if not running)"
    echo "  2. Run: ./scripts/docker-build.sh build"
    echo "  3. Run: ./scripts/docker-build.sh start-dev"
else
    print_error "Some required files are missing:"
    for file in "${missing_dockerfiles[@]}" "${missing_compose[@]}"; do
        echo "  - $file"
    done
fi

echo ""
print_status "Docker containerization setup validation complete!"