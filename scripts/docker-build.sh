#!/bin/bash

# Docker Build and Management Script for Enterprise SaaS Platform
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
COMPOSE_PROJECT_NAME="enterprise-saas"
REGISTRY_URL="${DOCKER_REGISTRY:-localhost:5000}"
VERSION="${VERSION:-latest}"

# Function to print colored output
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

# Function to check if Docker is running
check_docker() {
    if ! docker info > /dev/null 2>&1; then
        print_error "Docker is not running. Please start Docker and try again."
        exit 1
    fi
}

# Function to build all services
build_all() {
    print_status "Building all services..."
    
    services=("gateway" "user" "tenant" "data" "ml")
    
    for service in "${services[@]}"; do
        print_status "Building $service service..."
        docker build -f "Dockerfile.$service" -t "$COMPOSE_PROJECT_NAME-$service:$VERSION" .
        print_success "$service service built successfully"
    done
    
    print_success "All services built successfully!"
}

# Function to build a specific service
build_service() {
    local service=$1
    if [ -z "$service" ]; then
        print_error "Service name is required"
        exit 1
    fi
    
    if [ ! -f "Dockerfile.$service" ]; then
        print_error "Dockerfile.$service not found"
        exit 1
    fi
    
    print_status "Building $service service..."
    docker build -f "Dockerfile.$service" -t "$COMPOSE_PROJECT_NAME-$service:$VERSION" .
    print_success "$service service built successfully!"
}

# Function to start development environment
start_dev() {
    print_status "Starting development environment..."
    docker-compose -f docker-compose.yml -f docker-compose.dev.yml up -d
    print_success "Development environment started!"
    print_status "Services available at:"
    echo "  - API Gateway: http://localhost:8000"
    echo "  - User Service: http://localhost:8001"
    echo "  - Tenant Service: http://localhost:8002"
    echo "  - Data Service: http://localhost:8003"
    echo "  - ML Service: http://localhost:8004"
}

# Function to start production environment
start_prod() {
    print_status "Starting production environment..."
    docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
    print_success "Production environment started!"
    print_status "Services available at:"
    echo "  - Load Balancer: http://localhost:80"
    echo "  - HTTPS: https://localhost:443"
}

# Function to stop all services
stop_all() {
    print_status "Stopping all services..."
    docker-compose -f docker-compose.yml -f docker-compose.dev.yml -f docker-compose.prod.yml down
    print_success "All services stopped!"
}

# Function to clean up Docker resources
cleanup() {
    print_status "Cleaning up Docker resources..."
    
    # Stop and remove containers
    docker-compose -f docker-compose.yml -f docker-compose.dev.yml -f docker-compose.prod.yml down --remove-orphans
    
    # Remove unused images
    docker image prune -f
    
    # Remove unused volumes (with confirmation)
    read -p "Do you want to remove unused volumes? This will delete data! (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        docker volume prune -f
        print_warning "Volumes removed. Data may be lost!"
    fi
    
    print_success "Cleanup completed!"
}

# Function to show logs
show_logs() {
    local service=$1
    if [ -z "$service" ]; then
        print_status "Showing logs for all services..."
        docker-compose logs -f
    else
        print_status "Showing logs for $service service..."
        docker-compose logs -f "$service"
    fi
}

# Function to run tests
run_tests() {
    print_status "Running tests..."
    docker-compose -f docker-compose.yml -f docker-compose.dev.yml exec api-gateway python -m pytest tests/ -v
    print_success "Tests completed!"
}

# Function to check service health
health_check() {
    print_status "Checking service health..."
    
    services=("api-gateway:8000" "user-service:8001" "tenant-service:8002" "data-service:8003" "ml-service:8004")
    
    for service in "${services[@]}"; do
        service_name=$(echo $service | cut -d: -f1)
        port=$(echo $service | cut -d: -f2)
        
        if curl -f -s "http://localhost:$port/health" > /dev/null; then
            print_success "$service_name is healthy"
        else
            print_error "$service_name is not responding"
        fi
    done
}

# Function to push images to registry
push_images() {
    if [ -z "$DOCKER_REGISTRY" ]; then
        print_error "DOCKER_REGISTRY environment variable is not set"
        exit 1
    fi
    
    print_status "Pushing images to registry: $REGISTRY_URL"
    
    services=("gateway" "user" "tenant" "data" "ml")
    
    for service in "${services[@]}"; do
        local_tag="$COMPOSE_PROJECT_NAME-$service:$VERSION"
        remote_tag="$REGISTRY_URL/$COMPOSE_PROJECT_NAME-$service:$VERSION"
        
        print_status "Tagging and pushing $service..."
        docker tag "$local_tag" "$remote_tag"
        docker push "$remote_tag"
        print_success "$service pushed successfully"
    done
    
    print_success "All images pushed to registry!"
}

# Function to show help
show_help() {
    echo "Docker Build and Management Script for Enterprise SaaS Platform"
    echo ""
    echo "Usage: $0 [COMMAND] [OPTIONS]"
    echo ""
    echo "Commands:"
    echo "  build [SERVICE]     Build all services or a specific service"
    echo "  start-dev          Start development environment"
    echo "  start-prod         Start production environment"
    echo "  stop               Stop all services"
    echo "  restart            Restart all services"
    echo "  logs [SERVICE]     Show logs for all services or a specific service"
    echo "  health             Check health of all services"
    echo "  test               Run tests"
    echo "  cleanup            Clean up Docker resources"
    echo "  push               Push images to registry"
    echo "  help               Show this help message"
    echo ""
    echo "Environment Variables:"
    echo "  VERSION            Image version tag (default: latest)"
    echo "  DOCKER_REGISTRY    Docker registry URL for pushing images"
    echo ""
    echo "Examples:"
    echo "  $0 build                    # Build all services"
    echo "  $0 build gateway            # Build only gateway service"
    echo "  $0 start-dev               # Start development environment"
    echo "  $0 logs api-gateway        # Show logs for API gateway"
    echo "  VERSION=v1.0.0 $0 build    # Build with specific version"
}

# Main script logic
main() {
    check_docker
    
    case "${1:-help}" in
        "build")
            if [ -n "$2" ]; then
                build_service "$2"
            else
                build_all
            fi
            ;;
        "start-dev")
            start_dev
            ;;
        "start-prod")
            start_prod
            ;;
        "stop")
            stop_all
            ;;
        "restart")
            stop_all
            sleep 2
            start_dev
            ;;
        "logs")
            show_logs "$2"
            ;;
        "health")
            health_check
            ;;
        "test")
            run_tests
            ;;
        "cleanup")
            cleanup
            ;;
        "push")
            push_images
            ;;
        "help"|*)
            show_help
            ;;
    esac
}

# Run main function with all arguments
main "$@"