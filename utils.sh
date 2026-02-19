#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# Clone Hero Content Manager - Utility Script
# Single-container simplified version
###############################################################################

# Debugging
DEBUG=false
[[ "$DEBUG" == "true" ]] && set -x

# Default non-interactive mode to false
NON_INTERACTIVE=false

# Parse command-line options
while getopts "y" opt; do
  case "$opt" in
    y)
      NON_INTERACTIVE=true
      ;;
    *)
      echo "Usage: $(basename "$0") [-y]"
      exit 1
      ;;
  esac
done
shift "$((OPTIND-1))"

# Logging
readonly LOG_DIR="/tmp/utils"
readonly LOG_FILE="${LOG_DIR}/utils.log"

# Docker Hub Settings
DOCKERHUB_USERNAME="nuniesmith"
DOCKERHUB_REPOSITORY="clonehero"

# Docker Compose
COMPOSE_FILE="docker-compose.yml"

# Image name
IMAGE_NAME="${DOCKERHUB_USERNAME}/${DOCKERHUB_REPOSITORY}:app"

###############################################################################
# INITIAL SETUP
###############################################################################

mkdir -p "$LOG_DIR"
chmod 700 "$LOG_DIR"

trap 'log_info "Script interrupted."; exit 130' INT
trap 'log_info "Script terminated."; exit 143' TERM

###############################################################################
# LOGGING FUNCTIONS
###############################################################################
log_info() {
    echo "[INFO]  $(date '+%Y-%m-%d %H:%M:%S') - $*"
    echo "[INFO]  $(date '+%Y-%m-%d %H:%M:%S') - $*" >> "$LOG_FILE"
}
log_warn() {
    echo "[WARN]  $(date '+%Y-%m-%d %H:%M:%S') - $*"
    echo "[WARN]  $(date '+%Y-%m-%d %H:%M:%S') - $*" >> "$LOG_FILE"
}
log_error() {
    echo "[ERROR] $(date '+%Y-%m-%d %H:%M:%S') - $*" >&2
    echo "[ERROR] $(date '+%Y-%m-%d %H:%M:%S') - $*" >> "$LOG_FILE"
}

confirm() {
    local prompt="$1"
    if [[ "$NON_INTERACTIVE" == true ]]; then
        echo "$prompt [y/N]: y (auto-confirmed)"
        return 0
    else
        read -r -p "$prompt [y/N]: " response
        [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]
    fi
}

###############################################################################
# ENVIRONMENT CHECK
###############################################################################
check_env_file() {
    if [[ ! -f ".env" ]]; then
        if [[ -f ".env.example" ]]; then
            log_warn ".env file not found. Copying from .env.example..."
            cp .env.example .env
            log_info "Created .env from .env.example — please review and update settings."
        else
            log_error "No .env or .env.example file found."
            exit 1
        fi
    fi
}

check_docker() {
    if ! command -v docker &>/dev/null; then
        log_error "Docker is not installed. Please install Docker first."
        log_info "Visit: https://docs.docker.com/get-docker/"
        exit 1
    fi
    log_info "Docker is available."
}

###############################################################################
# SERVICE MANAGEMENT
###############################################################################
start_service() {
    log_info "Starting Clone Hero Content Manager..."
    check_env_file

    if [[ ! -f "$COMPOSE_FILE" ]]; then
        log_error "Docker Compose file '$COMPOSE_FILE' not found."
        exit 1
    fi

    # Create data directories if they don't exist
    mkdir -p data/clonehero_content/{songs,backgrounds,colors,highways,generator,temp}
    mkdir -p data/logs

    docker compose -f "$COMPOSE_FILE" up -d --build
    log_info "Service is starting up..."

    # Wait for health check
    log_info "Waiting for service to become healthy..."
    local retries=15
    local delay=2
    for ((i=1; i<=retries; i++)); do
        if curl -sf http://localhost:${APP_PORT:-8000}/api/health > /dev/null 2>&1; then
            log_info "✅ Service is healthy and ready!"
            log_info "🌐 Open http://localhost:${APP_PORT:-8000} in your browser."
            return 0
        fi
        sleep "$delay"
    done

    log_warn "Service may still be starting. Check logs with: docker compose logs -f"
}

stop_service() {
    log_info "Stopping Clone Hero Content Manager..."
    docker compose -f "$COMPOSE_FILE" down --remove-orphans
    log_info "Service stopped."
}

restart_service() {
    log_info "Restarting Clone Hero Content Manager..."
    stop_service
    start_service
}

show_logs() {
    log_info "Showing service logs (Ctrl+C to exit)..."
    docker compose -f "$COMPOSE_FILE" logs -f
}

show_status() {
    echo ""
    echo "=== Service Status ==="
    docker compose -f "$COMPOSE_FILE" ps
    echo ""

    # Check health endpoint
    local port="${APP_PORT:-8000}"
    if curl -sf "http://localhost:${port}/api/health" > /dev/null 2>&1; then
        echo "Health: ✅ Healthy"
        curl -sf "http://localhost:${port}/api/health" 2>/dev/null | python3 -m json.tool 2>/dev/null || true
    else
        echo "Health: ❌ Not responding"
    fi
    echo ""
}

###############################################################################
# BUILD & PUSH
###############################################################################
build_image() {
    log_info "Building Docker image: ${IMAGE_NAME}..."
    docker build -f docker/Dockerfile -t "$IMAGE_NAME" .
    log_info "✅ Image built successfully: ${IMAGE_NAME}"
}

push_image() {
    log_info "Pushing Docker image: ${IMAGE_NAME}..."
    docker push "$IMAGE_NAME"
    log_info "✅ Image pushed to Docker Hub: ${IMAGE_NAME}"
}

build_and_push() {
    build_image
    if confirm "Push image to Docker Hub?"; then
        push_image
    fi
}

###############################################################################
# DEVELOPMENT
###############################################################################
dev_run() {
    log_info "Starting in development mode (with auto-reload)..."
    check_env_file

    mkdir -p data/clonehero_content/{songs,backgrounds,colors,highways,generator,temp}
    mkdir -p data/logs

    # Source .env for local development
    set -a
    # shellcheck disable=SC1091
    source .env 2>/dev/null || true
    set +a

    export PYTHONPATH="${PYTHONPATH:-$(pwd)}"
    export DATA_DIR="${DATA_DIR:-$(pwd)/data}"
    export LOG_DIR="${LOG_DIR:-$(pwd)/data/logs}"

    python -m uvicorn src.app.main:app \
        --host "${APP_HOST:-0.0.0.0}" \
        --port "${APP_PORT:-8000}" \
        --reload \
        --reload-dir src
}

###############################################################################
# CLEANUP
###############################################################################
docker_prune() {
    local choice="${1:-all}"
    case "$choice" in
        all)
            log_info "Pruning all unused Docker resources..."
            docker system prune -af --volumes
            ;;
        images)
            log_info "Pruning unused Docker images..."
            docker image prune -af
            ;;
        volumes)
            log_info "Pruning unused Docker volumes..."
            docker volume prune -f
            ;;
        containers)
            log_info "Pruning stopped Docker containers..."
            docker container prune -f
            ;;
        *)
            log_error "Invalid prune target: $choice"
            ;;
    esac
}

fix_permissions() {
    log_info "Fixing data directory permissions..."
    if [[ -d "data" ]]; then
        chmod -R 755 data/
        log_info "✅ Permissions fixed for data/"
    else
        log_warn "data/ directory not found."
    fi
}

###############################################################################
# BACKUP / RESTORE
###############################################################################
backup_data() {
    local timestamp
    timestamp=$(date '+%Y%m%d_%H%M%S')
    local backup_file="backup_clonehero_${timestamp}.tar.gz"

    if [[ ! -d "data" ]]; then
        log_error "No data/ directory to back up."
        return 1
    fi

    log_info "Creating backup: ${backup_file}..."
    tar -czf "$backup_file" data/
    log_info "✅ Backup created: ${backup_file} ($(du -h "$backup_file" | cut -f1))"
}

###############################################################################
# MAIN MENU
###############################################################################
display_main_menu() {
    echo ""
    echo "========================================"
    echo "  Clone Hero Content Manager  v2.0"
    echo "========================================"
    echo ""
    echo "  Service:"
    echo "    [0] Start service"
    echo "    [1] Stop service"
    echo "    [2] Restart service"
    echo "    [3] Show status"
    echo "    [4] View logs"
    echo ""
    echo "  Development:"
    echo "    [5] Run locally (dev mode)"
    echo ""
    echo "  Build:"
    echo "    [6] Build & push Docker image"
    echo ""
    echo "  Maintenance:"
    echo "    [7] Fix data permissions"
    echo "    [8] Backup data"
    echo "    [9] Docker cleanup (prune all)"
    echo ""
    echo "    [q] Quit"
    echo ""
    echo "========================================"
}

main() {
    check_docker

    # Non-interactive mode: just start the service
    if [[ "$NON_INTERACTIVE" == true ]]; then
        start_service
        exit 0
    fi

    while true; do
        display_main_menu
        read -r -p "Select an option: " choice
        echo ""
        case "$choice" in
            0) start_service ;;
            1) stop_service ;;
            2) restart_service ;;
            3) show_status ;;
            4) show_logs ;;
            5) dev_run ;;
            6) build_and_push ;;
            7) fix_permissions ;;
            8) backup_data ;;
            9) docker_prune all ;;
            q|Q)
                log_info "Exiting."
                exit 0
                ;;
            *)
                log_error "Invalid choice: $choice"
                ;;
        esac
    done
}

main "$@"
