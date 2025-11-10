#!/bin/bash
# Migration helper script for Streamlit to Django conversion

set -e

echo "========================================="
echo "Clone Hero: Streamlit → Django Migration"
echo "========================================="
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if running from project root
if [ ! -f "docker-compose.yml" ]; then
    echo -e "${RED}Error: Please run this script from the project root directory${NC}"
    exit 1
fi

echo "This script will help you migrate from Streamlit to Django frontend."
echo ""
echo "Steps to be performed:"
echo "  1. Stop current services"
echo "  2. Rebuild frontend container"
echo "  3. Start services with new Django frontend"
echo "  4. Verify deployment"
echo ""
read -p "Continue? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Migration cancelled."
    exit 0
fi

echo ""
echo -e "${YELLOW}Step 1: Stopping current services...${NC}"
docker compose down

echo ""
echo -e "${YELLOW}Step 2: Rebuilding frontend container...${NC}"
docker compose build frontend

echo ""
echo -e "${YELLOW}Step 3: Starting all services...${NC}"
docker compose up -d

echo ""
echo -e "${YELLOW}Step 4: Waiting for services to be ready...${NC}"
sleep 10

echo ""
echo -e "${YELLOW}Checking service status...${NC}"
docker compose ps

echo ""
echo -e "${YELLOW}Checking frontend logs...${NC}"
docker compose logs --tail=20 frontend

echo ""
echo "========================================="
echo -e "${GREEN}Migration Complete!${NC}"
echo "========================================="
echo ""
echo "Access your Django frontend at: http://localhost:8501"
echo ""
echo "Useful commands:"
echo "  - View frontend logs:     docker compose logs -f frontend"
echo "  - Restart frontend:       docker compose restart frontend"
echo "  - Collect static files:   docker compose exec frontend python src/frontend_django/manage.py collectstatic --noinput"
echo "  - Access Django shell:    docker compose exec frontend python src/frontend_django/manage.py shell"
echo "  - Create admin user:      docker compose exec frontend python src/frontend_django/manage.py createsuperuser"
echo ""
echo "For detailed information, see MIGRATION_GUIDE.md"
echo ""

# Run a basic health check
echo -e "${YELLOW}Running health check...${NC}"
sleep 5
if curl -f http://localhost:8501/ > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Frontend is responding correctly!${NC}"
else
    echo -e "${RED}⚠️  Frontend health check failed. Check logs with: docker compose logs frontend${NC}"
fi

echo ""
echo "Done! 🎸"
