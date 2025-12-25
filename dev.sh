#!/bin/bash
# Development server script for Tensorscope
# Usage:
#   ./dev.sh        - Start both API and web servers
#   ./dev.sh --api  - Start API server only
#   ./dev.sh --web  - Start web server only

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Trap for graceful shutdown
cleanup() {
    echo -e "\n${YELLOW}Shutting down servers...${NC}"
    if [ -n "$API_PID" ]; then
        kill $API_PID 2>/dev/null || true
    fi
    if [ -n "$WEB_PID" ]; then
        kill $WEB_PID 2>/dev/null || true
    fi
    wait 2>/dev/null
    echo -e "${GREEN}Servers stopped.${NC}"
    exit 0
}

trap cleanup SIGINT SIGTERM

# Check if virtual environment exists
check_venv() {
    if [ ! -d ".venv" ]; then
        echo -e "${YELLOW}Virtual environment not found. Creating one...${NC}"
        python3 -m venv .venv
        echo -e "${GREEN}Virtual environment created.${NC}"
    fi
}

# Activate virtual environment
activate_venv() {
    source .venv/bin/activate
}

# Check if package is installed
check_package() {
    if ! python -c "import tensorscope" 2>/dev/null; then
        echo -e "${YELLOW}Tensorscope not installed. Installing in dev mode...${NC}"
        pip install -e ".[dev]"
        echo -e "${GREEN}Tensorscope installed.${NC}"
    fi
}

# Start API server
start_api() {
    echo -e "${GREEN}Starting API server on http://localhost:8000${NC}"
    uvicorn tensorscope.server.main:app --reload --host 0.0.0.0 --port 8000 &
    API_PID=$!
}

# Start web server
start_web() {
    if [ -d "web/node_modules" ]; then
        echo -e "${GREEN}Starting web server on http://localhost:5173${NC}"
        cd web && npm run dev &
        WEB_PID=$!
        cd ..
    else
        echo -e "${YELLOW}Web dependencies not installed. Run 'cd web && npm install' first.${NC}"
        echo -e "${YELLOW}Skipping web server for now.${NC}"
    fi
}

# Parse arguments
RUN_API=false
RUN_WEB=false

if [ $# -eq 0 ]; then
    RUN_API=true
    RUN_WEB=true
else
    for arg in "$@"; do
        case $arg in
            --api)
                RUN_API=true
                ;;
            --web)
                RUN_WEB=true
                ;;
            --help|-h)
                echo "Usage: ./dev.sh [--api] [--web]"
                echo "  --api   Start API server only"
                echo "  --web   Start web server only"
                echo "  (no args) Start both servers"
                exit 0
                ;;
            *)
                echo -e "${RED}Unknown option: $arg${NC}"
                exit 1
                ;;
        esac
    done
fi

# Main execution
echo -e "${GREEN}=== Tensorscope Development Server ===${NC}"

check_venv
activate_venv
check_package

if $RUN_API; then
    start_api
fi

if $RUN_WEB; then
    start_web
fi

# Wait for processes
echo -e "\n${GREEN}Servers running. Press Ctrl+C to stop.${NC}"
wait
