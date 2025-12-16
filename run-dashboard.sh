#!/bin/bash

# gRINN Dashboard Runner
# Helper script to run the dashboard with proper Docker configuration

set -e

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_usage() {
    cat << EOF
Usage: $0 <results_folder> [OPTIONS]

Run gRINN dashboard with Docker

Arguments:
  results_folder    Path to the gRINN results folder (required)

Options:
  --port PORT       Port to expose dashboard on (default: 8060)
  --with-chatbot    Enable AI chatbot (requires GEMINI_API_KEY)
  --image IMAGE     Docker image to use (default: grinn-dashboard:latest)
  --help, -h        Show this help message

Environment Variables (when using --with-chatbot):
  GEMINI_API_KEY          Your Gemini API key (for Gemini models)
  ANTHROPIC_API_KEY       Your Anthropic API key (for Claude models)
  PANDASAI_MODELS         Available models list (JSON array or comma-separated)
                          Example: '["gemini/gemini-pro-latest","claude-sonnet-4-20250514"]'
  PANDASAI_DEFAULT_MODEL  Default model to preselect
  PANDASAI_TOKEN_LIMIT    Token budget per session (0 or empty = unlimited)

Examples:
  # Basic dashboard (no chatbot)
  $0 /path/to/results

  # Dashboard with AI chatbot (Gemini)
  export GEMINI_API_KEY="your-gemini-key"
  $0 /path/to/results --with-chatbot

  # Dashboard with AI chatbot (Claude)
  export ANTHROPIC_API_KEY="your-anthropic-key"
  export PANDASAI_MODELS='["claude-sonnet-4-20250514"]'
  $0 /path/to/results --with-chatbot

  # Dashboard with multiple models
  export GEMINI_API_KEY="your-gemini-key"
  export ANTHROPIC_API_KEY="your-anthropic-key"
  export PANDASAI_MODELS='["gemini/gemini-pro-latest","claude-sonnet-4-20250514"]'
  $0 /path/to/results --with-chatbot

  # Custom port
  $0 /path/to/results --port 8080

  # Different image
  $0 /path/to/results --image grinn:gromacs-2024.1

EOF
}

# Default values
PORT=8060
WITH_CHATBOT=false
IMAGE="grinn-dashboard:latest"
RESULTS_FOLDER=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --port)
            PORT="$2"
            shift 2
            ;;
        --with-chatbot)
            WITH_CHATBOT=true
            shift
            ;;
        --image)
            IMAGE="$2"
            shift 2
            ;;
        --help|-h)
            print_usage
            exit 0
            ;;
        *)
            if [[ -z "$RESULTS_FOLDER" ]]; then
                RESULTS_FOLDER="$1"
                shift
            else
                echo -e "${RED}Error: Unknown option: $1${NC}"
                print_usage
                exit 1
            fi
            ;;
    esac
done

# Validate required arguments
if [[ -z "$RESULTS_FOLDER" ]]; then
    echo -e "${RED}Error: results_folder is required${NC}"
    print_usage
    exit 1
fi

# Validate results folder exists
if [[ ! -d "$RESULTS_FOLDER" ]]; then
    echo -e "${RED}Error: Results folder does not exist: $RESULTS_FOLDER${NC}"
    exit 1
fi

# Convert to absolute path
RESULTS_FOLDER=$(cd "$RESULTS_FOLDER" && pwd)

# Check if Docker is running
if ! docker info >/dev/null 2>&1; then
    echo -e "${RED}Error: Docker is not running${NC}"
    exit 1
fi

# Check if image exists
if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    echo -e "${RED}Error: Docker image not found: $IMAGE${NC}"
    echo -e "${YELLOW}Build the image first with: ./build-grinn.sh --dashboard-only${NC}"
    exit 1
fi

# Build Docker command
DOCKER_CMD="docker run --rm -it"
DOCKER_CMD="$DOCKER_CMD -p $PORT:8060"
DOCKER_CMD="$DOCKER_CMD -v $RESULTS_FOLDER:/data"

# Add chatbot requirements if enabled
if [[ "$WITH_CHATBOT" == "true" ]]; then
    echo -e "${BLUE}🤖 Chatbot enabled${NC}"
    
    # Check for at least one API key
    if [[ -z "$GEMINI_API_KEY" ]] && [[ -z "$GOOGLE_API_KEY" ]] && [[ -z "$ANTHROPIC_API_KEY" ]]; then
        echo -e "${RED}Error: At least one API key is required for chatbot${NC}"
        echo -e "${YELLOW}Set one of:${NC}"
        echo -e "${YELLOW}  export GEMINI_API_KEY='your-gemini-key'${NC}"
        echo -e "${YELLOW}  export ANTHROPIC_API_KEY='your-anthropic-key'${NC}"
        exit 1
    fi
    
    # Mount Docker socket
    DOCKER_CMD="$DOCKER_CMD -v /var/run/docker.sock:/var/run/docker.sock"
    
    # Pass API key
    if [[ -n "$GEMINI_API_KEY" ]]; then
        DOCKER_CMD="$DOCKER_CMD -e GEMINI_API_KEY=$GEMINI_API_KEY"
    elif [[ -n "$GOOGLE_API_KEY" ]]; then
        DOCKER_CMD="$DOCKER_CMD -e GOOGLE_API_KEY=$GOOGLE_API_KEY"
    fi
    
    # Pass model if set
    if [[ -n "$PANDASAI_MODEL" ]]; then
        DOCKER_CMD="$DOCKER_CMD -e PANDASAI_MODEL=$PANDASAI_MODEL"
    fi

    # Pass Anthropic API key if set
    if [[ -n "$ANTHROPIC_API_KEY" ]]; then
        DOCKER_CMD="$DOCKER_CMD -e ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY"
    fi

    # Pass model list/default if set
    if [[ -n "$PANDASAI_MODELS" ]]; then
        DOCKER_CMD="$DOCKER_CMD -e PANDASAI_MODELS=$PANDASAI_MODELS"
    fi
    if [[ -n "$PANDASAI_DEFAULT_MODEL" ]]; then
        DOCKER_CMD="$DOCKER_CMD -e PANDASAI_DEFAULT_MODEL=$PANDASAI_DEFAULT_MODEL"
    fi

    # Pass token limit if set
    if [[ -n "$PANDASAI_TOKEN_LIMIT" ]]; then
        DOCKER_CMD="$DOCKER_CMD -e PANDASAI_TOKEN_LIMIT=$PANDASAI_TOKEN_LIMIT"
    fi
    
    echo -e "${GREEN}✓ Docker socket mounted for sandbox execution${NC}"
    echo -e "${GREEN}✓ API key(s) configured${NC}"
else
    echo -e "${YELLOW}⚠️  Chatbot disabled. Use --with-chatbot to enable AI features.${NC}"
fi

# Add image and command
DOCKER_CMD="$DOCKER_CMD $IMAGE dashboard /data"

# Display info
echo -e "${BLUE}================================================================${NC}"
echo -e "${GREEN}Starting gRINN Dashboard${NC}"
echo -e "${BLUE}================================================================${NC}"
echo -e "Results folder: ${YELLOW}$RESULTS_FOLDER${NC}"
echo -e "Dashboard URL:  ${GREEN}http://localhost:$PORT${NC}"
echo -e "Image:          ${YELLOW}$IMAGE${NC}"
echo -e "Chatbot:        ${YELLOW}$([ "$WITH_CHATBOT" == "true" ] && echo "Enabled" || echo "Disabled")${NC}"
echo -e "${BLUE}================================================================${NC}"
echo ""

# Run the container
echo -e "${BLUE}Running:${NC} $DOCKER_CMD"
echo ""
eval "$DOCKER_CMD"
