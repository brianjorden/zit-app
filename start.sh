#!/bin/bash
#
# Z-Image-Turbo Launcher
#
# Usage:
#   ./start.sh app [gpu]                    - Start main app on GPU (default: 0)
#   ./start.sh encoder [device]             - Start encoder service (gpu number or "cpu")
#   ./start.sh vae [gpu]                    - Start VAE service on GPU
#   ./start.sh minimal [app_gpu] [enc_dev]  - App + encoder (VAE in app)
#   ./start.sh full [app] [enc] [vae]       - All three services
#

set -euo pipefail

cd "$(dirname "$0")"
source .venv/bin/activate

# Common environment settings
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export GRADIO_SERVER_NAME=0.0.0.0

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
    exit 1
}

show_help() {
    echo "Z-Image-Turbo Launcher"
    echo ""
    echo "Usage: $0 <command> [options]"
    echo ""
    echo "Commands:"
    echo "  app [gpu] [flags]          Start main app on specified GPU (default: 0)"
    echo "  encoder [device]           Start encoder service on port 7888"
    echo "  vae [gpu]                  Start VAE service on port 7999"
    echo "  minimal [app_gpu] [enc]    Start app + encoder (VAE runs in app)"
    echo "  full [app] [enc] [vae]     Start all three services"
    echo ""
    echo "App Flags:"
    echo "  --encoder-api              Use remote encoder at localhost:7888"
    echo "  --vae-api                  Use remote VAE at localhost:7999"
    echo ""
    echo "Examples:"
    echo "  $0 app 0                   # Main app on cuda:0 (local encoder/VAE)"
    echo "  $0 app 0 --encoder-api     # App on cuda:0, use remote encoder"
    echo "  $0 app 0 --encoder-api --vae-api  # Use both remote services"
    echo "  $0 encoder 4               # Encoder on cuda:4 (port 7888)"
    echo "  $0 encoder cpu             # Encoder on CPU"
    echo "  $0 vae 4                   # VAE on cuda:4 (port 7999)"
    echo "  $0 minimal 0 1             # App on cuda:0, encoder on cuda:1"
    echo "  $0 full 0 1 2              # Full distribution across 3 GPUs"
    echo ""
}

start_app() {
    local gpu="${1:-0}"
    shift || true

    local use_encoder_api=false
    local use_vae_api=false
    local use_debug=false

    # Parse flags
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --encoder-api)
                use_encoder_api=true
                shift
                ;;
            --vae-api)
                use_vae_api=true
                shift
                ;;
            --debug)
                use_debug=true
                shift
                ;;
            *)
                warn "Unknown flag: $1"
                shift
                ;;
        esac
    done

    info "Starting main app on cuda:$gpu..."

    if [ "$use_encoder_api" = true ]; then
        export ENCODER_URL="http://localhost:7888"
        info "  Using remote encoder at $ENCODER_URL"
    fi

    if [ "$use_vae_api" = true ]; then
        export VAE_URL="http://localhost:7999"
        info "  Using remote VAE at $VAE_URL"
    fi

    local debug_flag=""
    if [ "$use_debug" = true ]; then
        debug_flag="--debug"
        info "  Debug mode enabled"
    fi

    CUDA_VISIBLE_DEVICES=$gpu python app.py $debug_flag
}

start_encoder() {
    local device="${1:-1}"
    if [ "$device" = "cpu" ]; then
        info "Starting encoder service on CPU..."
        python encoder.py --device cpu
    else
        info "Starting encoder service on cuda:$device..."
        CUDA_VISIBLE_DEVICES=$device python encoder.py --device cuda:0
    fi
}

start_vae() {
    local gpu="${1:-2}"
    info "Starting VAE service on cuda:$gpu..."
    CUDA_VISIBLE_DEVICES=$gpu python vae.py --device cuda:0
}

start_minimal() {
    local app_gpu="${1:-0}"
    local enc_device="${2:-1}"

    info "Starting minimal configuration..."
    info "  App: cuda:$app_gpu"
    info "  Encoder: $enc_device"

    # Start encoder in background
    if [ "$enc_device" = "cpu" ]; then
        python encoder.py --device cpu &
    else
        CUDA_VISIBLE_DEVICES=$enc_device python encoder.py --device cuda:0 &
    fi
    local enc_pid=$!

    # Wait a moment for encoder to start
    sleep 5

    # Set encoder URL for app
    export ENCODER_URL="http://localhost:7888"

    # Start app in foreground
    CUDA_VISIBLE_DEVICES=$app_gpu python app.py

    # Cleanup
    kill $enc_pid 2>/dev/null || true
}

start_full() {
    local app_gpu="${1:-0}"
    local enc_device="${2:-1}"
    local vae_gpu="${3:-2}"

    info "Starting full distribution..."
    info "  App: cuda:$app_gpu"
    info "  Encoder: $enc_device"
    info "  VAE: cuda:$vae_gpu"

    # Start encoder in background
    if [ "$enc_device" = "cpu" ]; then
        python encoder.py --device cpu &
    else
        CUDA_VISIBLE_DEVICES=$enc_device python encoder.py --device cuda:0 &
    fi
    local enc_pid=$!

    # Start VAE in background
    CUDA_VISIBLE_DEVICES=$vae_gpu python vae.py --device cuda:0 &
    local vae_pid=$!

    # Wait a moment for services to start
    sleep 5

    # Set service URLs for app
    export ENCODER_URL="http://localhost:7888"
    export VAE_URL="http://localhost:7999"

    # Start app in foreground
    CUDA_VISIBLE_DEVICES=$app_gpu python app.py

    # Cleanup
    kill $enc_pid 2>/dev/null || true
    kill $vae_pid 2>/dev/null || true
}

# Main
if [ $# -lt 1 ]; then
    show_help
    exit 0
fi

command="$1"
shift

case "$command" in
    app)
        start_app "$@"
        ;;
    encoder)
        start_encoder "$@"
        ;;
    vae)
        start_vae "$@"
        ;;
    minimal)
        start_minimal "$@"
        ;;
    full)
        start_full "$@"
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        error "Unknown command: $command"
        ;;
esac
