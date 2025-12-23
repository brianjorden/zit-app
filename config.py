"""
Configuration for Z-Image-Turbo App.

Handles environment variables, model paths, resolution definitions,
and runtime settings.
"""

import os
import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class AppConfig:
    """Application configuration with sensible defaults."""

    # Model paths (defaults to local downloads)
    transformer_path: str = "/home/brian/local/Z-Image-Turbo/transformer"
    tokenizer_path: str = "/home/brian/local/Z-Image-Turbo/tokenizer"
    vae_path: str = "/home/brian/local/Z-Image-Turbo/vae"
    scheduler_path: str = "/home/brian/local/Z-Image-Turbo/scheduler"
    text_encoder_path: str = "/home/brian/local/zit_text_encoder_chat"
    lora_dir: Path = field(default_factory=lambda: Path("/home/brian/local/zit-loras"))

    # Service configuration
    encoder_url: str | None = None  # If set, use remote encoder
    vae_url: str | None = None  # If set, use remote VAE

    # Device configuration
    transformer_device: str = "cuda:0"
    vae_device: str = "cuda:0"  # Can be same as transformer or separate
    encoder_device: str = "cuda:1"  # Must be different from transformer

    # Runtime options
    warmup_enabled: bool = False
    attention_backend: str = "flash"
    output_dir: Path = field(default_factory=lambda: Path("./output"))

    # Gradio settings
    server_name: str = "0.0.0.0"
    app_port: int = 7860
    encoder_port: int = 7888
    vae_port: int = 7999


def load_config() -> AppConfig:
    """Load configuration from environment variables with defaults."""
    config = AppConfig()

    # Model paths
    if path := os.environ.get("TRANSFORMER_PATH"):
        config.transformer_path = path
    if path := os.environ.get("TOKENIZER_PATH"):
        config.tokenizer_path = path
    if path := os.environ.get("VAE_PATH"):
        config.vae_path = path
    if path := os.environ.get("SCHEDULER_PATH"):
        config.scheduler_path = path
    if path := os.environ.get("TEXT_ENCODER_PATH"):
        config.text_encoder_path = path
    if path := os.environ.get("LORA_DIR"):
        config.lora_dir = Path(path)

    # Service URLs
    config.encoder_url = os.environ.get("ENCODER_URL")
    config.vae_url = os.environ.get("VAE_URL")

    # Device configuration
    if device := os.environ.get("TRANSFORMER_DEVICE"):
        config.transformer_device = device
    if device := os.environ.get("VAE_DEVICE"):
        config.vae_device = device
    if device := os.environ.get("ENCODER_DEVICE"):
        config.encoder_device = device

    # Runtime options
    config.warmup_enabled = os.environ.get("ENABLE_WARMUP", "false").lower() == "true"
    if backend := os.environ.get("ATTENTION_BACKEND"):
        config.attention_backend = backend
    if output := os.environ.get("OUTPUT_DIR"):
        config.output_dir = Path(output)

    # Gradio settings
    if name := os.environ.get("GRADIO_SERVER_NAME"):
        config.server_name = name
    if port := os.environ.get("APP_PORT"):
        config.app_port = int(port)
    if port := os.environ.get("ENCODER_PORT"):
        config.encoder_port = int(port)
    if port := os.environ.get("VAE_PORT"):
        config.vae_port = int(port)

    return config


# Resolution categories with aspect ratios
# All dimensions are divisible by 32 (required by VAE)
RESOLUTION_CHOICES: dict[str, list[str]] = {
    "1024": [
        "1024x1024 ( 1:1 )",
        "1152x896 ( 9:7 )",
        "896x1152 ( 7:9 )",
        "1152x864 ( 4:3 )",
        "864x1152 ( 3:4 )",
        "1248x832 ( 3:2 )",
        "832x1248 ( 2:3 )",
        "1280x720 ( 16:9 )",
        "720x1280 ( 9:16 )",
        "1344x576 ( 21:9 )",
        "576x1344 ( 9:21 )",
    ],
    "1280": [
        "1280x1280 ( 1:1 )",
        "1440x1120 ( 9:7 )",
        "1120x1440 ( 7:9 )",
        "1472x1104 ( 4:3 )",
        "1104x1472 ( 3:4 )",
        "1536x1024 ( 3:2 )",
        "1024x1536 ( 2:3 )",
        "1600x896 ( 16:9 )",
        "896x1600 ( 9:16 )",
        "1680x720 ( 21:9 )",
        "720x1680 ( 9:21 )",
    ],
    "1536": [
        "1536x1536 ( 1:1 )",
        "1728x1344 ( 9:7 )",
        "1344x1728 ( 7:9 )",
        "1760x1328 ( 4:3 )",
        "1328x1760 ( 3:4 )",
        "1872x1248 ( 3:2 )",
        "1248x1872 ( 2:3 )",
        "1920x1088 ( 16:9 )",
        "1088x1920 ( 9:16 )",
        "2016x864 ( 21:9 )",
        "864x2016 ( 9:21 )",
    ],
    "1792": [
        "1792x1792 ( 1:1 )",
        "2016x1568 ( 9:7 )",
        "1568x2016 ( 7:9 )",
        "2048x1536 ( 4:3 )",
        "1536x2048 ( 3:4 )",
        "2176x1456 ( 3:2 )",
        "1456x2176 ( 2:3 )",
        "2240x1264 ( 16:9 )",
        "1264x2240 ( 9:16 )",
        "2352x1008 ( 21:9 )",
        "1008x2352 ( 9:21 )",
    ],
    "2048": [
        "2048x2048 ( 1:1 )",
        "2304x1792 ( 9:7 )",
        "1792x2304 ( 7:9 )",
        "2368x1760 ( 4:3 )",
        "1760x2368 ( 3:4 )",
        "2496x1664 ( 3:2 )",
        "1664x2496 ( 2:3 )",
        "2560x1440 ( 16:9 )",
        "1440x2560 ( 9:16 )",
        "2688x1152 ( 21:9 )",
        "1152x2688 ( 9:21 )",
    ],
}

# Flat list of all resolutions
ALL_RESOLUTIONS: list[str] = []
for resolutions in RESOLUTION_CHOICES.values():
    ALL_RESOLUTIONS.extend(resolutions)


def get_resolution(resolution_str: str) -> tuple[int, int]:
    """
    Parse a resolution string into width and height.

    Args:
        resolution_str: String like "1024x1024 ( 1:1 )" or "1024x1024"

    Returns:
        Tuple of (width, height)
    """
    match = re.search(r"(\d+)\s*[×x]\s*(\d+)", resolution_str)
    if match:
        return int(match.group(1)), int(match.group(2))
    return 1024, 1024  # Default fallback


def get_resolution_category(resolution_str: str) -> str:
    """
    Get the category (base size) for a resolution string.

    Args:
        resolution_str: String like "1024x1024 ( 1:1 )"

    Returns:
        Category string like "1024"
    """
    width, height = get_resolution(resolution_str)
    max_dim = max(width, height)

    # Find the closest category
    categories = sorted([int(k) for k in RESOLUTION_CHOICES.keys()])
    for cat in categories:
        if max_dim <= cat + 128:  # Allow some tolerance
            return str(cat)
    return str(categories[-1])  # Default to largest


# Default generation parameters
DEFAULT_STEPS = 8
DEFAULT_SHIFT = 3.0
DEFAULT_GUIDANCE_SCALE = 0.0  # Must be 0 for Turbo model
DEFAULT_MAX_SEQUENCE_LENGTH = 2048
DEFAULT_ENCODER_SEED = 42
