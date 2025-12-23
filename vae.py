#!/usr/bin/env python3
"""
VAE Microservice for Z-Image-Turbo.

A standalone Gradio app that provides:
1. Image to latent encoding API
2. Latent to image decoding API
3. Interactive test interface

Usage:
    python vae.py --device cuda:2 --port 7999
"""

import argparse
import base64
import io
import json
import os
import warnings
from pathlib import Path

import gradio as gr
import numpy as np
import torch
from diffusers import AutoencoderKL
from PIL import Image
from safetensors.torch import load as safetensors_load, save as safetensors_save

from config import load_config

warnings.filterwarnings("ignore")

# Global model references
vae = None
device = None
model_path = None


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="VAE Microservice for Z-Image-Turbo")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to VAE model directory (default: from config)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device: cuda, cuda:0, cuda:1, etc. (default: from config)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Port to run the service on (default: from config)",
    )
    return parser.parse_args()


def load_vae(model_path: str, device: str) -> AutoencoderKL:
    """
    Load VAE model.

    Args:
        model_path: Path to the VAE model directory
        device: Device to load the model on

    Returns:
        AutoencoderKL model
    """
    print(f"Loading VAE from {model_path} to {device}...")

    vae_model = AutoencoderKL.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
    ).to(device)

    vae_model.eval()
    print(f"VAE loaded: scale_factor={vae_model.config.scaling_factor}")
    return vae_model


# =============================================================================
# Encoding / Decoding
# =============================================================================


def encode_image(image: Image.Image) -> torch.Tensor:
    """
    Encode an image to latent space.

    Args:
        image: PIL Image to encode

    Returns:
        Latent tensor of shape [1, C, H//8, W//8]
    """
    global vae, device

    # Convert to RGB if needed
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Convert to tensor and normalize to [-1, 1]
    img_array = np.array(image).astype(np.float32) / 255.0
    img_array = (img_array * 2.0) - 1.0
    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)
    img_tensor = img_tensor.to(device=device, dtype=torch.bfloat16)

    with torch.no_grad():
        latent_dist = vae.encode(img_tensor).latent_dist
        latents = latent_dist.sample()
        latents = latents * vae.config.scaling_factor

    return latents


def decode_latents(latents: torch.Tensor) -> Image.Image:
    """
    Decode latents to an image.

    Args:
        latents: Latent tensor of shape [1, C, H, W]

    Returns:
        PIL Image
    """
    global vae, device

    latents = latents.to(device=device, dtype=torch.bfloat16)

    with torch.no_grad():
        # Unscale latents
        latents = latents / vae.config.scaling_factor
        image = vae.decode(latents, return_dict=False)[0]

    # Convert to PIL
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).float().numpy()
    image = (image[0] * 255).round().astype(np.uint8)
    return Image.fromarray(image)


# =============================================================================
# Serialization
# =============================================================================


def serialize_latents(latents: torch.Tensor) -> dict:
    """
    Serialize latents to JSON-safe format.

    Args:
        latents: Latent tensor

    Returns:
        dict with 'data' (base64), 'shape', 'dtype'
    """
    tensors = {"latents": latents.cpu()}
    raw_bytes = safetensors_save(tensors)

    return {
        "data": base64.b64encode(raw_bytes).decode("ascii"),
        "shape": list(latents.shape),
        "dtype": str(latents.dtype),
    }


def deserialize_latents(data: dict) -> torch.Tensor:
    """
    Deserialize latents from JSON format.

    Args:
        data: dict with 'data' (base64)

    Returns:
        Latent tensor
    """
    raw_bytes = base64.b64decode(data["data"])
    tensors = safetensors_load(raw_bytes)
    return tensors["latents"]


def image_to_base64(image: Image.Image) -> str:
    """Convert PIL Image to base64 string."""
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def base64_to_image(data: str) -> Image.Image:
    """Convert base64 string to PIL Image."""
    buffer = io.BytesIO(base64.b64decode(data))
    return Image.open(buffer)


# =============================================================================
# API Endpoints
# =============================================================================


def encode_api(image_b64: str) -> str:
    """
    API endpoint for encoding images to latents.

    Input: Base64 encoded image
    Output: JSON string with serialized latents
    """
    try:
        image = base64_to_image(image_b64)
        latents = encode_image(image)
        result = serialize_latents(latents)
        return json.dumps(result)
    except Exception as e:
        return json.dumps({"error": str(e)})


def decode_api(latents_json: str) -> str:
    """
    API endpoint for decoding latents to images.

    Input: JSON string with serialized latents
    Output: Base64 encoded image
    """
    try:
        data = json.loads(latents_json)
        latents = deserialize_latents(data)
        image = decode_latents(latents)
        result = {
            "image": image_to_base64(image),
            "size": list(image.size),
        }
        return json.dumps(result)
    except Exception as e:
        return json.dumps({"error": str(e)})


# =============================================================================
# Gradio UI
# =============================================================================


def create_ui() -> gr.Blocks:
    """Create the Gradio interface."""
    global model_path, device

    with gr.Blocks(title="Z-Image VAE Service") as demo:
        gr.Markdown("# Z-Image VAE Service")
        gr.Markdown(f"**Model:** `{model_path}` | **Device:** `{device}`")

        with gr.Tab("Encode"):
            gr.Markdown("## Encode Image to Latents")
            gr.Markdown("Upload an image to encode it to latent space.")

            with gr.Row():
                with gr.Column():
                    encode_input = gr.Image(
                        label="Input Image",
                        type="pil",
                    )
                    encode_btn = gr.Button("Encode", variant="primary")

                with gr.Column():
                    encode_output = gr.Textbox(
                        label="Latents (JSON, truncated)",
                        lines=10,
                    )
                    encode_shape = gr.Textbox(
                        label="Shape",
                        interactive=False,
                    )

            def encode_ui(image):
                if image is None:
                    return "Upload an image first", ""
                try:
                    latents = encode_image(image)
                    result = serialize_latents(latents)
                    # Truncate data for display
                    display = {
                        "data": result["data"][:100] + "...",
                        "shape": result["shape"],
                        "dtype": result["dtype"],
                    }
                    return json.dumps(display, indent=2), f"{result['shape']}"
                except Exception as e:
                    return f"Error: {e}", ""

            encode_btn.click(
                encode_ui,
                inputs=[encode_input],
                outputs=[encode_output, encode_shape],
            )

        with gr.Tab("Decode"):
            gr.Markdown("## Decode Latents to Image")
            gr.Markdown("Paste latents JSON to decode to an image.")

            with gr.Row():
                with gr.Column():
                    decode_input = gr.Textbox(
                        label="Latents (JSON)",
                        placeholder='{"data": "...", "shape": [...], "dtype": "..."}',
                        lines=10,
                    )
                    decode_btn = gr.Button("Decode", variant="primary")

                with gr.Column():
                    decode_output = gr.Image(
                        label="Output Image",
                        type="pil",
                    )

            def decode_ui(latents_json):
                if not latents_json.strip():
                    return None
                try:
                    data = json.loads(latents_json)
                    latents = deserialize_latents(data)
                    image = decode_latents(latents)
                    return image
                except Exception as e:
                    print(f"Decode error: {e}")
                    return None

            decode_btn.click(
                decode_ui,
                inputs=[decode_input],
                outputs=[decode_output],
            )

        with gr.Tab("Round Trip"):
            gr.Markdown("## Encode → Decode Round Trip")
            gr.Markdown("Test the quality of VAE reconstruction.")

            with gr.Row():
                with gr.Column():
                    rt_input = gr.Image(
                        label="Original Image",
                        type="pil",
                    )
                    rt_btn = gr.Button("Round Trip", variant="primary")

                with gr.Column():
                    rt_output = gr.Image(
                        label="Reconstructed Image",
                        type="pil",
                    )
                    rt_info = gr.Textbox(
                        label="Info",
                        interactive=False,
                    )

            def round_trip(image):
                if image is None:
                    return None, "Upload an image first"
                try:
                    latents = encode_image(image)
                    reconstructed = decode_latents(latents)
                    info = f"Original: {image.size}, Latent shape: {list(latents.shape)}"
                    return reconstructed, info
                except Exception as e:
                    return None, f"Error: {e}"

            rt_btn.click(
                round_trip,
                inputs=[rt_input],
                outputs=[rt_output, rt_info],
            )

        with gr.Tab("API Docs"):
            gr.Markdown(
                """
## VAE API Endpoints

### Encode Image

**Endpoint:** `POST /gradio_api/api/encode_image`

**Request:**
```json
{
    "data": ["<base64 encoded image>"]
}
```

**Response:**
```json
{
    "data": ["<JSON with data, shape, dtype>"]
}
```

### Decode Latents

**Endpoint:** `POST /gradio_api/api/decode_latents`

**Request:**
```json
{
    "data": ["<JSON with serialized latents>"]
}
```

**Response:**
```json
{
    "data": ["<JSON with base64 image and size>"]
}
```
"""
            )

            # Hidden API endpoints
            api_encode_input = gr.Textbox(visible=False)
            api_encode_output = gr.Textbox(visible=False)
            api_encode_input.change(
                encode_api,
                inputs=[api_encode_input],
                outputs=[api_encode_output],
                api_name="encode_image",
            )

            api_decode_input = gr.Textbox(visible=False)
            api_decode_output = gr.Textbox(visible=False)
            api_decode_input.change(
                decode_api,
                inputs=[api_decode_input],
                outputs=[api_decode_output],
                api_name="decode_latents",
            )

    return demo


# =============================================================================
# Main Entry Point
# =============================================================================


def main():
    global vae, device, model_path

    args = parse_args()
    config = load_config()

    # Determine device
    device = args.device or config.vae_device

    # Determine model path
    model_path = args.model or config.vae_path
    if not os.path.isabs(model_path) and os.path.exists(model_path):
        model_path = os.path.abspath(model_path)

    # Determine port
    port = args.port or config.vae_port

    # Offline mode
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    # Load model
    vae = load_vae(model_path, device)

    # Create and launch UI
    demo = create_ui()
    demo.queue(api_open=True)
    print(f"\nStarting VAE service on port {port}...")
    print(f"API endpoints:")
    print(f"  Encode: http://localhost:{port}/gradio_api/api/encode_image")
    print(f"  Decode: http://localhost:{port}/gradio_api/api/decode_latents")
    print(f"Web interface: http://localhost:{port}")
    demo.launch(
        server_name=config.server_name,
        server_port=port,
        show_error=True,
        ssr_mode=False,
    )


if __name__ == "__main__":
    main()
