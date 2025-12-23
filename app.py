#!/usr/bin/env python3
"""
Z-Image-Turbo Main Application.

A Gradio-based web application for AI image generation using the
Z-Image diffusion transformer model with LoRA support.

Features:
- Distributed architecture (encoder on separate device)
- LoRA support with dynamic loading
- Performance timing and extended metadata
- Gallery with fullscreen view and navigation
- Keyboard shortcuts and settings recall
"""

import argparse
import base64
import io
import json
import os
import random
import sys
import time
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import gradio as gr
import httpx
import torch
from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler
from diffusers.models.transformers.transformer_z_image import ZImageTransformer2DModel
from diffusers.pipelines.z_image import ZImagePipeline
from PIL import Image
from PIL.PngImagePlugin import PngInfo
from safetensors.torch import load as safetensors_load

from config import (
    load_config,
    RESOLUTION_CHOICES,
    ALL_RESOLUTIONS,
    get_resolution,
    DEFAULT_STEPS,
    DEFAULT_SHIFT,
    DEFAULT_GUIDANCE_SCALE,
    DEFAULT_MAX_SEQUENCE_LENGTH,
    DEFAULT_ENCODER_SEED,
)
from lora import LoRAManager, LoRAConfig, configs_from_ui_state, format_loras_for_metadata

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Global debug mode flag
debug_mode = False


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class TimingMetrics:
    """Timing metrics for generation stages."""

    encode_ms: int = 0
    diffusion_ms: int = 0
    vae_ms: int = 0

    @property
    def total_ms(self) -> int:
        return self.encode_ms + self.diffusion_ms + self.vae_ms


@dataclass
class ExtendedMetadata:
    """Extended metadata for image reproduction."""

    prompt: str
    resolution: str
    seed: int
    encoder_seed: int
    steps: int
    shift: float
    loras: str
    text_encoder: str
    diffusion_model: str
    scheduler: str
    vae: str
    timing_encode_ms: int
    timing_diffusion_ms: int
    timing_vae_ms: int
    timing_total_ms: int
    timestamp: str


# =============================================================================
# Global State
# =============================================================================


config = None
pipe = None
lora_manager = None
last_generation_settings = None


# =============================================================================
# Remote Encoder Client
# =============================================================================


def deserialize_embeddings(
    data: dict,
) -> tuple[list[torch.Tensor], list[torch.Tensor | None]]:
    """Deserialize embeddings and weights from remote encoder response."""
    raw_bytes = base64.b64decode(data["data"])
    tensors = safetensors_load(raw_bytes)

    embeddings = []
    weights = []
    has_weights = data.get("has_weights", False)

    for i in range(data["count"]):
        embeddings.append(tensors[f"emb_{i}"])
        if has_weights and f"weights_{i}" in tensors:
            weights.append(tensors[f"weights_{i}"])
        else:
            weights.append(None)

    return embeddings, weights


def encode_prompt_remote(
    prompt: str,
    max_sequence_length: int = DEFAULT_MAX_SEQUENCE_LENGTH,
    seed: int = DEFAULT_ENCODER_SEED,
) -> tuple[list[torch.Tensor], list[torch.Tensor | None]]:
    """
    Encode prompts using remote encoder service.

    Args:
        prompt: The prompt to encode
        max_sequence_length: Maximum sequence length
        seed: Encoder seed for reproducibility

    Returns:
        Tuple of:
        - List of embedding tensors
        - List of weight tensors (or None for each)
    """
    global config

    if not config.encoder_url:
        raise RuntimeError("ENCODER_URL not configured")

    prompts = [prompt] if isinstance(prompt, str) else prompt

    request_data = {
        "prompts": prompts,
        "max_sequence_length": max_sequence_length,
        "seed": seed,
        "apply_weighting": True,
    }

    # Gradio 6.x async API format
    # Step 1: Initiate the call
    call_response = httpx.post(
        f"{config.encoder_url}/gradio_api/call/encode_batch",
        json={"data": [json.dumps(request_data), max_sequence_length]},
        timeout=30.0,
    )
    call_response.raise_for_status()
    event_id = call_response.json()["event_id"]

    # Step 2: Poll for result (SSE endpoint)
    result_response = httpx.get(
        f"{config.encoder_url}/gradio_api/call/encode_batch/{event_id}",
        timeout=120.0,
    )
    result_response.raise_for_status()

    # Parse SSE response - look for "event: complete" followed by "data: ..."
    lines = result_response.text.strip().split("\n")
    result_json = None
    for i, line in enumerate(lines):
        if line.startswith("data: "):
            data_content = line[6:]  # Remove "data: " prefix
            try:
                parsed = json.loads(data_content)
                if isinstance(parsed, list) and len(parsed) > 0:
                    result_json = parsed[0]
                    break
            except json.JSONDecodeError:
                continue

    if result_json is None:
        raise RuntimeError(f"Failed to parse encoder response: {result_response.text[:500]}")

    result = json.loads(result_json)

    if "error" in result:
        raise RuntimeError(f"Encoder error: {result['error']}")

    return deserialize_embeddings(result)


# =============================================================================
# Remote VAE Client (Optional)
# =============================================================================


def decode_latents_remote(latents: torch.Tensor) -> Image.Image:
    """
    Decode latents using remote VAE service.

    Args:
        latents: Latent tensor

    Returns:
        PIL Image
    """
    global config

    if not config.vae_url:
        raise RuntimeError("VAE_URL not configured")

    from safetensors.torch import save as safetensors_save

    # Serialize latents
    raw_bytes = safetensors_save({"latents": latents.cpu()})
    data = {
        "data": base64.b64encode(raw_bytes).decode("ascii"),
        "shape": list(latents.shape),
        "dtype": str(latents.dtype),
    }

    response = httpx.post(
        f"{config.vae_url}/gradio_api/api/decode_latents",
        json={"data": [json.dumps(data)]},
        timeout=120.0,
    )
    response.raise_for_status()

    result_json = response.json()["data"][0]
    result = json.loads(result_json)

    if "error" in result:
        raise RuntimeError(f"VAE error: {result['error']}")

    # Decode image
    img_bytes = base64.b64decode(result["image"])
    return Image.open(io.BytesIO(img_bytes))


# =============================================================================
# Model Loading
# =============================================================================


def load_models():
    """
    Load VAE, transformer, and scheduler.

    Text encoder is loaded separately in encoder.py service.
    """
    global config, pipe

    print(f"Loading models...")
    print(f"  Transformer: {config.transformer_path}")
    print(f"  VAE: {config.vae_path}")
    print(f"  Device: {config.transformer_device}")

    # Load VAE (unless using remote VAE)
    vae = None
    if not config.vae_url:
        vae = AutoencoderKL.from_pretrained(
            config.vae_path,
            torch_dtype=torch.bfloat16,
        ).to(config.transformer_device)
        vae.eval()
        print(f"  VAE loaded locally")
    else:
        print(f"  VAE: using remote service at {config.vae_url}")

    # Load transformer
    transformer = ZImageTransformer2DModel.from_pretrained(
        config.transformer_path
    ).to(torch.bfloat16).to(config.transformer_device)

    transformer.set_attention_backend(config.attention_backend)
    print(f"  Transformer loaded, attention: {config.attention_backend}")

    # Create pipeline (without text encoder - we use remote)
    pipe = ZImagePipeline(
        scheduler=None,
        vae=vae,
        text_encoder=None,
        tokenizer=None,
        transformer=transformer,
    )

    # Override execution device
    pipe._force_execution_device = torch.device(config.transformer_device)
    original_execution_device = type(pipe)._execution_device.fget

    def patched_execution_device(self):
        if hasattr(self, "_force_execution_device"):
            return self._force_execution_device
        return original_execution_device(self)

    type(pipe)._execution_device = property(patched_execution_device)

    pipe.safety_feature_extractor = None
    pipe.safety_checker = None

    print("Models loaded successfully")


# =============================================================================
# Image Generation
# =============================================================================


def generate_image(
    prompt: str,
    width: int,
    height: int,
    seed: int,
    encoder_seed: int,
    steps: int,
    shift: float,
    lora_configs: list[LoRAConfig],
    progress_callback=None,
) -> tuple[Image.Image, TimingMetrics]:
    """
    Generate an image.

    Args:
        prompt: Text prompt
        width: Image width
        height: Image height
        seed: Generation seed
        encoder_seed: Encoder seed
        steps: Number of inference steps
        shift: Time shift for scheduler
        lora_configs: LoRA configurations
        progress_callback: Optional progress callback

    Returns:
        Tuple of (image, timing_metrics)
    """
    global pipe, lora_manager, config

    timing = TimingMetrics()

    # Clear CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Apply LoRAs
    if lora_manager:
        lora_manager.apply_loras(pipe.transformer, lora_configs)

    # Encode prompt
    t0 = time.perf_counter()
    prompt_embeds, prompt_weights = encode_prompt_remote(
        prompt,
        max_sequence_length=DEFAULT_MAX_SEQUENCE_LENGTH,
        seed=encoder_seed,
    )
    timing.encode_ms = int((time.perf_counter() - t0) * 1000)

    # Debug: print embedding and weight stats
    import sys
    emb = prompt_embeds[0]
    weights = prompt_weights[0]

    if debug_mode:
        token_norms = [f"{emb[i].norm().item():.1f}" for i in range(min(8, len(emb)))]
        if weights is not None:
            non_default = int((weights != 1.0).sum().item())
            print(f"[DEBUG] Prompt: {prompt[:50]!r}, shape: {emb.shape}, norms: {token_norms}, weights_non_default: {non_default}", file=sys.stderr, flush=True)
        else:
            print(f"[DEBUG] Prompt: {prompt[:50]!r}, shape: {emb.shape}, norms: {token_norms}, no weights", file=sys.stderr, flush=True)

    # Move embeddings to device
    prompt_embeds = [emb.to(config.transformer_device) for emb in prompt_embeds]
    if weights is not None:
        weights = weights.to(config.transformer_device)

    # Set up scheduler
    scheduler = FlowMatchEulerDiscreteScheduler(
        num_train_timesteps=1000,
        shift=shift,
    )
    pipe.scheduler = scheduler

    # Wrap scheduler.step to ensure tensors are on correct device
    original_step = scheduler.step

    def wrapped_step(model_output, timestep, sample, *args, **kwargs):
        device = torch.device(config.transformer_device)
        if hasattr(scheduler, "sigmas") and scheduler.sigmas is not None:
            if scheduler.sigmas.device != device:
                scheduler.sigmas = scheduler.sigmas.to(device)
        if hasattr(scheduler, "timesteps") and scheduler.timesteps is not None:
            if scheduler.timesteps.device != device:
                scheduler.timesteps = scheduler.timesteps.to(device)
        return original_step(
            model_output.to(device),
            timestep,
            sample.to(device),
            *args,
            **kwargs,
        )

    scheduler.step = wrapped_step

    # Register forward hook on cap_embedder to apply weights AFTER RMSNorm+Linear
    # This bypasses the normalization that was eliminating magnitude differences
    hook_handle = None
    if weights is not None and (weights != 1.0).any():
        def cap_embedder_hook(module, input, output):
            """Apply token weights after RMSNorm+Linear."""
            # output shape: [batch, seq_len, hidden_dim] or [seq_len, hidden_dim]
            # weights shape: [num_prompt_tokens]
            # Note: output may be larger than weights if other tokens are concatenated
            nonlocal weights

            weight_len = weights.size(0)

            if output.dim() == 3:
                # Batch dimension: [B, seq, dim]
                seq_len = output.size(1)
                # Create weight tensor for full sequence, default to 1.0
                w = torch.ones(seq_len, device=output.device, dtype=torch.float32)
                # Apply our weights to the first weight_len tokens (the prompt)
                apply_len = min(weight_len, seq_len)
                w[:apply_len] = weights[:apply_len]
                w = w.view(1, -1, 1)  # [1, seq_len, 1]
                weighted = output * w.to(output.dtype)
            else:
                # No batch: [seq, dim]
                seq_len = output.size(0)
                w = torch.ones(seq_len, device=output.device, dtype=torch.float32)
                apply_len = min(weight_len, seq_len)
                w[:apply_len] = weights[:apply_len]
                w = w.view(-1, 1)  # [seq_len, 1]
                weighted = output * w.to(output.dtype)

            # Debug log
            if debug_mode:
                non_default = int((w.flatten() != 1.0).sum().item())
                if non_default > 0:
                    print(f"[DEBUG] HOOK: Applied {non_default} non-default weights (seq={seq_len}, weights={weight_len})", file=sys.stderr, flush=True)

            return weighted

        hook_handle = pipe.transformer.cap_embedder.register_forward_hook(cap_embedder_hook)
        if debug_mode:
            print(f"[DEBUG] HOOK: Registered cap_embedder hook for token weighting", file=sys.stderr, flush=True)

    # Generate
    generator = torch.Generator(device=config.transformer_device).manual_seed(seed)

    t0 = time.perf_counter()
    try:
        result = pipe(
            prompt=None,
            prompt_embeds=prompt_embeds,
            height=height,
            width=width,
            guidance_scale=DEFAULT_GUIDANCE_SCALE,
            num_inference_steps=steps + 1,  # Z-Image convention
            generator=generator,
            max_sequence_length=DEFAULT_MAX_SEQUENCE_LENGTH,
            output_type="latent" if config.vae_url else "pil",
        )
    finally:
        # Always remove the hook
        if hook_handle is not None:
            hook_handle.remove()
            if debug_mode:
                print(f"[DEBUG] HOOK: Removed cap_embedder hook", file=sys.stderr, flush=True)

    timing.diffusion_ms = int((time.perf_counter() - t0) * 1000)

    # Handle VAE decoding (already done by pipeline if local VAE)
    if config.vae_url:
        # Remote VAE - pipeline returned latents (keep batch dim)
        t0 = time.perf_counter()
        image = decode_latents_remote(result.images)
        timing.vae_ms = int((time.perf_counter() - t0) * 1000)
    else:
        # Local VAE - pipeline returned list of images
        image = result.images[0]
        # VAE timing is included in diffusion for local

    return image, timing


# =============================================================================
# Image Saving
# =============================================================================


def get_output_dir() -> Path:
    """Get today's output directory, creating if needed."""
    global config

    today = datetime.now().strftime("%Y-%m-%d")
    output_dir = config.output_dir / today
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def save_image(image: Image.Image, metadata: ExtendedMetadata) -> Path:
    """
    Save image with extended metadata.

    Args:
        image: PIL Image to save
        metadata: Extended metadata

    Returns:
        Path to saved image
    """
    output_dir = get_output_dir()

    # Generate filename
    timestamp = datetime.now().strftime("%H%M%S_%f")
    filename = f"image_{timestamp}.png"
    output_path = output_dir / filename

    # Create PNG metadata
    pnginfo = PngInfo()
    pnginfo.add_text("prompt", metadata.prompt)
    pnginfo.add_text("resolution", metadata.resolution)
    pnginfo.add_text("seed", str(metadata.seed))
    pnginfo.add_text("encoder_seed", str(metadata.encoder_seed))
    pnginfo.add_text("steps", str(metadata.steps))
    pnginfo.add_text("shift", str(metadata.shift))
    pnginfo.add_text("loras", metadata.loras)
    pnginfo.add_text("text_encoder", metadata.text_encoder)
    pnginfo.add_text("diffusion_model", metadata.diffusion_model)
    pnginfo.add_text("scheduler", metadata.scheduler)
    pnginfo.add_text("vae", metadata.vae)
    pnginfo.add_text("timing_encode_ms", str(metadata.timing_encode_ms))
    pnginfo.add_text("timing_diffusion_ms", str(metadata.timing_diffusion_ms))
    pnginfo.add_text("timing_vae_ms", str(metadata.timing_vae_ms))
    pnginfo.add_text("timing_total_ms", str(metadata.timing_total_ms))
    pnginfo.add_text("timestamp", metadata.timestamp)

    image.save(output_path, pnginfo=pnginfo)
    return output_path


# =============================================================================
# Gradio Generate Function
# =============================================================================


def generate(
    prompt: str,
    resolution: str,
    seed: int,
    encoder_seed: int,
    steps: int,
    shift: float,
    gallery_images: list,
    lora_state: list[dict],
    progress=gr.Progress(track_tqdm=True),
) -> tuple[list, str, str, dict]:
    """
    Main generation function for Gradio.

    Returns:
        Tuple of (gallery_images, status, settings_text, last_settings)
    """
    global pipe, config, last_generation_settings

    if pipe is None:
        raise gr.Error("Models not loaded. Check console for errors.")

    # Parse resolution
    width, height = get_resolution(resolution)

    # Handle seeds
    actual_seed = seed if seed >= 0 else random.randint(1, 1000000)
    actual_encoder_seed = encoder_seed if encoder_seed >= 0 else random.randint(1, 1000000)

    # Convert LoRA state to configs
    lora_configs = configs_from_ui_state(lora_state)

    try:
        # Generate
        image, timing = generate_image(
            prompt=prompt,
            width=width,
            height=height,
            seed=actual_seed,
            encoder_seed=actual_encoder_seed,
            steps=steps,
            shift=shift,
            lora_configs=lora_configs,
        )

        # Create metadata
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        metadata = ExtendedMetadata(
            prompt=prompt,
            resolution=resolution.split(" ")[0],
            seed=actual_seed,
            encoder_seed=actual_encoder_seed,
            steps=steps,
            shift=shift,
            loras=format_loras_for_metadata(lora_configs),
            text_encoder=Path(config.text_encoder_path).name,
            diffusion_model=Path(config.transformer_path).name,
            scheduler="FlowMatchEulerDiscrete",
            vae=Path(config.vae_path).name,
            timing_encode_ms=timing.encode_ms,
            timing_diffusion_ms=timing.diffusion_ms,
            timing_vae_ms=timing.vae_ms,
            timing_total_ms=timing.total_ms,
            timestamp=timestamp,
        )

        # Save image
        output_path = save_image(image, metadata)

        # Update gallery
        if gallery_images is None:
            gallery_images = []
        gallery_images = [str(output_path)] + gallery_images

        # Format settings for display
        settings_lines = [
            f"prompt: {prompt}",
            f"resolution: {metadata.resolution}",
            f"seed: {actual_seed}",
            f"encoder_seed: {actual_encoder_seed}",
            f"steps: {steps}",
            f"shift: {shift}",
            f"loras: {metadata.loras}",
            f"timing: encode={timing.encode_ms}ms, diffusion={timing.diffusion_ms}ms, vae={timing.vae_ms}ms, total={timing.total_ms}ms",
        ]
        settings_text = "\n".join(settings_lines)

        # Store for recall
        last_generation_settings = {
            "prompt": prompt,
            "resolution": resolution,
            "seed": actual_seed,
            "encoder_seed": actual_encoder_seed,
            "steps": steps,
            "shift": shift,
            "lora_state": lora_state,
        }

        status = f"Generated in {timing.total_ms}ms"
        return gallery_images, status, settings_text, last_generation_settings

    except Exception as e:
        raise gr.Error(f"Generation failed: {e}")


def recall_settings(last_settings: dict) -> tuple:
    """Recall settings from last generation."""
    if not last_settings:
        return gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update()

    return (
        last_settings.get("prompt", ""),
        last_settings.get("resolution", "1024x1024 ( 1:1 )"),
        last_settings.get("seed", -1),
        last_settings.get("encoder_seed", 42),
        last_settings.get("steps", DEFAULT_STEPS),
        last_settings.get("shift", DEFAULT_SHIFT),
        last_settings.get("lora_state", [{"name": "None", "scale": 1.0}]),
    )


# =============================================================================
# Gallery Helpers
# =============================================================================


def load_image_metadata(file_path) -> tuple:
    """Load image and extract metadata for display."""
    if file_path is None:
        return None, "No file selected", ""

    try:
        img = Image.open(file_path)
        settings = []
        if hasattr(img, "text") and img.text:
            for key in [
                "prompt",
                "resolution",
                "seed",
                "encoder_seed",
                "steps",
                "shift",
                "loras",
                "timing_encode_ms",
                "timing_diffusion_ms",
                "timing_vae_ms",
                "timing_total_ms",
                "timestamp",
            ]:
                if key in img.text:
                    settings.append(f"{key}: {img.text[key]}")

        settings_text = "\n".join(settings) if settings else "No metadata found"
        return file_path, "Image loaded", settings_text
    except Exception as e:
        return None, f"Error: {e}", ""


def on_gallery_select(evt: gr.SelectData) -> str:
    """Display settings for selected gallery image."""
    if evt.value is None:
        return "No image selected"

    img_path = None
    if isinstance(evt.value, dict):
        img_path = evt.value.get("image", {}).get("path") if isinstance(
            evt.value.get("image"), dict
        ) else evt.value.get("image")
    elif isinstance(evt.value, str):
        img_path = evt.value

    if not img_path:
        return "Could not get image path"

    try:
        img = Image.open(img_path)
        if not hasattr(img, "text") or not img.text:
            return "No metadata found"

        settings = []
        for key in [
            "prompt",
            "resolution",
            "seed",
            "encoder_seed",
            "steps",
            "shift",
            "loras",
            "timing_encode_ms",
            "timing_diffusion_ms",
            "timing_vae_ms",
            "timing_total_ms",
            "timestamp",
        ]:
            if key in img.text:
                settings.append(f"{key}: {img.text[key]}")
        return "\n".join(settings) if settings else "No metadata found"
    except Exception as e:
        return f"Error: {e}"


# =============================================================================
# Gradio UI
# =============================================================================


def create_ui() -> gr.Blocks:
    """Create the Gradio interface."""
    global lora_manager

    # Custom CSS
    css = """
    .fillable { max-width: 1400px !important; }
    #output-gallery .progress-bar, #output-gallery .wrap, #output-gallery .generating,
    #settings-display .progress-bar, #settings-display .wrap, #settings-display .generating {
        display: none !important;
    }
    #token-count {
        margin-top: -8px;
        text-align: right;
        padding-right: 4px;
    }
    """

    # Custom JS for keyboard shortcuts and title progress
    js_head = """
    <script>
    // Ctrl+Enter to generate
    document.addEventListener('keydown', function(e) {
        if (e.ctrlKey && e.key === 'Enter') {
            const btn = document.querySelector('button.primary');
            if (btn) btn.click();
        }
    });

    // Title bar progress (updated via custom events)
    window.updateTitleProgress = function(percent) {
        if (percent >= 0 && percent < 100) {
            document.title = '[' + percent + '%] Z-Image';
        } else {
            document.title = 'Z-Image';
        }
    };
    </script>
    """

    with gr.Blocks(title="Z-Image", css=css, head=js_head) as demo:
        # State for settings recall
        last_settings_state = gr.State({})

        with gr.Row():
            # Left column - Controls
            with gr.Column(scale=1):
                prompt_input = gr.Textbox(
                    label="Prompt",
                    lines=3,
                    placeholder="Enter your prompt here...",
                    elem_id="prompt-input",
                )

                with gr.Row():
                    res_cat = gr.Dropdown(
                        value=1024,
                        choices=[int(k) for k in RESOLUTION_CHOICES.keys()],
                        label="Resolution Category",
                    )
                    resolution = gr.Dropdown(
                        value=RESOLUTION_CHOICES["1024"][0],
                        choices=ALL_RESOLUTIONS,
                        label="Width x Height (Ratio)",
                    )

                with gr.Row():
                    seed = gr.Number(label="Seed (-1 = random)", value=-1, precision=0)
                    encoder_seed = gr.Number(
                        label="Encoder Seed (-1 = random)",
                        value=DEFAULT_ENCODER_SEED,
                        precision=0,
                    )

                with gr.Row():
                    steps = gr.Slider(
                        label="Steps",
                        minimum=1,
                        maximum=20,
                        value=DEFAULT_STEPS,
                        step=1,
                    )
                    shift = gr.Slider(
                        label="Time Shift",
                        minimum=1.0,
                        maximum=10.0,
                        value=DEFAULT_SHIFT,
                        step=0.1,
                    )

                # LoRA Section
                lora_state = gr.State([{"name": "None", "scale": 1.0}])

                with gr.Accordion("LoRAs", open=True):

                    @gr.render(inputs=[lora_state])
                    def render_loras(lora_configs):
                        available_loras = lora_manager.get_available_loras() if lora_manager else []
                        lora_choices = ["None"] + available_loras

                        for idx, cfg in enumerate(lora_configs):
                            with gr.Row():
                                dropdown = gr.Dropdown(
                                    choices=lora_choices,
                                    value=cfg.get("name", "None"),
                                    label=f"LoRA {idx + 1}" if idx == 0 else "",
                                    scale=3,
                                    interactive=True,
                                )
                                slider = gr.Slider(
                                    minimum=-10.0,
                                    maximum=10.0,
                                    value=cfg.get("scale", 1.0),
                                    step=0.05,
                                    label="Strength" if idx == 0 else "",
                                    scale=1,
                                    interactive=True,
                                )

                                def update_lora_config(name, scale, configs, index=idx):
                                    new_configs = [dict(c) for c in configs]
                                    new_configs[index] = {"name": name, "scale": scale}
                                    if index == len(new_configs) - 1 and name != "None":
                                        new_configs.append({"name": "None", "scale": 1.0})
                                    while (
                                        len(new_configs) > 1
                                        and new_configs[-1]["name"] == "None"
                                        and new_configs[-2]["name"] == "None"
                                    ):
                                        new_configs.pop()
                                    return new_configs

                                dropdown.change(
                                    fn=update_lora_config,
                                    inputs=[dropdown, slider, lora_state],
                                    outputs=[lora_state],
                                )
                                slider.change(
                                    fn=update_lora_config,
                                    inputs=[dropdown, slider, lora_state],
                                    outputs=[lora_state],
                                )

                    def refresh_loras(configs):
                        return configs

                    refresh_btn = gr.Button("Refresh LoRAs", size="sm")
                    refresh_btn.click(fn=refresh_loras, inputs=[lora_state], outputs=[lora_state])

                with gr.Row():
                    generate_btn = gr.Button("Generate", variant="primary")
                    stop_btn = gr.Button("Stop", variant="stop")

                with gr.Row():
                    recall_btn = gr.Button("Recall Last Settings", size="sm")
                    auto_run = gr.Checkbox(label="Auto-run", value=False)

            # Right column - Output
            with gr.Column(scale=1):
                output_gallery = gr.Gallery(
                    label="Generated Images",
                    columns=2,
                    rows=2,
                    height=600,
                    object_fit="contain",
                    format="png",
                    interactive=False,
                    elem_id="output-gallery",
                )
                progress_box = gr.Textbox(
                    label="Status",
                    value="Ready",
                    interactive=False,
                    lines=1,
                )
                settings_display = gr.Textbox(
                    label="Settings",
                    interactive=False,
                    lines=8,
                    elem_id="settings-display",
                )
                load_image_btn = gr.UploadButton("Load Image", file_types=["image"])

        # Event handlers
        def update_res_choices(res_cat):
            if str(res_cat) in RESOLUTION_CHOICES:
                choices = RESOLUTION_CHOICES[str(res_cat)]
            else:
                choices = RESOLUTION_CHOICES["1024"]
            return gr.update(value=choices[0], choices=choices)

        res_cat.change(update_res_choices, inputs=res_cat, outputs=resolution)

        # Generate
        generate_event = generate_btn.click(
            generate,
            inputs=[
                prompt_input,
                resolution,
                seed,
                encoder_seed,
                steps,
                shift,
                output_gallery,
                lora_state,
            ],
            outputs=[output_gallery, progress_box, settings_display, last_settings_state],
        )

        # Auto-run
        generate_event.then(
            fn=lambda auto: None,
            inputs=[auto_run],
            outputs=None,
            js="(auto) => { if (auto) { setTimeout(() => { document.querySelector('button.primary').click(); }, 100); } }",
        )

        # Stop
        stop_btn.click(fn=None, inputs=None, outputs=None, cancels=[generate_event])

        # Recall settings
        recall_btn.click(
            recall_settings,
            inputs=[last_settings_state],
            outputs=[prompt_input, resolution, seed, encoder_seed, steps, shift, lora_state],
        )

        # Load image
        load_image_btn.upload(
            load_image_metadata,
            inputs=[load_image_btn],
            outputs=[output_gallery, progress_box, settings_display],
        )

        # Gallery selection
        output_gallery.select(on_gallery_select, None, settings_display)

    return demo


# =============================================================================
# Main
# =============================================================================


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Z-Image-Turbo Main Application")
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Port to run on (default: from config)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug output (prompt info, timing details)",
    )
    return parser.parse_args()


def main():
    global config, pipe, lora_manager, debug_mode

    args = parse_args()
    debug_mode = args.debug
    config = load_config()

    # Set DEBUG env var for other modules (encoder.py uses this)
    if debug_mode:
        os.environ["DEBUG"] = "1"

    # Use explicit port if specified via CLI, otherwise let Gradio auto-find
    port = args.port  # None = auto-find available port

    # Validate encoder URL
    if not config.encoder_url:
        print("ERROR: ENCODER_URL not set. The encoder service must be running.")
        print("Start it with: ./start.sh encoder <device>")
        print("Then set: export ENCODER_URL=http://localhost:7888")
        return

    print(f"Using encoder at: {config.encoder_url}")

    # Initialize LoRA manager
    lora_manager = LoRAManager(config.lora_dir)
    available = lora_manager.get_available_loras()
    print(f"Found {len(available)} LoRAs in {config.lora_dir}")

    # Load models
    try:
        load_models()
    except Exception as e:
        print(f"ERROR loading models: {e}")
        import traceback

        traceback.print_exc()
        return

    # Create and launch UI
    demo = create_ui()
    if port:
        print(f"\nStarting Z-Image on port {port}...")
    else:
        print(f"\nStarting Z-Image (auto-selecting port)...")
    demo.launch(
        server_name=config.server_name,
        server_port=port,
        show_error=True,
    )


if __name__ == "__main__":
    main()
