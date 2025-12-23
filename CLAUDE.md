# CLAUDE.md - Z-Image-Turbo App Project Context

## Project Overview
**zit-app** - An opinionated bespoke inference approach to Z-Image-Turbo

## Model Information

### Z-Image-Turbo
- **Model**: 6B parameter text-to-image diffusion model from Alibaba Tongyi Lab
- **Architecture**: Single-Stream Diffusion Transformer (S3-DiT) - text, visual semantic tokens, and image VAE tokens concatenated at sequence level
- **Inference**: 8 NFEs (Number of Function Evaluations) / 9 num_inference_steps
- **guidance_scale**: Should be 0.0 for Turbo model (no CFG needed)
- **VRAM**: Fits in 16GB (official claim, actual performance varies by GPU)
- **Features**: Bilingual text rendering (English/Chinese), strong instruction adherence

### Model Status (December 2025)
- **Z-Image-Turbo**: Released and available
- **Z-Image-Base**: NOT YET RELEASED
- **Z-Image-Edit**: NOT YET RELEASED

### Model Card Location
```
/home/brian/local/Z-Image-Turbo/README.md
```

## Pipeline Files

### Main Pipeline Location
```
/home/brian/apps/zit-app/.venv/lib/python3.11/site-packages/diffusers/pipelines/z_image/
```

### Available Pipelines
| File | Class | Purpose |
|------|-------|---------|
| `pipeline_z_image.py` | `ZImagePipeline` | Text-to-image generation |
| `pipeline_z_image_img2img.py` | `ZImageImg2ImgPipeline` | Image-to-image with strength |
| `pipeline_z_image_controlnet.py` | `ZImageControlNetPipeline` | ControlNet support |
| `pipeline_z_image_controlnet_inpaint.py` | `ZImageControlNetInpaintPipeline` | ControlNet + inpainting |
| `pipeline_output.py` | `ZImagePipelineOutput` | Output dataclass |

### Modular Pipelines Location
```
/home/brian/apps/zit-app/.venv/lib/python3.11/site-packages/diffusers/modular_pipelines/z_image/
```

## Technical Details

### Pipeline Components
- **Scheduler**: `FlowMatchEulerDiscreteScheduler`
- **Text Encoder**: Uses chat template with `enable_thinking=True`
- **VAE**: `AutoencoderKL` with scale_factor=8, image_processor uses vae_scale_factor*2=16
- **Transformer**: `ZImageTransformer2DModel`

### Key Parameters
```python
pipe(
    prompt="...",
    height=1024,
    width=1024,
    num_inference_steps=9,  # Results in 8 DiT forwards
    guidance_scale=0.0,     # Must be 0 for Turbo
    generator=torch.Generator("cuda").manual_seed(42),
)
```

### Dimension Requirements
- Height and width must be divisible by 32 (vae_scale_factor * 2 * 2 = 32)

## Development Environment

### Python/Package Management
- **ALWAYS use `uv`** for package management
- Python version: 3.11.14

### Virtual Environment
```bash
# ALWAYS activate before running anything
source .venv/bin/activate
```

### Key Dependencies
- PyTorch 2.9.1+cu130
- CUDA 13.0
- diffusers (from git - dev version 0.36.0.dev0)
- transformers (from git - dev version 5.0.0.dev0)
- gradio 6.2.0
- flash-attn 2.8.3+cu130torch2.9

### Hardware
- 5x NVIDIA GeForce RTX 4060 Ti GPUs

## Package Documentation Strategy
When looking up how Gradio, transformers, diffusers, or other packages work:
- **Primary source**: Review the code in `.venv/lib/python3.11/site-packages/`
- This ensures accuracy for the exact versions installed

## Reminders for Claude
- When asked about pipeline specifics, **re-read the pipeline files** for accurate responses
- When asked about the model, **re-read the model card** at the location above
- Current date context: December 2025
- Always activate .venv before running tests/scripts
- Use `uv` for all package operations
