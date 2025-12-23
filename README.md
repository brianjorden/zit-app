# zit-app

An opinionated bespoke inference approach to Z-Image-Turbo.

## Architecture

This application uses a distributed architecture to work within GPU memory constraints:

```
┌─────────────────────────────────────────────────────────────────┐
│  GPU 0: app.py (Main Application)                               │
│  - Transformer (ZImageTransformer2DModel)                       │
│  - VAE (optional, can use remote)                               │
│  - LoRA management                                              │
└─────────────────────────────────────────────────────────────────┘
                              │ HTTP API
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  GPU 1 (or CPU): encoder.py (Text Encoder Service)              │
│  - Qwen3-based text encoder                                     │
│  - Embedding API                                                │
│  - Chat interface                                               │
│  - Prompt enhancement                                           │
└─────────────────────────────────────────────────────────────────┘
                              │ HTTP API (optional)
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  GPU 2 (optional): vae.py (VAE Microservice)                    │
│  - Encode images to latents                                     │
│  - Decode latents to images                                     │
└─────────────────────────────────────────────────────────────────┘
```

## Quick Start

```bash
# Install dependencies
./install.sh

# Start encoder service (required, port 7888)
./start.sh encoder 1          # On cuda:1

# Start main app (in another terminal)
./start.sh app 0 --encoder-api
```

## Usage

### Launcher Commands

```bash
./start.sh app [gpu] [flags]            # Main app on specified GPU
./start.sh encoder [device]             # Encoder service on port 7888
./start.sh vae [gpu]                    # VAE microservice on port 7999
./start.sh minimal [app_gpu] [enc_dev]  # App + encoder (VAE in app)
./start.sh full [app] [enc] [vae]       # All three services
```

### App Flags

| Flag | Description |
|------|-------------|
| `--encoder-api` | Use remote encoder at localhost:7888 |
| `--vae-api` | Use remote VAE at localhost:7999 |

### Examples

```bash
# Services (fixed ports)
./start.sh encoder 4           # Encoder on cuda:4 (port 7888)
./start.sh encoder cpu         # Encoder on CPU
./start.sh vae 4               # VAE on cuda:4 (port 7999)

# App instances
./start.sh app 0 --encoder-api              # Use remote encoder only
./start.sh app 0 --encoder-api --vae-api    # Use both remote services

# Multi-GPU setup (encoder+VAE on cuda:4, apps on cuda:0-3)
./start.sh encoder 4
./start.sh vae 4
./start.sh app 0 --encoder-api --vae-api
./start.sh app 1 --encoder-api --vae-api
./start.sh app 2 --encoder-api --vae-api
./start.sh app 3 --encoder-api --vae-api

# Legacy modes
./start.sh minimal 0 1         # App on cuda:0, encoder on cuda:1
./start.sh full 0 1 2          # Full distribution across 3 GPUs
```

## Features

- **LoRA Support**: Dynamic loading with alpha scaling
- **Prompt Weighting**: `(word:weight)` syntax for emphasis/de-emphasis
- **Token Blending**: `[a|b|c]` syntax to blend multiple concepts
- **Fullscreen Gallery**: Click images to view fullscreen with arrow navigation
- **Performance Timing**: Encoding, diffusion, and VAE metrics
- **Extended Metadata**: Full reproduction info in PNG
- **Keyboard Shortcuts**: Ctrl+Enter to generate, Arrow keys in fullscreen
- **Settings Recall**: Restore previous generation settings
- **Date Organization**: Output sorted by date

## Prompt Weighting

Adjust the influence of specific words or phrases using ComfyUI/Automatic1111-style syntax.

### Syntax

```
(word:weight)           # Apply weight to a single word
(multiple words:weight) # Apply weight to a phrase
((nested:1.5):2)        # Nested weights multiply (1.5 * 2 = 3.0)
```

### Weight Values

| Weight | Effect |
|--------|--------|
| `> 1.0` | Emphasize (stronger influence) |
| `1.0` | Normal (default) |
| `< 1.0` | De-emphasize (weaker influence) |
| `0` | Remove entirely (word excluded from prompt) |
| `< 0` | Negative emphasis (experimental, may produce unexpected results) |

### Examples

```
A (beautiful:1.5) sunset                   # Emphasize "beautiful"
A cat with (blue eyes:0.5)                 # Subtle blue eyes
A (rainy:0) day in Paris                   # Remove "rainy" entirely
A forest with (fog:-0.5)                   # Reduce fog effect
A ((very:1.2) tall:1.5) building           # Nested: "very"=1.2, "tall"=1.8
```

### How It Works

1. **Parsing**: The prompt is parsed to extract weighted segments
2. **Tokenization**: Clean text (syntax removed) is tokenized by the encoder
3. **Weight Mapping**: Per-character weights are mapped to tokens
4. **Post-RMSNorm Application**: Weights are applied **after** the transformer's
   RMSNorm+Linear layer via a forward hook, bypassing normalization that would
   otherwise eliminate magnitude differences

### Technical Details

The Z-Image transformer uses `cap_embedder = nn.Sequential(RMSNorm(...), nn.Linear(...))`
to process text embeddings. RMSNorm normalizes by `x / sqrt(mean(x²) + eps)`, which
eliminates all magnitude differences if weights are applied before it.

Our implementation registers a forward hook on `cap_embedder` that applies weights
to the output **after** normalization, ensuring the scaling actually affects generation.

### Multi-Token Phrases

When weighting a phrase like `(bright sunset:1.5)`:
- All tokens in the phrase receive the weight
- The weight is determined by the characters each token covers
- For mixed weights, `min()` is used for de-emphasis (`< 1.0`), `max()` for emphasis

### Limitations

- Extreme weights (e.g., `99999`) will produce extreme/broken results
- Negative weights are experimental and may not behave intuitively

## Token Blending

Blend multiple concepts into a single embedding position using `[a|b|c]` syntax.

### Syntax

```
[cat|dog]                    # Blend cat and dog equally
[red|blue|green]             # Blend three colors
([fire|ice]:1.5) dragon      # Blend with outer weight applied
[(cat:0.8)|(dog:1.2)]        # Individual concept weights
```

### How It Works

1. Each concept in the blend is encoded separately
2. Multi-token concepts are average-pooled to a single embedding
3. Individual concept weights are applied (if specified)
4. All concept embeddings are averaged together
5. The outer weight (if any) is applied to the final blend
6. The blended embedding replaces the placeholder tokens

### Use Cases

- Combining visual styles: `a [watercolor|oil painting] landscape`
- Hybrid creatures: `a [cat|fox|wolf] in the forest`
- Color mixing: `a [red|blue] car` (produces purple-ish)

## Files

| File | Purpose |
|------|---------|
| `app.py` | Main image generation application |
| `encoder.py` | Text encoder service |
| `vae.py` | VAE microservice |
| `lora.py` | LoRA management |
| `prompt.py` | Token weighting/blending parser |
| `config.py` | Configuration and resolutions |
| `start.sh` | Launcher script |

## Resolution Categories

- 1024 (1024x1024 to 1344x576)
- 1280 (1280x1280 to 1680x720)
- 1536 (1536x1536 to 2016x864)
- 1792 (1792x1792 to 2352x1008)
- 2048 (2048x2048 to 2688x1152)

## Configuration

### Service Ports

| Service | Port |
|---------|------|
| Encoder | 7888 |
| VAE | 7999 |
| App | 7860 (auto-increments if busy) |

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ENCODER_URL` | - | Set automatically by `--encoder-api` flag |
| `VAE_URL` | - | Set automatically by `--vae-api` flag |
| `TRANSFORMER_PATH` | see config.py | Path to transformer |
| `VAE_PATH` | see config.py | Path to VAE |
| `LORA_DIR` | see config.py | Path to LoRA files |

## License

MIT
