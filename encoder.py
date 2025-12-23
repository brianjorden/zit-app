#!/usr/bin/env python3
"""
Text Encoder Service for Z-Image-Turbo.

A standalone Gradio app that provides:
1. Embedding API endpoint for image generation
2. Full chat interface with streaming
3. Prompt enhancement using the loaded model
4. Token counting

Usage:
    python encoder.py --device cuda:1 --port 7888
    python encoder.py --device cpu --port 7888
"""

import argparse
import base64
import json
import os
import threading
import warnings
from pathlib import Path

import gradio as gr
import torch
from safetensors.torch import load as safetensors_load, save as safetensors_save
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

from config import load_config, DEFAULT_MAX_SEQUENCE_LENGTH, DEFAULT_ENCODER_SEED
from prompt import (
    parse_prompt,
    apply_weights_to_embeddings,
    has_special_syntax,
    blend_embeddings,
    BlendInfo,
)

warnings.filterwarnings("ignore")

# Global model references
text_encoder = None
tokenizer = None
device = None
model_path = None


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Text Encoder Service for Z-Image-Turbo")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path or HuggingFace model name for the encoder (default: from config)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device: cpu, cuda, cuda:0, cuda:1, etc. (default: from config)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Port to run the service on (default: from config)",
    )
    return parser.parse_args()


def get_device(device_arg: str | None, config_device: str) -> str:
    """Determine the device to use."""
    if device_arg is not None:
        return device_arg
    return config_device


def load_encoder(model_path: str, device: str) -> tuple:
    """
    Load text encoder and tokenizer.

    Args:
        model_path: Path to the model directory
        device: Device to load the model on

    Returns:
        Tuple of (encoder, tokenizer)
    """
    print(f"Loading encoder from {model_path} to {device}...")

    # Load model without device_map to avoid distributed training issues
    encoder = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
    )
    encoder = encoder.to(device).eval()

    tok = AutoTokenizer.from_pretrained(model_path)
    tok.padding_side = "left"

    print(
        f"Encoder loaded: {encoder.config.hidden_size} hidden dim, "
        f"{encoder.config.num_hidden_layers} layers"
    )
    return encoder, tok


# =============================================================================
# Embedding Encoding
# =============================================================================


def encode_prompts(
    prompts: list[str],
    max_sequence_length: int = DEFAULT_MAX_SEQUENCE_LENGTH,
    seed: int = DEFAULT_ENCODER_SEED,
    apply_weighting: bool = True,
) -> tuple[list[torch.Tensor], list[torch.Tensor | None]]:
    """
    Encode prompts to embeddings.

    Process:
    1. Parse and strip weight syntax from prompts
    2. Apply chat template with enable_thinking=True
    3. Tokenize with left padding to max_sequence_length
    4. Run through encoder with output_hidden_states=True
    5. Extract hidden_states[-2] (second-to-last layer)
    6. Mask to actual sequence lengths (remove padding)
    7. Compute per-token weights (returned separately, NOT applied to embeddings)

    Args:
        prompts: List of prompt strings
        max_sequence_length: Maximum sequence length for tokenization
        seed: Random seed for reproducibility
        apply_weighting: Whether to compute token weights from syntax

    Returns:
        Tuple of:
        - List of torch.Tensor embeddings, each shaped [actual_seq_len, hidden_dim] in bfloat16
        - List of torch.Tensor weights (or None), each shaped [actual_seq_len] in float32
          Weights are NOT applied to embeddings - caller must apply them after RMSNorm
    """
    global text_encoder, tokenizer, device

    if isinstance(prompts, str):
        prompts = [prompts]

    # Set seed for reproducibility
    if seed >= 0:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

    # Parse prompts and prepare clean versions + weight info + blend info
    from prompt import prepare_prompt_for_encoding
    prompt_info = []
    for prompt in prompts:
        if apply_weighting:
            clean_prompt, char_weights, blend_infos = prepare_prompt_for_encoding(prompt)
        else:
            clean_prompt = prompt
            char_weights = None
            blend_infos = []
        prompt_info.append((clean_prompt, char_weights, blend_infos))

    # Apply chat template to CLEAN prompts (without weight syntax)
    processed_prompts = []
    for clean_prompt, _, _ in prompt_info:
        messages = [{"role": "user", "content": clean_prompt}]
        processed = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        processed_prompts.append(processed)

    # Tokenize with left padding
    text_inputs = tokenizer(
        processed_prompts,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        return_tensors="pt",
    )

    text_input_ids = text_inputs.input_ids.to(device)
    prompt_masks = text_inputs.attention_mask.bool().to(device)

    # Run through encoder
    with torch.no_grad():
        outputs = text_encoder(
            input_ids=text_input_ids,
            attention_mask=prompt_masks,
            output_hidden_states=True,
        )
        prompt_embeds = outputs.hidden_states[-2]  # Second-to-last layer

    # Extract embeddings and compute weights for each prompt
    embeddings_list = []
    weights_list = []
    for i, (clean_prompt, char_weights, blend_infos) in enumerate(prompt_info):
        # Only keep non-padded tokens
        emb = prompt_embeds[i][prompt_masks[i]].cpu()

        # Process blend segments if any
        if blend_infos:
            emb = _apply_blends(
                emb, blend_infos, clean_prompt, processed_prompts[i]
            )

        embeddings_list.append(emb)

        # Compute token weights if we have char_weights (but don't apply them)
        if char_weights is not None:
            token_weights = _compute_token_weights(
                emb.size(0), char_weights, tokenizer, clean_prompt, processed_prompts[i]
            )
            weights_list.append(token_weights)
        else:
            weights_list.append(None)

    return embeddings_list, weights_list


def _encode_single_concept(text: str) -> torch.Tensor:
    """
    Encode a single concept text to embeddings.

    Used by blend_embeddings to encode each concept in a blend.

    Args:
        text: The concept text to encode

    Returns:
        Embedding tensor of shape [seq_len, hidden_dim]
    """
    global text_encoder, tokenizer, device

    # Apply chat template
    messages = [{"role": "user", "content": text}]
    processed = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )

    # Tokenize (no padding needed for single concept)
    inputs = tokenizer(
        processed,
        return_tensors="pt",
        truncation=True,
        max_length=DEFAULT_MAX_SEQUENCE_LENGTH,
    )

    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.bool().to(device)

    # Encode
    with torch.no_grad():
        outputs = text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        embeds = outputs.hidden_states[-2]

    # Return only non-padded tokens
    return embeds[0][attention_mask[0]].cpu()


def _apply_blends(
    embeddings: torch.Tensor,
    blend_infos: list[BlendInfo],
    clean_prompt: str,
    chat_processed: str,
) -> torch.Tensor:
    """
    Apply blend operations to embeddings.

    For each blend segment, encodes all concepts, averages them,
    and replaces the placeholder tokens in the embedding.

    Args:
        embeddings: Original embeddings [seq_len, hidden_dim]
        blend_infos: List of BlendInfo describing each blend
        clean_prompt: The clean prompt text (used for token mapping)
        chat_processed: The chat-template-processed prompt

    Returns:
        Modified embeddings with blends applied
    """
    global tokenizer

    # Make a copy to modify
    result = embeddings.clone()

    for blend_info in blend_infos:
        # Get the blended embedding
        blended = blend_embeddings(_encode_single_concept, blend_info.segment)

        # Find which tokens correspond to this blend's position
        # The blend placeholder is the first concept's text
        first_concept_text = blend_info.segment.blend[0].text

        # Find the token indices for this blend
        # We need to map character positions to token positions
        token_start, token_end = _find_token_range_for_chars(
            tokenizer, chat_processed, clean_prompt,
            blend_info.char_start, blend_info.char_end
        )

        if token_start is not None and token_end is not None:
            # The blended embedding is a single vector [1, hidden_dim] or [hidden_dim]
            if blended.dim() == 1:
                blended = blended.unsqueeze(0)

            # Replace tokens with blended embedding
            # If blend spans multiple tokens, we average-pool the blended result
            # into each token position (or just use the same embedding)
            num_tokens = token_end - token_start
            if num_tokens > 0:
                # Expand blended to fill all token positions
                expanded = blended.expand(num_tokens, -1)
                result[token_start:token_end] = expanded

            if os.environ.get("DEBUG"):
                print(f"[DEBUG] BLEND: Applied {len(blend_info.segment.blend)} concepts "
                      f"at tokens {token_start}:{token_end}", flush=True)

    return result


def _find_token_range_for_chars(
    tokenizer,
    chat_processed: str,
    clean_prompt: str,
    char_start: int,
    char_end: int,
) -> tuple[int | None, int | None]:
    """
    Find token indices corresponding to character positions.

    Args:
        tokenizer: The tokenizer
        chat_processed: The full chat-template-processed prompt
        clean_prompt: The clean prompt (without chat template)
        char_start: Start character position in clean_prompt
        char_end: End character position in clean_prompt

    Returns:
        Tuple of (token_start, token_end) indices, or (None, None) if not found
    """
    # Find where clean_prompt appears in chat_processed
    # The clean prompt is embedded in the chat template
    prompt_start = chat_processed.find(clean_prompt)
    if prompt_start < 0:
        # Fallback: try to find the substring
        target = clean_prompt[char_start:char_end]
        idx = chat_processed.find(target)
        if idx < 0:
            return None, None
        adjusted_start = idx
        adjusted_end = idx + len(target)
    else:
        adjusted_start = prompt_start + char_start
        adjusted_end = prompt_start + char_end

    # Tokenize to get offsets
    encoding = tokenizer(
        chat_processed,
        return_offsets_mapping=True,
        add_special_tokens=False,
    )

    offsets = encoding.offset_mapping
    token_start = None
    token_end = None

    for i, (start, end) in enumerate(offsets):
        if start <= adjusted_start < end:
            token_start = i
        if start < adjusted_end <= end:
            token_end = i + 1
            break

    # Fallback if we didn't find exact boundaries
    if token_start is None:
        for i, (start, end) in enumerate(offsets):
            if end > adjusted_start:
                token_start = i
                break

    if token_end is None and token_start is not None:
        for i, (start, end) in enumerate(offsets):
            if start >= adjusted_end:
                token_end = i
                break
        if token_end is None:
            token_end = len(offsets)

    return token_start, token_end


def _compute_token_weights(
    seq_len: int,
    char_weights: list[float],
    tokenizer,
    clean_prompt: str,
    chat_processed: str,
) -> torch.Tensor:
    """
    Compute per-token weights based on per-character weights.

    This finds where the prompt content is within the chat template tokens,
    then computes weights for those tokens.

    Args:
        seq_len: Number of tokens in the sequence
        char_weights: Per-character weight list for clean_prompt
        tokenizer: The tokenizer
        clean_prompt: The clean prompt (weight=0 segments already excluded)
        chat_processed: The full chat-templated string

    Returns:
        Weight tensor of shape [seq_len] in float32
    """
    # Create weight tensor, default to 1.0
    weights = torch.ones(seq_len, dtype=torch.float32)

    # Find where the clean prompt appears in the chat template
    prompt_start = chat_processed.find(clean_prompt)
    if prompt_start < 0:
        # Prompt not found in template, return all 1s
        return weights

    prompt_end = prompt_start + len(clean_prompt)

    # Tokenize the full chat string to get token-to-char mapping
    encoding = tokenizer(
        chat_processed,
        return_offsets_mapping=True,
        add_special_tokens=False,
    )
    offsets = encoding.get("offset_mapping", [])

    if not offsets:
        # No offset mapping available, return all 1s
        return weights

    # For each token, check if it overlaps with the prompt content
    for tok_idx, (start, end) in enumerate(offsets):
        if tok_idx >= len(weights):
            break
        # Check if this token overlaps with the prompt content
        if start < prompt_end and end > prompt_start:
            # Calculate overlap with prompt region
            overlap_start = max(start, prompt_start)
            overlap_end = min(end, prompt_end)

            # Get character positions within clean prompt
            char_start = overlap_start - prompt_start
            char_end = overlap_end - prompt_start

            # Collect weights for all characters this token covers
            token_char_weights = []
            for cp in range(char_start, char_end):
                if cp < len(char_weights):
                    token_char_weights.append(char_weights[cp])

            # Determine token weight from character weights
            # Use min if any weight < 1 (de-emphasis), else max (emphasis)
            if token_char_weights:
                min_w = min(token_char_weights)
                max_w = max(token_char_weights)
                # If any character wants de-emphasis, use min; otherwise use max
                weights[tok_idx] = min_w if min_w < 1.0 else max_w

    # Log non-default weights
    non_default = (weights != 1.0).sum().item()
    if non_default > 0:
        print(f"[WEIGHTING] Computed weights for {non_default} tokens: {weights[weights != 1.0].tolist()}")

    return weights


# =============================================================================
# Serialization (safetensors + base64)
# =============================================================================


def serialize_embeddings(
    embeddings: list[torch.Tensor],
    weights: list[torch.Tensor | None] | None = None,
) -> dict:
    """
    Serialize embeddings and optional weights to JSON-safe format using safetensors.

    Args:
        embeddings: List of torch.Tensor embeddings
        weights: Optional list of weight tensors (or None for each)

    Returns:
        dict with 'data' (base64 string), 'count', 'dtype', and 'has_weights'
    """
    tensors = {f"emb_{i}": emb for i, emb in enumerate(embeddings)}

    # Add weights if provided
    has_weights = False
    if weights:
        for i, w in enumerate(weights):
            if w is not None:
                tensors[f"weights_{i}"] = w
                has_weights = True

    raw_bytes = safetensors_save(tensors)

    return {
        "data": base64.b64encode(raw_bytes).decode("ascii"),
        "count": len(embeddings),
        "dtype": str(embeddings[0].dtype) if embeddings else "bfloat16",
        "has_weights": has_weights,
    }


def deserialize_embeddings(
    data: dict,
) -> tuple[list[torch.Tensor], list[torch.Tensor | None]]:
    """
    Deserialize embeddings and weights from JSON format.

    Args:
        data: dict with 'data' (base64 string), 'count', and optionally 'has_weights'

    Returns:
        Tuple of:
        - List of torch.Tensor embeddings
        - List of weight tensors (or None for each)
    """
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


# =============================================================================
# API Endpoint
# =============================================================================


def encode_api(prompts_json: str, max_seq_len: int = DEFAULT_MAX_SEQUENCE_LENGTH) -> str:
    """
    API endpoint for encoding prompts.

    Input: JSON string with {"prompts": [...], "max_sequence_length": 2048, "seed": 42}
    Output: JSON string with serialized embeddings and weights
    """
    try:
        request = json.loads(prompts_json)
        prompts = request.get("prompts", [])
        max_seq = request.get("max_sequence_length", int(max_seq_len))
        seed = request.get("seed", DEFAULT_ENCODER_SEED)
        apply_weighting = request.get("apply_weighting", True)

        embeddings, weights = encode_prompts(prompts, max_seq, seed, apply_weighting)
        result = serialize_embeddings(embeddings, weights)

        return json.dumps(result)
    except Exception as e:
        return json.dumps({"error": str(e)})


# =============================================================================
# Chat Interface
# =============================================================================

# Chat settings stored as state
chat_settings = {"temperature": 0.7, "max_tokens": 2048, "enable_thinking": True}


def chat_respond(message: str, history: list):
    """
    Streaming chat completion.

    Gradio 6.0 ChatInterface format:
    - message: str - the user's current message
    - history: list of OpenAI-style dicts

    Yields:
        Partial response strings for streaming
    """
    global text_encoder, tokenizer, device, chat_settings

    # Build messages list
    messages = []
    for msg in history:
        content = msg.get("content", "")
        if isinstance(content, list):
            content = "".join(
                item.get("text", "") if isinstance(item, dict) else str(item)
                for item in content
            )
        messages.append({"role": msg["role"], "content": content})
    messages.append({"role": "user", "content": message})

    # Apply chat template
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=chat_settings["enable_thinking"],
    )

    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    # Set up streamer
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    # Generation parameters
    generation_kwargs = dict(
        **model_inputs,
        max_new_tokens=chat_settings["max_tokens"],
        streamer=streamer,
    )

    if chat_settings["temperature"] > 0:
        generation_kwargs["temperature"] = chat_settings["temperature"]
        generation_kwargs["do_sample"] = True
    else:
        generation_kwargs["do_sample"] = False

    # Start generation in background thread
    thread = threading.Thread(target=text_encoder.generate, kwargs=generation_kwargs)
    thread.start()

    # Stream responses
    partial_response = ""
    for new_text in streamer:
        partial_response += new_text
        yield partial_response

    thread.join()


# =============================================================================
# Prompt Enhancement
# =============================================================================

# System prompt for prompt enhancement
ENHANCEMENT_SYSTEM_PROMPT = """You are an expert prompt engineer for AI image generation. Your task is to expand short, simple prompts into detailed, vivid descriptions that will produce high-quality images.

Guidelines:
1. Add specific visual details (lighting, colors, textures, materials)
2. Include artistic style references when appropriate
3. Describe the scene composition and perspective
4. Add atmospheric and mood elements
5. Keep the core subject and intent from the original prompt
6. Use natural, flowing language
7. Aim for 50-150 words

Respond with ONLY the enhanced prompt, no explanations or prefixes."""


def enhance_prompt(prompt: str, max_tokens: int = 512) -> str:
    """
    Enhance a short prompt using the loaded model.

    Args:
        prompt: Short prompt to enhance
        max_tokens: Maximum tokens for the response

    Returns:
        Enhanced prompt string
    """
    global text_encoder, tokenizer, device

    if not prompt.strip():
        return ""

    messages = [
        {"role": "system", "content": ENHANCEMENT_SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,  # Faster without thinking
    )

    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = text_encoder.generate(
            **model_inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )

    # Decode only the new tokens
    response = tokenizer.decode(
        outputs[0][model_inputs.input_ids.shape[1] :],
        skip_special_tokens=True,
    )

    return response.strip()


def enhance_prompt_streaming(prompt: str, max_tokens: int = 512):
    """
    Enhance a prompt with streaming output.

    Yields:
        Partial response strings
    """
    global text_encoder, tokenizer, device

    if not prompt.strip():
        yield ""
        return

    messages = [
        {"role": "system", "content": ENHANCEMENT_SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    model_inputs = tokenizer([text], return_tensors="pt").to(device)
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    generation_kwargs = dict(
        **model_inputs,
        max_new_tokens=max_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        streamer=streamer,
    )

    thread = threading.Thread(target=text_encoder.generate, kwargs=generation_kwargs)
    thread.start()

    partial = ""
    for new_text in streamer:
        partial += new_text
        yield partial

    thread.join()


# =============================================================================
# Token Counter
# =============================================================================


def count_tokens_api(text: str) -> dict:
    """Count tokens in text using the tokenizer."""
    global tokenizer
    if not text or not text.strip():
        return {"count": 0}
    try:
        tokens = tokenizer.encode(text)
        return {"count": len(tokens)}
    except Exception as e:
        return {"count": -1, "error": str(e)}


# =============================================================================
# Gradio UI
# =============================================================================


def update_chat_settings(temp: float, max_tok: int, thinking: bool) -> str:
    """Update global chat settings."""
    global chat_settings
    chat_settings["temperature"] = temp
    chat_settings["max_tokens"] = max_tok
    chat_settings["enable_thinking"] = thinking
    return f"Settings: temp={temp}, max_tokens={max_tok}, thinking={thinking}"


def create_ui() -> gr.Blocks:
    """Create the Gradio interface."""
    global model_path, device

    with gr.Blocks(title="Z-Image Encoder Service") as demo:
        gr.Markdown(f"# Z-Image Encoder Service")
        gr.Markdown(f"**Model:** `{model_path}` | **Device:** `{device}`")

        with gr.Tab("Chat"):
            with gr.Accordion("Settings", open=False):
                with gr.Row():
                    temperature = gr.Slider(
                        minimum=0.0,
                        maximum=2.0,
                        value=0.7,
                        step=0.1,
                        label="Temperature (0 = greedy)",
                    )
                    max_tokens = gr.Slider(
                        minimum=1,
                        maximum=8192,
                        value=2048,
                        step=1,
                        label="Max Tokens",
                    )
                    enable_thinking = gr.Checkbox(
                        value=True,
                        label="Enable Thinking Mode",
                    )
                settings_status = gr.Textbox(
                    value="Using defaults: temp=0.7, max_tokens=2048, thinking=True",
                    label="Status",
                    interactive=False,
                )

                temperature.change(
                    update_chat_settings,
                    inputs=[temperature, max_tokens, enable_thinking],
                    outputs=[settings_status],
                )
                max_tokens.change(
                    update_chat_settings,
                    inputs=[temperature, max_tokens, enable_thinking],
                    outputs=[settings_status],
                )
                enable_thinking.change(
                    update_chat_settings,
                    inputs=[temperature, max_tokens, enable_thinking],
                    outputs=[settings_status],
                )

            gr.ChatInterface(
                fn=chat_respond,
                examples=["Hello!", "What is the meaning of life?", "Write a haiku about coding."],
            )

        with gr.Tab("Prompt Enhancement"):
            gr.Markdown("## Expand Short Prompts")
            gr.Markdown(
                "Enter a brief prompt and get an expanded, detailed version "
                "optimized for image generation."
            )

            with gr.Row():
                with gr.Column():
                    input_prompt = gr.Textbox(
                        label="Short Prompt",
                        placeholder="a cat sitting on a couch",
                        lines=2,
                    )
                    enhance_btn = gr.Button("Enhance", variant="primary")

                with gr.Column():
                    output_prompt = gr.Textbox(
                        label="Enhanced Prompt",
                        lines=6,
                        interactive=True,
                    )
                    copy_btn = gr.Button("Copy to Clipboard", size="sm")

            enhance_btn.click(
                enhance_prompt_streaming,
                inputs=[input_prompt],
                outputs=[output_prompt],
            )

            # Copy to clipboard via JS
            copy_btn.click(
                fn=None,
                inputs=[output_prompt],
                outputs=None,
                js="(text) => { navigator.clipboard.writeText(text); }",
            )

            gr.Examples(
                examples=[
                    "a wizard casting a spell",
                    "sunset over mountains",
                    "cyberpunk city street",
                    "portrait of a woman",
                ],
                inputs=[input_prompt],
            )

        with gr.Tab("Embedding API"):
            gr.Markdown("## Embedding API Endpoint")
            gr.Markdown(
                """
This service exposes an embedding API for the Z-Image pipeline.

**Endpoint:** `POST /gradio_api/api/encode`

**Request format:**
```json
{
    "data": ["{\\\"prompts\\\": [\\\"your prompt\\\"], \\\"max_sequence_length\\\": 2048, \\\"seed\\\": 42}", 2048]
}
```

**Response format:**
```json
{
    "data": ["<JSON with base64-encoded safetensors embeddings>"]
}
```
"""
            )

            gr.Markdown("---")
            gr.Markdown("### Test Interface")

            test_prompt = gr.Textbox(
                label="Prompt",
                placeholder="Enter a prompt to test encoding...",
                lines=2,
            )
            with gr.Row():
                test_max_seq = gr.Slider(
                    minimum=128,
                    maximum=4096,
                    value=2048,
                    step=128,
                    label="Max Sequence Length",
                )
                test_seed = gr.Number(
                    value=42,
                    label="Seed (-1 for random)",
                    precision=0,
                )
            test_btn = gr.Button("Encode", variant="primary")
            test_result = gr.Textbox(label="Result (truncated)", lines=10)

            def test_encode(prompt: str, max_seq: int, seed: int) -> str:
                if not prompt.strip():
                    return "Enter a prompt first"
                try:
                    embeddings, weights = encode_prompts([prompt], int(max_seq), int(seed))
                    result = serialize_embeddings(embeddings, weights)
                    # Truncate data for display
                    result["data"] = result["data"][:200] + "..."
                    result["shape"] = list(embeddings[0].shape)
                    if weights[0] is not None:
                        result["weights_shape"] = list(weights[0].shape)
                        result["weights_non_default"] = int((weights[0] != 1.0).sum().item())
                    return json.dumps(result, indent=2)
                except Exception as e:
                    return f"Error: {e}"

            test_btn.click(
                test_encode,
                inputs=[test_prompt, test_max_seq, test_seed],
                outputs=[test_result],
                api_name="encode",
            )

        with gr.Tab("Token Counter"):
            gr.Markdown("## Token Counter")
            gr.Markdown("Count tokens in text using this model's tokenizer.")

            token_input = gr.Textbox(
                label="Text",
                placeholder="Enter text to count tokens...",
                lines=5,
            )
            token_result = gr.JSON(label="Token Count")

            token_input.change(
                count_tokens_api,
                inputs=[token_input],
                outputs=[token_result],
            )

        # Hidden API endpoint for batch encoding (used by app.py)
        with gr.Row(visible=False):
            api_input = gr.Textbox()
            api_max_seq = gr.Number(value=DEFAULT_MAX_SEQUENCE_LENGTH)
            api_output = gr.Textbox()
            api_btn = gr.Button()
            api_btn.click(
                encode_api,
                inputs=[api_input, api_max_seq],
                outputs=[api_output],
                api_name="encode_batch",
            )

    return demo


# =============================================================================
# Main Entry Point
# =============================================================================


def main():
    global text_encoder, tokenizer, device, model_path

    args = parse_args()
    config = load_config()

    # Determine device
    device = get_device(args.device, config.encoder_device)

    # Determine model path
    model_path = args.model or config.text_encoder_path
    if not os.path.isabs(model_path) and os.path.exists(model_path):
        model_path = os.path.abspath(model_path)

    # Determine port
    port = args.port or config.encoder_port

    # Offline mode
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    # Load model
    text_encoder, tokenizer = load_encoder(model_path, device)

    # Create and launch UI
    demo = create_ui()
    demo.queue(api_open=True)
    print(f"\nStarting encoder service on port {port}...")
    print(f"API endpoint: http://localhost:{port}/gradio_api/api/encode")
    print(f"Web interface: http://localhost:{port}")
    demo.launch(
        server_name=config.server_name,
        server_port=port,
        show_error=True,
        ssr_mode=False,
    )


if __name__ == "__main__":
    main()
