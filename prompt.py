"""
Token Weighting and Blending for Z-Image-Turbo.

Supports ComfyUI/Automatic1111 style prompt syntax:
- Weighting: (word:weight) or ((nested:1.5):2)
- Blending: [conceptA|conceptB|conceptC]
- Combined: ([red|blue]:0.5) car
"""

from dataclasses import dataclass, field
from typing import Callable

import torch


@dataclass
class PromptSegment:
    """A segment of parsed prompt with weight and optional blend."""

    text: str
    weight: float = 1.0
    blend: list["PromptSegment"] | None = None

    def is_blend(self) -> bool:
        """Check if this segment is a blend of multiple concepts."""
        return self.blend is not None and len(self.blend) > 0

    def __repr__(self) -> str:
        if self.is_blend():
            concepts = "|".join(s.text for s in self.blend)
            return f"Blend([{concepts}], weight={self.weight})"
        return f"Segment({self.text!r}, weight={self.weight})"


class PromptParser:
    """
    Parser for weighted and blended prompts.

    Syntax:
    - (text:weight) - apply weight to text
    - ((text:w1):w2) - nested weights multiply (w1 * w2)
    - [a|b|c] - blend concepts a, b, c equally
    - ([a|b]:weight) - blend then apply weight
    """

    def parse(self, prompt: str) -> list[PromptSegment]:
        """
        Parse a prompt string into segments with weights and blends.

        Args:
            prompt: The prompt string to parse

        Returns:
            List of PromptSegment objects
        """
        if not prompt:
            return []

        segments = []
        current_text = ""
        i = 0

        while i < len(prompt):
            char = prompt[i]

            if char == "(":
                # Save any accumulated text
                if current_text:
                    segments.append(PromptSegment(text=current_text))
                    current_text = ""

                # Find matching closing paren
                content, weight, end_pos = self._parse_weighted(prompt, i)
                if content is not None:
                    # Check if content is a blend
                    if content.startswith("[") and content.endswith("]"):
                        blend_segments = self._parse_blend_content(content[1:-1])
                        segments.append(PromptSegment(text="", weight=weight, blend=blend_segments))
                    else:
                        # Recursively parse content for nested weights
                        inner_segments = self.parse(content)
                        # Apply outer weight to all inner segments
                        for seg in inner_segments:
                            seg.weight *= weight
                        segments.extend(inner_segments)
                    i = end_pos
                else:
                    current_text += char
                    i += 1

            elif char == "[":
                # Save any accumulated text
                if current_text:
                    segments.append(PromptSegment(text=current_text))
                    current_text = ""

                # Find matching closing bracket
                content, end_pos = self._find_matching(prompt, i, "[", "]")
                if content is not None:
                    blend_segments = self._parse_blend_content(content)
                    segments.append(PromptSegment(text="", weight=1.0, blend=blend_segments))
                    i = end_pos
                else:
                    current_text += char
                    i += 1

            else:
                current_text += char
                i += 1

        # Don't forget trailing text
        if current_text:
            segments.append(PromptSegment(text=current_text))

        return segments

    def _parse_weighted(
        self, text: str, start: int
    ) -> tuple[str | None, float, int]:
        """
        Parse a weighted expression starting at position start.

        Returns (content, weight, end_position) or (None, 1.0, start) if invalid.
        """
        content, end_pos = self._find_matching(text, start, "(", ")")
        if content is None:
            return None, 1.0, start

        # Check for :weight suffix
        weight = 1.0
        colon_pos = content.rfind(":")
        if colon_pos > 0:
            try:
                weight = float(content[colon_pos + 1 :])
                content = content[:colon_pos]
            except ValueError:
                pass  # Not a valid weight, keep as-is

        return content, weight, end_pos

    def _find_matching(
        self, text: str, start: int, open_char: str, close_char: str
    ) -> tuple[str | None, int]:
        """
        Find content between matching open/close characters.

        Returns (content, end_position) or (None, start) if not found.
        """
        if start >= len(text) or text[start] != open_char:
            return None, start

        depth = 1
        i = start + 1

        while i < len(text) and depth > 0:
            if text[i] == open_char:
                depth += 1
            elif text[i] == close_char:
                depth -= 1
            i += 1

        if depth == 0:
            # Return content without the delimiters
            return text[start + 1 : i - 1], i
        return None, start

    def _parse_blend_content(self, content: str) -> list[PromptSegment]:
        """
        Parse the content of a blend expression.

        Args:
            content: String like "cat|dog|wolf" or "(black:-2)|thong|panties"

        Returns:
            List of PromptSegment for each concept
        """
        segments = []
        # Split by | but handle nested parens
        concepts = self._split_blend(content)

        for concept in concepts:
            concept = concept.strip()
            if not concept:
                continue

            # Check if concept has weight syntax
            if concept.startswith("(") and ":" in concept:
                inner = self.parse(concept)
                if inner:
                    segments.extend(inner)
            else:
                segments.append(PromptSegment(text=concept))

        return segments

    def _split_blend(self, content: str) -> list[str]:
        """
        Split blend content by | while respecting nested parens.
        """
        parts = []
        current = ""
        depth = 0

        for char in content:
            if char == "(":
                depth += 1
                current += char
            elif char == ")":
                depth -= 1
                current += char
            elif char == "|" and depth == 0:
                parts.append(current)
                current = ""
            else:
                current += char

        if current:
            parts.append(current)

        return parts


def parse_prompt(prompt: str) -> list[PromptSegment]:
    """
    Parse a prompt string into weighted/blended segments.

    Args:
        prompt: The prompt string

    Returns:
        List of PromptSegment objects
    """
    parser = PromptParser()
    return parser.parse(prompt)


def apply_weights_to_embeddings(
    embeddings: torch.Tensor,
    segments: list[PromptSegment],
    tokenizer,
    prompt: str,
) -> torch.Tensor:
    """
    Apply token weights to embeddings based on parsed segments.

    This maps each token back to its segment to determine weight,
    then multiplies the embedding by that weight.

    Args:
        embeddings: Tensor of shape [seq_len, hidden_dim] or [batch, seq_len, hidden_dim]
        segments: Parsed prompt segments with weights
        tokenizer: The tokenizer used to encode the prompt
        prompt: The original prompt string

    Returns:
        Weighted embeddings with same shape as input
    """
    # Handle batch dimension
    if embeddings.dim() == 3:
        # Process each batch item
        results = []
        for i in range(embeddings.size(0)):
            weighted = _apply_weights_single(
                embeddings[i], segments, tokenizer, prompt
            )
            results.append(weighted)
        return torch.stack(results)
    else:
        return _apply_weights_single(embeddings, segments, tokenizer, prompt)


def _apply_weights_single(
    embeddings: torch.Tensor,
    segments: list[PromptSegment],
    tokenizer,
    prompt: str,
) -> torch.Tensor:
    """Apply weights to a single sequence of embeddings."""
    # Build weight map from segments
    # This is a simplified approach that works for most cases
    weights = torch.ones(embeddings.size(0), device=embeddings.device, dtype=embeddings.dtype)

    # Strip weight/blend syntax from prompt to get plain text
    plain_prompt = _strip_syntax(prompt)

    # Tokenize the plain prompt to get token positions
    tokens = tokenizer.encode(plain_prompt, add_special_tokens=False)

    # Build character-to-weight mapping from segments
    char_weights = _build_char_weights(segments)

    # Map tokens to their positions in the original text
    token_positions = _get_token_positions(tokenizer, plain_prompt, tokens)

    # Assign weights to each token based on character position
    for i, (start, end) in enumerate(token_positions):
        if i < len(weights) and start < len(char_weights):
            # Use the weight of the first character of the token
            weights[i] = char_weights[start]

    # Apply weights
    return embeddings * weights.unsqueeze(-1)


def _strip_syntax(prompt: str) -> str:
    """
    Strip weight and blend syntax from prompt, leaving just text.

    (word:2) -> word
    [cat|dog] -> cat dog
    """
    result = ""
    i = 0
    while i < len(prompt):
        if prompt[i] == "(":
            # Find matching close and extract content
            depth = 1
            j = i + 1
            while j < len(prompt) and depth > 0:
                if prompt[j] == "(":
                    depth += 1
                elif prompt[j] == ")":
                    depth -= 1
                j += 1
            if depth == 0:
                content = prompt[i + 1 : j - 1]
                # Remove :weight suffix
                colon = content.rfind(":")
                if colon > 0:
                    try:
                        float(content[colon + 1 :])
                        content = content[:colon]
                    except ValueError:
                        pass
                result += _strip_syntax(content)
                i = j
            else:
                result += prompt[i]
                i += 1
        elif prompt[i] == "[":
            # Find matching close
            depth = 1
            j = i + 1
            while j < len(prompt) and depth > 0:
                if prompt[j] == "[":
                    depth += 1
                elif prompt[j] == "]":
                    depth -= 1
                j += 1
            if depth == 0:
                content = prompt[i + 1 : j - 1]
                # Replace | with space
                content = content.replace("|", " ")
                result += _strip_syntax(content)
                i = j
            else:
                result += prompt[i]
                i += 1
        else:
            result += prompt[i]
            i += 1
    return result


def _build_char_weights(segments: list[PromptSegment]) -> list[float]:
    """Build a list of weights, one per character of reconstructed text."""
    weights = []
    for seg in segments:
        if seg.is_blend():
            # For blends, use the first concept's text length
            blend_text = " ".join(s.text for s in seg.blend)
            weights.extend([seg.weight] * len(blend_text))
        else:
            weights.extend([seg.weight] * len(seg.text))
    return weights


@dataclass
class BlendInfo:
    """Information about a blend segment for post-encoding processing."""
    char_start: int  # Start position in clean text
    char_end: int    # End position in clean text
    segment: "PromptSegment"  # The original segment with blend list


def prepare_prompt_for_encoding(
    prompt: str,
) -> tuple[str, list[float] | None, list[BlendInfo]]:
    """
    Prepare a prompt for encoding by handling weight syntax properly.

    This function:
    1. Parses the prompt to extract segments with weights
    2. EXCLUDES segments with weight=0 entirely (they won't be tokenized at all)
    3. Builds per-character weights for remaining segments
    4. Tracks blend segments for post-encoding processing

    Args:
        prompt: Raw prompt string, possibly with (word:weight) syntax

    Returns:
        Tuple of (clean_prompt, char_weights, blend_infos):
        - clean_prompt: Text with syntax removed, weight=0 segments excluded
        - char_weights: Per-character weight list, or None if no weighting needed
        - blend_infos: List of BlendInfo for blends that need post-processing
    """
    if not has_special_syntax(prompt):
        return prompt, None, []

    segments = parse_prompt(prompt)

    # Check if there are any non-default weights or blends
    has_weights = any(seg.weight != 1.0 for seg in segments)
    has_blends = any(seg.is_blend() for seg in segments)

    if not has_weights and not has_blends:
        return _strip_syntax(prompt), None, []

    # Build clean text, char weights, and track blends
    text_parts = []
    char_weights = []
    blend_infos = []
    current_pos = 0

    for seg in segments:
        if seg.weight == 0:
            # Exclude entirely - this segment won't exist in the tokenized output
            continue

        if seg.is_blend():
            # For blends, use first concept's text as placeholder
            blend_text = seg.blend[0].text if seg.blend else ""
            char_start = current_pos
            char_end = current_pos + len(blend_text)

            # Track this blend for post-processing
            blend_infos.append(BlendInfo(
                char_start=char_start,
                char_end=char_end,
                segment=seg,
            ))

            text_parts.append(blend_text)
            char_weights.extend([seg.weight] * len(blend_text))
            current_pos = char_end
        else:
            text_parts.append(seg.text)
            char_weights.extend([seg.weight] * len(seg.text))
            current_pos += len(seg.text)

    clean_prompt = "".join(text_parts)

    # Check if all remaining weights are 1.0 (blends still need processing)
    weights_result = char_weights if any(w != 1.0 for w in char_weights) else None

    return clean_prompt, weights_result, blend_infos


def _get_token_positions(
    tokenizer, text: str, token_ids: list[int]
) -> list[tuple[int, int]]:
    """
    Get character positions for each token in the text.

    Returns list of (start, end) tuples.
    """
    positions = []
    current_pos = 0

    for token_id in token_ids:
        token_text = tokenizer.decode([token_id])
        # Find this token in the remaining text
        idx = text.find(token_text, current_pos)
        if idx >= 0:
            positions.append((idx, idx + len(token_text)))
            current_pos = idx + len(token_text)
        else:
            # Token not found, use current position
            positions.append((current_pos, current_pos + 1))

    return positions


def blend_embeddings(
    encoder_fn: Callable[[str], torch.Tensor],
    segment: PromptSegment,
) -> torch.Tensor:
    """
    Blend embeddings from multiple concepts.

    Args:
        encoder_fn: Function that encodes a text string to embeddings
        segment: A PromptSegment with blend list

    Returns:
        Blended embedding tensor
    """
    if not segment.is_blend():
        raise ValueError("Segment is not a blend")

    embeddings = []
    for concept in segment.blend:
        # Encode the concept
        emb = encoder_fn(concept.text)
        # Apply concept's individual weight
        emb = emb * concept.weight
        # Average pool to single embedding if multi-token
        if emb.dim() == 2 and emb.size(0) > 1:
            emb = emb.mean(dim=0, keepdim=True)
        embeddings.append(emb)

    # Average all concept embeddings
    blended = torch.stack(embeddings).mean(dim=0)

    # Apply outer weight
    blended = blended * segment.weight

    return blended


def has_special_syntax(prompt: str) -> bool:
    """
    Check if a prompt contains weighting or blending syntax.

    Args:
        prompt: The prompt to check

    Returns:
        True if prompt contains ( : ) or [ | ]
    """
    return "(" in prompt or "[" in prompt
