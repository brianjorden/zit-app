"""
LoRA Management for Z-Image-Turbo.

Handles LoRA file discovery, loading, weight application with alpha scaling,
and original weight backup/restore for clean switching between configurations.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open


@dataclass
class LoRAConfig:
    """Configuration for a single LoRA."""

    name: str
    scale: float = 1.0

    def is_active(self) -> bool:
        """Check if this LoRA should be applied."""
        return self.name and self.name != "None" and self.scale != 0.0


class LoRAManager:
    """
    Manages LoRA loading and application for the transformer model.

    Features:
    - File discovery and caching
    - Robust key remapping for compatibility
    - Alpha scaling following standard LoRA math
    - Original weight backup/restore
    - State tracking for efficient switching
    """

    def __init__(self, lora_dir: Path):
        """
        Initialize the LoRA manager.

        Args:
            lora_dir: Directory containing .safetensors LoRA files
        """
        self.lora_dir = Path(lora_dir)
        self._original_weights: dict[str, torch.Tensor] = {}
        self._lora_cache: dict[str, dict[str, torch.Tensor]] = {}
        self._current_state: tuple[tuple[str, float], ...] | None = None

    def get_available_loras(self) -> list[str]:
        """
        Scan the loras directory for available .safetensors files.

        Returns:
            Sorted list of LoRA names (without .safetensors extension)
        """
        if not self.lora_dir.exists():
            return []
        return sorted([f.stem for f in self.lora_dir.glob("*.safetensors")])

    def load_lora(self, name: str) -> dict[str, torch.Tensor]:
        """
        Load LoRA weights from a safetensors file with key remapping.

        Args:
            name: LoRA name (without extension)

        Returns:
            Dict mapping module paths to tensors
        """
        if name in self._lora_cache:
            return self._lora_cache[name]

        lora_path = self.lora_dir / f"{name}.safetensors"
        if not lora_path.exists():
            raise FileNotFoundError(f"LoRA file not found: {lora_path}")

        state_dict = {}
        with safe_open(str(lora_path), framework="pt", device="cpu") as f:
            for key in f.keys():
                new_key = self._remap_key(key)
                state_dict[new_key] = f.get_tensor(key)

        self._lora_cache[name] = state_dict
        return state_dict

    def _remap_key(self, key: str) -> str:
        """
        Remap LoRA keys for Z-Image/Diffusers compatibility.

        Handles various training tool output formats.
        """
        new_key = key

        # Strip common training prefixes
        for prefix in ["diffusion_model.", "lora_unet.", "transformer."]:
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix) :]

        # Remap 'blocks' to 'transformer_blocks' (Z-Image uses transformer_blocks)
        if new_key.startswith("blocks."):
            new_key = new_key.replace("blocks.", "transformer_blocks.", 1)

        return new_key

    def _get_module_by_path(self, model: Any, path: str) -> Any:
        """
        Get a module from the model by dot-separated path.

        Args:
            model: The model to traverse
            path: Dot-separated path like "transformer_blocks.0.attn.to_q"

        Returns:
            The module at the specified path
        """
        parts = path.split(".")
        module = model
        for part in parts:
            if part.isdigit():
                module = module[int(part)]
            else:
                module = getattr(module, part)
        return module

    def apply_loras(
        self, transformer: Any, configs: list[LoRAConfig], force: bool = False
    ) -> int:
        """
        Apply LoRA configurations to the transformer.

        This merges LoRA weights directly into the model weights:
        new_weight = original_weight + (alpha/rank * scale) * (lora_B @ lora_A)

        Args:
            transformer: The transformer model to modify
            configs: List of LoRA configurations to apply
            force: Force re-application even if state hasn't changed

        Returns:
            Number of LoRA layers successfully applied
        """
        # Filter to active LoRAs only
        active_configs = [cfg for cfg in configs if cfg.is_active()]

        # Create hashable state for change detection
        new_state = (
            tuple((cfg.name, cfg.scale) for cfg in active_configs)
            if active_configs
            else None
        )

        # Skip if state hasn't changed
        if not force and new_state == self._current_state:
            return 0

        # Restore original weights first
        self.restore_original_weights(transformer)

        # If no active LoRAs, we're done
        if not active_configs:
            self._current_state = None
            return 0

        total_applied = 0

        for cfg in active_configs:
            try:
                lora_state_dict = self.load_lora(cfg.name)
            except FileNotFoundError:
                print(f"Warning: LoRA file not found: {cfg.name}")
                continue

            # Group by module path
            lora_groups = self._group_lora_weights(lora_state_dict)

            applied = self._apply_lora_group(
                transformer, lora_groups, cfg.scale, cfg.name
            )
            total_applied += applied

        # Clear CUDA cache to reduce fragmentation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self._current_state = new_state
        return total_applied

    def _group_lora_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, dict[str, torch.Tensor]]:
        """
        Group LoRA weights by module path.

        Returns dict mapping module paths to dicts with lora_A, lora_B, and optional alpha.
        """
        groups: dict[str, dict[str, torch.Tensor]] = {}

        for key, tensor in state_dict.items():
            if ".lora_A." in key:
                base = key.split(".lora_A.")[0]
                groups.setdefault(base, {})["lora_A"] = tensor
            elif ".lora_B." in key:
                base = key.split(".lora_B.")[0]
                groups.setdefault(base, {})["lora_B"] = tensor
            elif ".alpha" in key:
                base = key.split(".alpha")[0]
                groups.setdefault(base, {})["alpha"] = tensor

        return groups

    def _apply_lora_group(
        self,
        transformer: Any,
        groups: dict[str, dict[str, torch.Tensor]],
        user_scale: float,
        lora_name: str,
    ) -> int:
        """
        Apply a group of LoRA weights to the transformer.

        Returns number of layers successfully applied.
        """
        success_count = 0

        for module_path, weights in groups.items():
            if "lora_A" not in weights or "lora_B" not in weights:
                continue

            try:
                module = self._get_module_by_path(transformer, module_path)

                # Backup original weight if not already saved
                if module_path not in self._original_weights:
                    self._original_weights[module_path] = module.weight.data.clone().cpu()

                # Calculate scale factor with alpha
                rank = weights["lora_A"].shape[0]
                alpha = (
                    weights["alpha"].item() if "alpha" in weights else float(rank)
                )
                scale_factor = (alpha / rank) * user_scale

                # Apply: W = W + scale * (B @ A)
                dtype = module.weight.dtype
                device = module.weight.device
                lora_A = weights["lora_A"].to(device, dtype)
                lora_B = weights["lora_B"].to(device, dtype)
                delta = lora_B @ lora_A
                module.weight.data.add_(delta * scale_factor)

                success_count += 1

            except (AttributeError, IndexError, KeyError):
                # Module path doesn't match model structure - skip silently
                pass
            except Exception as e:
                print(f"Warning: Failed to apply LoRA to {module_path}: {e}")

        if success_count > 0:
            print(f"Applied {success_count} layers from {lora_name}")

        return success_count

    def restore_original_weights(self, transformer: Any) -> None:
        """
        Restore all modified weights to their original values.

        Args:
            transformer: The transformer model to restore
        """
        if not self._original_weights:
            return

        for module_path, weight_clone in self._original_weights.items():
            try:
                module = self._get_module_by_path(transformer, module_path)
                module.weight.data.copy_(weight_clone.to(device=module.weight.device))
            except Exception:
                # If restoration fails, the model structure may have changed
                pass

    def validate_state(self, configs: list[LoRAConfig]) -> bool:
        """
        Validate that the current LoRA state matches expected configs.

        Args:
            configs: Expected LoRA configurations

        Returns:
            True if current state matches expected
        """
        active_configs = [cfg for cfg in configs if cfg.is_active()]
        expected_state = (
            tuple((cfg.name, cfg.scale) for cfg in active_configs)
            if active_configs
            else None
        )
        return self._current_state == expected_state

    def clear_cache(self) -> None:
        """Clear the LoRA state dict cache to free memory."""
        self._lora_cache.clear()

    def get_current_loras(self) -> list[tuple[str, float]]:
        """
        Get the currently applied LoRAs.

        Returns:
            List of (name, scale) tuples
        """
        if self._current_state is None:
            return []
        return list(self._current_state)


def configs_from_ui_state(ui_state: list[dict]) -> list[LoRAConfig]:
    """
    Convert Gradio UI state to LoRAConfig list.

    Args:
        ui_state: List of dicts with 'name' and 'scale' keys

    Returns:
        List of LoRAConfig objects
    """
    return [
        LoRAConfig(name=item.get("name", "None"), scale=item.get("scale", 1.0))
        for item in ui_state
    ]


def format_loras_for_metadata(configs: list[LoRAConfig]) -> str:
    """
    Format LoRA configs for image metadata.

    Args:
        configs: List of LoRA configurations

    Returns:
        String like "style_lora:0.8, detail_lora:0.5" or "None"
    """
    active = [f"{cfg.name}:{cfg.scale}" for cfg in configs if cfg.is_active()]
    return ", ".join(active) if active else "None"
