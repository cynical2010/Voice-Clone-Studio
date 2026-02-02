"""
Model utilities for AI model management.

Shared utilities for model loading, device management, and VRAM optimization.
"""

import torch
from pathlib import Path


def get_device():
    """Get the appropriate device (CUDA or CPU)."""
    if torch.cuda.is_available():
        return "cuda:0"
    return "cpu"


def get_dtype(device=None):
    """Get appropriate dtype based on device."""
    if device is None:
        device = get_device()

    if device.startswith("cuda"):
        return torch.bfloat16
    return torch.float32


def get_attention_implementation(user_preference="auto"):
    """
    Get list of attention implementations to try, in order.

    Args:
        user_preference: User's attention preference from config:
            - "auto": Try best options in order
            - "flash_attention_2": Use Flash Attention 2
            - "sdpa": Use Scaled Dot-Product Attention
            - "eager": Use eager attention

    Returns:
        List of attention mechanism strings to try
    """
    if user_preference == "flash_attention_2":
        return ["flash_attention_2", "sdpa", "eager"]
    elif user_preference == "sdpa":
        return ["sdpa", "flash_attention_2", "eager"]
    elif user_preference == "eager":
        return ["eager"]
    else:  # "auto"
        return ["flash_attention_2", "sdpa", "eager"]


def check_model_available_locally(model_name):
    """
    Check if model is available in local models directory.

    Args:
        model_name: Model name/path (e.g., "Qwen/Qwen3-TTS-12Hz-1.7B-Base")

    Returns:
        Path to local model or None if not found
    """
    models_dir = Path(__file__).parent.parent.parent / "models"

    # Try exact model name
    model_path = models_dir / model_name.split("/")[-1]
    if model_path.exists() and list(model_path.glob("*.safetensors")):
        return model_path

    return None


def empty_cuda_cache():
    """Empty CUDA cache if available."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def log_gpu_memory(label=""):
    """Log current GPU memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        label_str = f" ({label})" if label else ""
        print(f"GPU Memory{label_str}: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")


def get_trained_models(models_dir=None):
    """
    Find trained model checkpoints in the models directory.

    Args:
        models_dir: Path to models directory (defaults to project models folder)

    Returns:
        List of dicts with display_name, path, and speaker_name
    """
    if models_dir is None:
        models_dir = Path(__file__).parent.parent.parent.parent / "models"

    models = []
    if models_dir.exists():
        for folder in models_dir.iterdir():
            if folder.is_dir():
                for checkpoint in folder.glob("checkpoint-*"):
                    if checkpoint.is_dir():
                        models.append({
                            'display_name': f"{folder.name} - {checkpoint.name}",
                            'path': str(checkpoint),
                            'speaker_name': folder.name
                        })
    return models
