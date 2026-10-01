"""Model construction and checkpoint loading."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models

from .config import HF_REPO_ID, HF_WEIGHTS_FILE, TARGET_DISEASES


def build_model(num_classes: int = len(TARGET_DISEASES), pretrained: bool = True) -> nn.Module:
    """DenseNet121 with the ImageNet head swapped for a multi-label head."""
    weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
    model = models.densenet121(weights=weights)
    model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    return model


def resolve_weights(path: str | Path | None = None) -> Path:
    """Return a local checkpoint path, downloading from the Hugging Face Hub
    when no path is given or the file does not exist."""
    if path is not None and Path(path).exists():
        return Path(path)
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(repo_id=HF_REPO_ID, filename=HF_WEIGHTS_FILE))


def load_model(path: str | Path | None = None, device: str | torch.device = "cpu") -> nn.Module:
    model = build_model(pretrained=False)
    state = torch.load(resolve_weights(path), map_location=device, weights_only=True)
    # tolerate checkpoints saved as {"model": state_dict, ...}
    if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
        state = state["model"]
    model.load_state_dict(state)
    return model.to(device).eval()
