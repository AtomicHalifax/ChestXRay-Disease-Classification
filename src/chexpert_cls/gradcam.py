"""Dependency-free Grad-CAM for the DenseNet121 classifier."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def gradcam(model: torch.nn.Module, x: torch.Tensor, class_idx: int) -> tuple[np.ndarray, float]:
    """Grad-CAM (Selvaraju et al., 2017) on DenseNet121's final feature map.

    Re-runs DenseNet's forward explicitly (features -> ReLU -> GAP ->
    classifier) instead of using module hooks, because torchvision's
    DenseNet applies an in-place ReLU that breaks full backward hooks.

    Returns (heatmap [H, W] in [0, 1], probability for class_idx).
    """
    model.eval()
    model.zero_grad(set_to_none=True)
    with torch.enable_grad():
        acts = F.relu(model.features(x))
        acts.retain_grad()
        pooled = F.adaptive_avg_pool2d(acts, 1).flatten(1)
        logits = model.classifier(pooled)
        logits[0, class_idx].backward()

    weights = acts.grad.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((weights * acts).sum(dim=1, keepdim=True)).detach()
    cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
    cam = cam[0, 0].cpu().numpy()
    cam -= cam.min()
    if cam.max() > 0:
        cam /= cam.max()
    prob = float(torch.sigmoid(logits[0, class_idx]).item())
    return cam, prob


def overlay(rgb: np.ndarray, cam: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """Blend a heatmap onto an RGB image (both HxW, rgb float in [0, 1])."""
    import matplotlib

    heat = matplotlib.colormaps["jet"](cam)[..., :3]
    return np.clip((1 - alpha) * rgb + alpha * heat, 0, 1)
