"""Run the released model on one or more chest X-ray images.

    python -m chexpert_cls.predict path/to/xray.jpg --gradcam out_dir/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .config import IMAGE_SIZE, TARGET_DISEASES
from .data import build_transforms
from .gradcam import gradcam, overlay
from .model import load_model


@torch.no_grad()
def predict_image(model: torch.nn.Module, image: Image.Image, device="cpu") -> dict[str, float]:
    x = build_transforms(train=False)(image.convert("RGB")).unsqueeze(0).to(device)
    probs = torch.sigmoid(model(x))[0].cpu().numpy()
    return {d: float(p) for d, p in zip(TARGET_DISEASES, probs)}


def gradcam_image(model, image: Image.Image, disease: str, device="cpu") -> np.ndarray:
    """RGB overlay (H, W, 3) float in [0, 1] for one disease."""
    rgb = np.asarray(image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE)), dtype=np.float32) / 255.0
    x = build_transforms(train=False)(image.convert("RGB")).unsqueeze(0).to(device)
    cam, _ = gradcam(model, x, TARGET_DISEASES.index(disease))
    return overlay(rgb, cam)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("images", nargs="+", type=Path)
    p.add_argument("--weights", type=Path, default=None,
                   help="Local checkpoint. Downloads from Hugging Face if omitted.")
    p.add_argument("--gradcam", type=Path, default=None,
                   help="Directory to write Grad-CAM overlays for the top finding.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args(argv)

    model = load_model(args.weights, args.device)
    results = {}
    for path in args.images:
        img = Image.open(path)
        probs = predict_image(model, img, args.device)
        results[str(path)] = probs
        if args.gradcam:
            args.gradcam.mkdir(parents=True, exist_ok=True)
            top = max(probs, key=probs.get)
            ov = gradcam_image(model, img, top, args.device)
            out = args.gradcam / f"{path.stem}_{top.replace(' ', '_')}.png"
            Image.fromarray((ov * 255).astype("uint8")).save(out)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
