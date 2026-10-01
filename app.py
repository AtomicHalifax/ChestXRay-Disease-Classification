"""Gradio demo: upload a chest X-ray, get five finding probabilities and a
Grad-CAM heatmap. Runs on CPU; deployable as a Hugging Face Space.

    pip install -r requirements.txt gradio
    python app.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import gradio as gr  # noqa: E402
import numpy as np  # noqa: E402

from chexpert_cls.config import TARGET_DISEASES  # noqa: E402
from chexpert_cls.model import load_model  # noqa: E402
from chexpert_cls.predict import gradcam_image, predict_image  # noqa: E402

MODEL = load_model(Path("models/best_densenet121.pth"))

DISCLAIMER = (
    "Research demo, not a medical device. Trained on CheXpert (Stanford, adult "
    "inpatients); predictions on other populations or scanners are unreliable."
)


def run(image, disease):
    if image is None:
        return None, None
    probs = predict_image(MODEL, image)
    target = disease if disease != "Top finding" else max(probs, key=probs.get)
    heat = (gradcam_image(MODEL, image, target) * 255).astype(np.uint8)
    return probs, heat


with gr.Blocks(title="Chest X-ray classifier (CheXpert, DenseNet121)") as demo:
    gr.Markdown("## Chest X-ray finding classifier\n" + DISCLAIMER)
    with gr.Row():
        with gr.Column():
            inp = gr.Image(type="pil", label="Chest X-ray (frontal)")
            which = gr.Dropdown(["Top finding", *TARGET_DISEASES], value="Top finding",
                                label="Explain which finding")
            btn = gr.Button("Analyze", variant="primary")
        with gr.Column():
            out_probs = gr.Label(num_top_classes=5, label="Probabilities")
            out_cam = gr.Image(label="Grad-CAM")
    btn.click(run, [inp, which], [out_probs, out_cam])

if __name__ == "__main__":
    demo.launch()
