"""Export the released checkpoint to ONNX for in-browser inference (website).

The graph returns (probs [1,5], cams [1,5,7,7]). For DenseNet121 the head is
ReLU -> global average pool -> Linear, so d(logit_c)/d(A) = W_c / (7*7)
everywhere and Grad-CAM reduces exactly to ReLU(sum_k W_ck A_k). Baking that
into the graph lets the browser draw the heatmap without gradients.

    python scripts/export_onnx.py --out docs/model.onnx
"""

import argparse

import numpy as np
import torch
import torch.nn.functional as F

from chexpert_cls.config import IMAGE_SIZE
from chexpert_cls.gradcam import gradcam
from chexpert_cls.model import load_model


class Exported(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        acts = F.relu(self.m.features(x))
        logits = self.m.classifier(F.adaptive_avg_pool2d(acts, 1).flatten(1))
        cams = F.relu(torch.einsum("ck,bkhw->bchw", self.m.classifier.weight, acts))
        return torch.sigmoid(logits), cams


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=None)
    ap.add_argument("--out", default="docs/model.onnx")
    args = ap.parse_args()

    model = load_model(args.weights).eval()
    net = Exported(model).eval()
    x = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)
    torch.onnx.export(net, x, args.out, input_names=["image"], output_names=["probs", "cams"],
                      opset_version=17, dynamo=False)

    # check: ONNX == PyTorch, and baked CAM == Grad-CAM from the API
    import onnxruntime as ort

    probs, cams = ort.InferenceSession(args.out).run(None, {"image": x.numpy()})
    with torch.no_grad():
        p_ref, _ = net(x)
    assert np.allclose(probs, p_ref.numpy(), atol=1e-4), "ONNX probs differ from PyTorch"
    for c in range(probs.shape[1]):
        ref, _ = gradcam(model, x, c)
        cam = torch.from_numpy(cams[:, c:c + 1])
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0].numpy()
        cam -= cam.min()
        cam /= max(cam.max(), 1e-12)
        assert np.abs(cam - ref).max() < 1e-3, f"CAM differs from Grad-CAM for class {c}"
    print(f"wrote {args.out}; ONNX matches PyTorch and Grad-CAM")


if __name__ == "__main__":
    main()
