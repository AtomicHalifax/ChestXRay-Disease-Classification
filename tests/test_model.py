"""Model / Grad-CAM smoke tests (CPU, random weights, no downloads)."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from PIL import Image  # noqa: E402

from chexpert_cls.config import IMAGE_SIZE, TARGET_DISEASES  # noqa: E402
from chexpert_cls.gradcam import gradcam, overlay  # noqa: E402
from chexpert_cls.model import build_model  # noqa: E402
from chexpert_cls.predict import gradcam_image, predict_image  # noqa: E402


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return build_model(pretrained=False).eval()


def _xray(size=(320, 390)):
    rng = np.random.default_rng(0)
    return Image.fromarray(rng.integers(0, 255, size, dtype=np.uint8))  # 2-D uint8 -> grayscale


def test_output_shape(model):
    out = model(torch.randn(2, 3, IMAGE_SIZE, IMAGE_SIZE))
    assert out.shape == (2, len(TARGET_DISEASES))


def test_predict_image_returns_probabilities(model):
    probs = predict_image(model, _xray())
    assert list(probs) == TARGET_DISEASES
    assert all(0.0 <= v <= 1.0 for v in probs.values())


def test_gradcam_shape_and_range(model):
    x = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)
    cam, prob = gradcam(model, x, class_idx=4)
    assert cam.shape == (IMAGE_SIZE, IMAGE_SIZE)
    assert cam.min() >= 0.0 and cam.max() <= 1.0
    assert 0.0 <= prob <= 1.0


def test_gradcam_matches_model_forward(model):
    """Grad-CAM re-implements the forward pass; it must agree with model()."""
    x = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)
    with torch.no_grad():
        ref = torch.sigmoid(model(x))[0, 1].item()
    _, prob = gradcam(model, x, class_idx=1)
    assert prob == pytest.approx(ref, abs=1e-5)


def test_overlay_and_gradcam_image(model):
    ov = gradcam_image(model, _xray(), "Edema")
    assert ov.shape == (IMAGE_SIZE, IMAGE_SIZE, 3)
    assert overlay(np.zeros((4, 4, 3)), np.ones((4, 4))).max() <= 1.0
