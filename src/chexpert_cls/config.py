"""Shared constants. Keep these in sync with the released checkpoint."""

TARGET_DISEASES = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Pleural Effusion",
]

IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Released weights (see models/MODEL.md). Bump MODEL_VERSION whenever the
# checkpoint changes; the API reports it on every response.
MODEL_VERSION = "densenet121-chexpert5-v1.0"
HF_REPO_ID = "AtomicHalifax/ChestXRay-DenseNet121"
HF_WEIGHTS_FILE = "best_densenet121.pth"
