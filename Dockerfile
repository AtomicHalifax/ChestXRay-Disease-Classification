# CPU inference image for the CheXpert API.
#   docker build -t chexpert-api .
#   docker run -p 8000:8000 chexpert-api
# Bake the weights into the image (no network needed at start-up):
#   docker build --build-arg BAKE_WEIGHTS=1 -t chexpert-api .
FROM python:3.11-slim

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/home/app/.cache/huggingface

RUN useradd --create-home app
WORKDIR /app

# CPU-only torch keeps the image ~4x smaller than the default CUDA wheels
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

COPY pyproject.toml ./
COPY src ./src
RUN pip install ".[serve]"

USER app
ARG BAKE_WEIGHTS=0
RUN if [ "$BAKE_WEIGHTS" = "1" ]; then python -c "from chexpert_cls.model import resolve_weights; print(resolve_weights())"; fi

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"

CMD ["uvicorn", "chexpert_cls.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
