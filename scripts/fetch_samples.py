"""Download the 10 demo chest X-rays and generate "bad input" test images.

    python scripts/fetch_samples.py            # -> samples/xray/, samples/bad_inputs/

The X-rays come from Wikimedia Commons under CC0 / public domain / CC BY-SA
(see samples/manifest.json and the generated samples/CREDITS.md). They are
fetched on demand instead of committed, and resized to at most 1024 px.
Their "expected" finding is the uploader's description, not a verified label.
"""

from __future__ import annotations

import io
import json
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "samples"
UA = "chexpert-cls-sample-fetcher/1.0 (https://github.com/AtomicHalifax/ChestXRay-Disease-Classification)"


def fetch(url: str) -> Image.Image:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=60) as r:
        return Image.open(io.BytesIO(r.read()))


def make_bad_inputs(out: Path) -> None:
    """Images the API should warn about (or reject with CHEXPERT_STRICT_INPUT=1)."""
    out.mkdir(parents=True, exist_ok=True)
    y, x = np.mgrid[0:512, 0:512]
    colour = np.stack([x / 2, y / 2, 255 - x / 2], axis=-1).astype(np.uint8)  # rainbow "photo"
    Image.fromarray(colour).save(out / "colour_photo.png")
    Image.fromarray(np.full((64, 64), 128, np.uint8)).save(out / "tiny_64px.png")
    Image.fromarray(np.tile(np.linspace(0, 255, 1200, dtype=np.uint8), (300, 1))).save(out / "panorama_4to1.png")
    (out / "not_an_image.png").write_bytes(b"this is text pretending to be a png")


def main() -> None:
    manifest = json.loads((SAMPLES / "manifest.json").read_text(encoding="utf-8"))
    xdir = SAMPLES / "xray"
    xdir.mkdir(parents=True, exist_ok=True)
    credits = ["# Sample image credits", "", "All images from Wikimedia Commons.", ""]
    for item in manifest["images"]:
        dest = xdir / f"{item['id']}.png"
        if not dest.exists():
            img = fetch(item["url"]).convert("L")
            img.thumbnail((1024, 1024))
            img.save(dest)
            print(f"saved {dest.relative_to(ROOT)}  ({item['expected']})")
        lic = f"[{item['license']}]({item['license_url']})" if item["license_url"] else item["license"]
        credits.append(f"- `{dest.name}`: [{item['title']}]({item['source_page']}) by {item['author']}, {lic}. "
                       "Converted to grayscale and resized.")
    (SAMPLES / "CREDITS.md").write_text("\n".join(credits) + "\n", encoding="utf-8")
    make_bad_inputs(SAMPLES / "bad_inputs")
    print("bad inputs written to samples/bad_inputs/")


if __name__ == "__main__":
    main()
