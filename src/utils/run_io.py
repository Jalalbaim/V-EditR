import json, os, time
from pathlib import Path
from PIL import Image

def make_run_dir(base="runs", tag="dev"):
    ts = time.strftime("%Y%m%d_%H%M%S")
    run_dir = Path(base) / f"{ts}_{tag}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "artifacts").mkdir(exist_ok=True)
    return run_dir

def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def load_image(path):
    return Image.open(path).convert("RGB")

def save_image(img, path):
    img.save(path)
