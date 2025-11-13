# src/grounding/visualize.py
from __future__ import annotations
from typing import List, Tuple
from PIL import Image, ImageDraw, ImageFont
import numpy as np

def draw_boxes(img: Image.Image, boxes: List[Tuple[int,int,int,int]], labels=None, color=(0,255,0), alpha=70) -> Image.Image:
    out = img.convert("RGBA")
    overlay = Image.new("RGBA", out.size, (0,0,0,0))
    d = ImageDraw.Draw(overlay)
    for i, b in enumerate(boxes):
        x1, y1, x2, y2 = map(int, b)
        # S'assurer que les coordonnées sont dans le bon ordre
        x_min, x_max = min(x1, x2), max(x1, x2)
        y_min, y_max = min(y1, y2), max(y1, y2)
        # Vérifier que la boîte a une taille valide
        if x_max <= x_min or y_max <= y_min:
            continue  # Ignorer les boîtes invalides
        d.rectangle([x_min, y_min, x_max, y_max], outline=color+(255,), width=3)
        d.rectangle([x_min, y_min, x_max, y_max], fill=color+(alpha,))
        if labels is not None:
            try:
                font = ImageFont.load_default()
                d.text((x_min+4, y_min+4), str(labels[i]), fill=(255,255,255,230), font=font)
            except Exception:
                pass
    return Image.alpha_composite(out, overlay).convert("RGB")

def draw_masks(img: Image.Image, masks: np.ndarray, alpha=70) -> Image.Image:
    out = img.convert("RGBA")
    overlay = Image.new("RGBA", out.size, (0,0,0,0))
    d = ImageDraw.Draw(overlay)
    h, w = img.size[1], img.size[0]
    # merge masks with color tint per index
    for i, m in enumerate(masks):
        tint = (int(60+170*(i%3==0)), int(60+170*(i%3==1)), int(60+170*(i%3==2)), alpha)
        ys, xs = np.where(m)
        for y, x in zip(ys, xs):
            overlay.putpixel((x, y), blend(overlay.getpixel((x,y)), tint))
    return Image.alpha_composite(out, overlay).convert("RGB")

def blend(bg, fg):
    # naive alpha blend on a pixel (bg RGBA, fg RGBA)
    a = fg[3]/255.0
    return (int(bg[0]*(1-a)+fg[0]*a), int(bg[1]*(1-a)+fg[1]*a), int(bg[2]*(1-a)+fg[2]*a), 255)
