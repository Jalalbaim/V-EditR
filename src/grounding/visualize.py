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
        x1,y1,x2,y2 = map(int, b)
        d.rectangle([x1,y1,x2,y2], outline=color+(255,), width=3)
        d.rectangle([x1,y1,x2,y2], fill=color+(alpha,))
        if labels is not None:
            try:
                font = ImageFont.load_default()
                d.text((x1+4, y1+4), str(labels[i]), fill=(255,255,255,230), font=font)
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
