from typing import Tuple, Dict, Any
from PIL import Image, ImageDraw

def locate_dummy(img: Image.Image, target_name: str) -> Dict[str, Any]:
    w, h = img.size
    bw, bh = int(0.3*w), int(0.3*h)
    x1, y1 = (w - bw)//2, (h - bh)//2
    x2, y2 = x1 + bw, y1 + bh
    return {"box": (x1, y1, x2, y2), "mask_hint": (x1, y1, x2, y2)}

def overlay_box(img: Image.Image, box: Tuple[int,int,int,int], color=(255,0,0)) -> Image.Image:
    out = img.copy()
    draw = ImageDraw.Draw(out, "RGBA")
    x1,y1,x2,y2 = box
    draw.rectangle([x1,y1,x2,y2], outline=color, width=4)
    draw.rectangle([x1,y1,x2,y2], fill=(color[0], color[1], color[2], 40))
    return out
