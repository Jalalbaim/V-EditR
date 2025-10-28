from PIL import Image, ImageDraw, ImageFont

def apply_dummy(img: Image.Image, op_type: str, box, label: str, color="red") -> Image.Image:
    out = img.copy()
    draw = ImageDraw.Draw(out, "RGBA")
    x1,y1,x2,y2 = box
    # simple visual cue: colored ellipse in the box
    draw.ellipse([x1, y1, x2, y2], outline=(255,255,255,180), width=6)
    # fill lightly with color hint
    fill_color = {
        "red": (255,0,0,60), "blue": (0,0,255,60), "green": (0,255,0,60),
        "black": (0,0,0,60), "white": (255,255,255,60), "yellow": (255,255,0,60),
        "silver": (180,180,200,60), "gray": (128,128,128,60), "grey": (128,128,128,60)
    }.get(color or "red", (255,0,0,60))
    draw.rectangle([x1, y1, x2, y2], fill=fill_color)
    # label
    try:
        font = ImageFont.load_default()
        draw.text((x1+8, y1+8), f"{op_type}:{label}", fill=(255,255,255,220), font=font)
    except Exception:
        pass
    return out
