# src/editors/real_editors.py
from __future__ import annotations
from typing import Optional
import numpy as np
from PIL import Image
import torch

from diffusers import (
    StableDiffusionInstructPix2PixPipeline,
    ControlNetModel,
    StableDiffusionControlNetInpaintPipeline,
)

# Générateur de profondeur pour ControlNet (control_image)
try:
    from controlnet_aux import MidasDetector
    _HAS_MIDAS = True
except Exception:
    _HAS_MIDAS = False
    MidasDetector = None


def _dtype(use_fp16: bool):
    return torch.float16 if use_fp16 and torch.cuda.is_available() else torch.float32


def load_instructpix2pix(model_name: str, device="cuda", use_fp16=True):
    try:
        pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            model_name,
            torch_dtype=_dtype(use_fp16),
            safety_checker=None,
            use_safetensors=False,
            low_cpu_mem_usage=True,
        ).to(device)
        pipe.enable_attention_slicing()
        return pipe
    except Exception as e:
        raise RuntimeError(f"InstructPix2Pix load failed: {e}")


def load_addit(base_model: str, controlnet_model: str, device="cuda", use_fp16=True):
    try:
        cn = ControlNetModel.from_pretrained(
            controlnet_model,
            torch_dtype=_dtype(use_fp16),
            use_safetensors=False,
        )
        pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
            base_model,
            controlnet=cn,
            torch_dtype=_dtype(use_fp16),
            safety_checker=None,
            use_safetensors=False,
            low_cpu_mem_usage=True,
        ).to(device)
        pipe.enable_attention_slicing()
        return pipe
    except Exception as e:
        raise RuntimeError(f"Add-It load failed: {e}")


def _ensure_pil_rgb(x) -> Image.Image:
    if isinstance(x, Image.Image):
        return x.convert("RGB")
    if isinstance(x, np.ndarray):
        if x.dtype != np.uint8:
            x = np.clip(x, 0, 255).astype(np.uint8)
        return Image.fromarray(x).convert("RGB")
    raise TypeError(f"Expected PIL.Image or np.ndarray, got {type(x)}")


def _resize_multiple_of_8(img: Image.Image, max_side: int = 768) -> Image.Image:
    w, h = img.size
    scale = min(max_side / max(w, h), 1.0)
    nw, nh = int((w * scale) // 8 * 8), int((h * scale) // 8 * 8)
    nw = max(nw, 8)
    nh = max(nh, 8)
    if (nw, nh) != (w, h):
        return img.resize((nw, nh), Image.LANCZOS)
    return img


def _build_control_image(img_rgb: Image.Image, device: str) -> Image.Image:
    """
    Génère une carte de profondeur (control image) si controlnet_aux est dispo.
    Sinon, retourne un gris uniforme comme fallback.
    """
    if _HAS_MIDAS:
        try:
            midas = MidasDetector.from_pretrained("lllyasviel/Annotators")
            depth = midas(img_rgb)  # PIL Image (mono)
            return depth
        except Exception:
            pass
    # Fallback neutre
    return Image.new("L", img_rgb.size, 128)


def run_instructpix2pix(pipe, image: Image.Image, prompt: str,
                        strength=0.8, guidance_scale=7.5, num_inference_steps=30):
    img_rgb = _ensure_pil_rgb(image)
    img_rgb = _resize_multiple_of_8(img_rgb)
    out = pipe(
        prompt=prompt,
        image=img_rgb,
        strength=strength,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
    )
    if not hasattr(out, "images") or not out.images:
        raise RuntimeError("InstructPix2Pix returned no image.")
    return out.images[0]


def run_addit(pipe, image: Image.Image, mask: Optional[np.ndarray], prompt: str,
              num_inference_steps=40, guidance_scale=7.5):
    # --- Sécurité et normalisation ---
    if image is None:
        raise ValueError("run_addit: image is None (check image loading path).")

    img_rgb = _ensure_pil_rgb(image)
    img_rgb = _resize_multiple_of_8(img_rgb)

    # Masque (L, 255 = zone à inpeindre)
    if mask is not None:
        # Handle boolean masks
        if mask.dtype == bool:
            m = mask.astype(np.uint8) * 255
        elif mask.dtype == np.uint8:
            # Already uint8, use as-is (should be 0 or 255)
            m = mask
        else:
            # Other numeric types: threshold at 0.5 if normalized
            if mask.max() <= 1.0:
                m = (mask > 0.5).astype(np.uint8) * 255
            else:
                m = (mask > 127).astype(np.uint8) * 255
        
        mask_pil = Image.fromarray(m) if isinstance(m, np.ndarray) else m
        if mask_pil.mode != "L":
            mask_pil = mask_pil.convert("L")
        mask_pil = mask_pil.resize(img_rgb.size, Image.NEAREST)
    else:
        # Par défaut: autoriser l'inpainting partout (plein blanc)
        mask_pil = Image.new("L", img_rgb.size, 255)

    # Control image (profondeur) pour ControlNet
    control_img = _build_control_image(img_rgb, device=str(pipe.device))
    # S'assurer que la control_image a exactement la même taille que l'image
    if control_img.size != img_rgb.size:
        control_img = control_img.resize(img_rgb.size, Image.LANCZOS)

    out = pipe(
        prompt=prompt,
        image=img_rgb,
        mask_image=mask_pil,
        control_image=control_img,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
    )
    if not hasattr(out, "images") or not out.images:
        raise RuntimeError("Add-It returned no image.")
    return out.images[0]