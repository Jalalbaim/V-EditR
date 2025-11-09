# src/editors/real_editors.py
from __future__ import annotations
from typing import Optional, Tuple
import torch
from PIL import Image
from diffusers import StableDiffusionInstructPix2PixPipeline, StableDiffusionInpaintPipeline, ControlNetModel, StableDiffusionControlNetInpaintPipeline
import numpy as np

def load_instructpix2pix(model_name: str, device="cuda", use_fp16=True):
    dtype = torch.float16 if use_fp16 and torch.cuda.is_available() else torch.float32
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        model_name,
        torch_dtype=dtype,
        safety_checker=None
    ).to(device)
    pipe.enable_attention_slicing()
    return pipe

def load_addit(base_model: str, controlnet_model: str, device="cuda", use_fp16=True):
    dtype = torch.float16 if use_fp16 and torch.cuda.is_available() else torch.float32
    controlnet = ControlNetModel.from_pretrained(controlnet_model, torch_dtype=dtype)
    pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
        base_model,
        controlnet=controlnet,
        torch_dtype=dtype,
        safety_checker=None
    ).to(device)
    pipe.enable_attention_slicing()
    return pipe

def run_instructpix2pix(pipe, image: Image.Image, prompt: str, strength=0.8, guidance_scale=7.5, num_inference_steps=30):
    result = pipe(
        prompt=prompt,
        image=image,
        strength=strength,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps
    )
    return result.images[0]

def run_addit(pipe, image: Image.Image, mask: Optional[np.ndarray], prompt: str, num_inference_steps=40, guidance_scale=7.5):
    image_rgb = image.convert("RGB")
    if mask is not None:
        mask_pil = Image.fromarray((mask * 255).astype(np.uint8))
    else:
        # fallback: use full white mask
        mask_pil = Image.new("L", image.size, 255)
    result = pipe(
        prompt=prompt,
        image=image_rgb,
        mask_image=mask_pil,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps
    )
    return result.images[0]
