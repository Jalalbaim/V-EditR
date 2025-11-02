# src/grounding/models.py
from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Optional
import torch

@dataclass
class DinoCfg:
    config: str               # NEW
    ckpt: str
    box_threshold: float
    text_threshold: float
    nms_iou: float
    max_detections_per_target: int

@dataclass
class SamCfg:
    ckpt: str
    variant: str  # "vit_h" | "vit_l" | "vit_b"

@dataclass
class GroundingCfg:
    device: str
    dino: DinoCfg
    sam: SamCfg

def load_grounding_cfg(yml: dict) -> GroundingCfg:
    g = yml.get("grounding", {})
    d = g.get("dino", {})
    s = g.get("sam", {})
    return GroundingCfg(
        device=g.get("device", "cuda" if torch.cuda.is_available() else "cpu"),
        dino=DinoCfg(
            config=d.get("config", ""),                          # NEW
            ckpt=d.get("ckpt", ""),
            box_threshold=float(d.get("box_threshold", 0.25)),
            text_threshold=float(d.get("text_threshold", 0.25)),
            nms_iou=float(d.get("nms_iou", 0.5)),
            max_detections_per_target=int(d.get("max_detections_per_target", 3)),
        ),
        sam=SamCfg(
            ckpt=s.get("ckpt", ""),
            variant=s.get("variant", "vit_h"),
        ),
    )

def try_load_groundingdino(cfg: DinoCfg, device: str):
    try:
        from groundingdino.util.inference import load_model
        if not os.path.isfile(cfg.ckpt):
            return None, "Missing GroundingDINO checkpoint"
        # Prefer the 2-arg API (config + ckpt). Fallback to old 1-arg if needed.
        try:
            if not os.path.isfile(cfg.config):
                return None, "Missing GroundingDINO config (.py)"
            model = load_model(cfg.config, cfg.ckpt)  # current API
        except TypeError:
            # older API variant
            model = load_model(cfg.ckpt)
        model.to(device)
        model.eval()
        return model, None
    except Exception as e:
        return None, f"GroundingDINO import/load failed: {e}"

def try_load_sam(cfg: SamCfg, device: str):
    try:
        from segment_anything import sam_model_registry, SamPredictor
        if not os.path.isfile(cfg.ckpt):
            return None, None, "Missing SAM checkpoint"
        sam = sam_model_registry[cfg.variant](checkpoint=cfg.ckpt)
        sam.to(device)
        predictor = SamPredictor(sam)
        return sam, predictor, None
    except Exception as e:
        return None, None, f"SAM import/load failed: {e}"
