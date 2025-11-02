# src/grounding/locate.py
from __future__ import annotations
from typing import Dict, Any, List, Tuple
from pathlib import Path
import json
import numpy as np
import torch
from PIL import Image
from tempfile import NamedTemporaryFile

from .models import load_grounding_cfg, try_load_groundingdino, try_load_sam
from .boxes_masks import nms_xyxy, sam_masks_from_boxes
from .visualize import draw_boxes, draw_masks
from groundingdino.util.inference import predict, load_image


# ---------- helpers ----------

def _to_np(x):
    if isinstance(x, np.ndarray):
        return x
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.array(x)

def _dummy_center_box(img: Image.Image) -> Tuple[int, int, int, int]:
    w, h = img.size
    bw, bh = int(0.3 * w), int(0.3 * h)
    x1, y1 = (w - bw) // 2, (h - bh) // 2
    return (x1, y1, x1 + bw, y1 + bh)


# ---------- main ----------

def locate_plan_aware(
    img: Image.Image,
    plan,
    cfg_yml: dict,
    save_debug_dir: Path | None = None
) -> Dict[str, Any]:

    gcfg = load_grounding_cfg(cfg_yml)
    device = gcfg.device

    # Load models (graceful on failure)
    dino, err_dino = try_load_groundingdino(gcfg.dino, device)
    sam, predictor, err_sam = try_load_sam(gcfg.sam, device)

    if dino is None:
        box = _dummy_center_box(img)
        if save_debug_dir:
            (save_debug_dir / "GROUNDING_FALLBACK.txt").write_text(f"{err_dino}\n{err_sam or ''}")
            draw_boxes(img, [box], labels=["dummy"]).save(save_debug_dir / "grounding_preview.jpg")
        return {
            "targets": [{
                "name": plan.targets[0].name if plan.targets else "object",
                "prompt": "fallback",
                "boxes": [list(map(int, box))],
                "scores": [1.0],
                "masks": None
            }],
            "meta": {"fallback": True, "errors": {"dino": err_dino, "sam": err_sam}}
        }

    # Prompts from plan
    text_prompts, target_names = [], []
    for t in plan.targets:
        text_prompts.append((f"{' '.join(t.attributes)} {t.name}").strip() if t.attributes else t.name)
        target_names.append(t.name)

    # DINO preproc (expects a file path)
    with NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        img.convert("RGB").save(tmp.name, "JPEG")
        temp_path = tmp.name

    # We only use the returned tensor; for width/height we rely on *img*
    _, image_tensor = load_image(temp_path)
    W, H = img.size  # <— FIX: get size from the original PIL image

    all_targets: List[Dict[str, Any]] = []

    for i, prompt in enumerate(text_prompts):
        boxes, logits, phrases = predict(
            model=dino,
            image=image_tensor,
            caption=prompt,
            box_threshold=gcfg.dino.box_threshold,
            text_threshold=gcfg.dino.text_threshold
        )

        boxes = _to_np(boxes)                 # [N,4], may be normalized
        scores = _to_np(logits).reshape(-1)

        # Retry with bare noun if attribute prompt gave nothing
        if boxes.size == 0 and " " in prompt:
            base = prompt.split()[-1]
            boxes, logits, _ = predict(
                model=dino, image=image_tensor, caption=base,
                box_threshold=gcfg.dino.box_threshold, text_threshold=gcfg.dino.text_threshold
            )
            boxes = _to_np(boxes)
            scores = _to_np(logits).reshape(-1)

        if boxes.size > 0:
            # If <=1.5, treat as normalized; convert to pixel xyxy
            if float(boxes.max()) <= 1.5:
                xyxy = boxes * np.array([W, H, W, H], dtype=np.float32)  # assume normalized xyxy
                # If degenerate for all, interpret as cxcywh
                deg = (xyxy[:, 2] <= xyxy[:, 0]) | (xyxy[:, 3] <= xyxy[:, 1])
                if deg.all():
                    cxcywh = boxes * np.array([W, H, W, H], dtype=np.float32)
                    x1y1 = cxcywh[:, :2] - cxcywh[:, 2:] / 2.0
                    x2y2 = cxcywh[:, :2] + cxcywh[:, 2:] / 2.0
                    xyxy = np.concatenate([x1y1, x2y2], axis=1)
                # clip
                xyxy[:, 0] = np.clip(xyxy[:, 0], 0, W - 1)
                xyxy[:, 1] = np.clip(xyxy[:, 1], 0, H - 1)
                xyxy[:, 2] = np.clip(xyxy[:, 2], 0, W - 1)
                xyxy[:, 3] = np.clip(xyxy[:, 3], 0, H - 1)
                boxes = xyxy.astype(np.float32)

            # NMS + per-target cap
            keep = nms_xyxy(boxes, scores, gcfg.dino.nms_iou)
            boxes = boxes[keep]
            scores = scores[keep]
            if len(boxes) > gcfg.dino.max_detections_per_target:
                order = np.argsort(scores)[::-1][:gcfg.dino.max_detections_per_target]
                boxes = boxes[order]
                scores = scores[order]
        else:
            boxes = np.zeros((0, 4), dtype=np.float32)
            scores = np.zeros((0,), dtype=np.float32)

        all_targets.append({
            "name": target_names[i],
            "prompt": prompt,
            "boxes": boxes.astype(np.int32).tolist(),
            "scores": scores.astype(float).tolist(),
            "masks": None
        })

    # SAM masks (if available)
    if predictor is not None:
        image_np = np.array(img.convert("RGB"))
        for t in all_targets:
            if len(t["boxes"]) == 0:
                t["masks"] = None
                continue
            masks = sam_masks_from_boxes(predictor, image_np, np.array(t["boxes"], dtype=np.float32))
            t["masks"] = masks  # boolean [N,H,W]

    # Debug artifacts
    if save_debug_dir:
        for t in all_targets:
            if len(t["boxes"]) > 0:
                draw_boxes(
                    img, t["boxes"],
                    labels=[f"{t['name']}:{i}" for i in range(len(t["boxes"]))]
                ).save(save_debug_dir / f"boxes_{t['name']}.jpg")
            if isinstance(t.get("masks"), np.ndarray) and t["masks"].size > 0:
                draw_masks(img, t["masks"]).save(save_debug_dir / f"masks_{t['name']}.jpg")

        (save_debug_dir / "targets.json").write_text(
            json.dumps([
                {
                    **{k: v for k, v in t.items() if k != "masks"},
                    "masks": None if t.get("masks") is None else [m.shape for m in t["masks"]]
                } for t in all_targets
            ], indent=2)
        )

    return {"targets": all_targets, "meta": {"fallback": False}}
