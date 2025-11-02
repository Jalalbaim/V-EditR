# src/grounding/boxes_masks.py
from __future__ import annotations
import numpy as np
import torch
from typing import List, Tuple, Dict, Any

def to_xyxy(boxes: np.ndarray) -> np.ndarray:
    # boxes already xyxy in GroundingDINO
    return boxes

def _to_np(x):
    if isinstance(x, np.ndarray):
        return x
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    # lists etc.
    return np.array(x)


def nms_xyxy(boxes: np.ndarray, scores: np.ndarray, iou_thresh: float) -> List[int]:
    import numpy as np
    import torch

    boxes = boxes if isinstance(boxes, np.ndarray) else _to_np(boxes)
    scores = scores if isinstance(scores, np.ndarray) else _to_np(scores)

    try:
        import torchvision
        t_boxes = torch.as_tensor(boxes, dtype=torch.float32)
        t_scores = torch.as_tensor(scores, dtype=torch.float32)
        keep = torchvision.ops.nms(t_boxes, t_scores, iou_thresh)
        return keep.cpu().numpy().tolist()
    except Exception:
        # Greedy NMS
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            if order.size == 1:
                break
            ious = _iou(boxes[i], boxes[order[1:]])
            inds = np.where(ious <= iou_thresh)[0]
            order = order[inds + 1]
        return keep

def _iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    xA = np.maximum(a[0], b[:,0]); yA = np.maximum(a[1], b[:,1])
    xB = np.minimum(a[2], b[:,2]); yB = np.minimum(a[3], b[:,3])
    inter = np.maximum(0, xB - xA) * np.maximum(0, yB - yA)
    areaA = (a[2]-a[0])*(a[3]-a[1])
    areaB = (b[:,2]-b[:,0])*(b[:,3]-b[:,1])
    return inter / (areaA + areaB - inter + 1e-6)

def sam_masks_from_boxes(predictor, image_np_bgr: np.ndarray, boxes_xyxy: np.ndarray) -> np.ndarray:
    """
    Returns boolean masks [N, H, W] given image and list of boxes.
    """
    predictor.set_image(image_np_bgr)  # SAM expects original image (BGR/RGB depends on preprocess upstream)
    masks_all = []
    for b in boxes_xyxy:
        box = np.array(b, dtype=np.float32)
        masks, _, _ = predictor.predict(point_coords=None, point_labels=None, box=box[None, :], multimask_output=False)
        masks_all.append(masks[0].astype(bool))
    if not masks_all:
        return np.zeros((0, image_np_bgr.shape[0], image_np_bgr.shape[1]), dtype=bool)
    return np.stack(masks_all, axis=0)
