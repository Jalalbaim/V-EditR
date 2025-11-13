# src/editors/edit_manager.py
from __future__ import annotations
from typing import Dict, Any
from PIL import Image
import numpy as np
import torch

from .real_editors import (
    load_instructpix2pix, load_addit,
    run_instructpix2pix, run_addit
)

class EditManager:
    def __init__(self, cfg: Dict[str, Any]):
        self.device = cfg["models"].get("device", "cuda" if torch.cuda.is_available() else "cpu")
        self.mode = cfg["models"]["editor"].get("mode", "auto")
        self.pipes = {"instruct": None, "addit": None}
        self.cfg = cfg

    def _get_instruct_pipe(self):
        if self.pipes["instruct"] is None:
            m = self.cfg["models"]["instructpix2pix"]
            self.pipes["instruct"] = load_instructpix2pix(m["name"], self.device, m.get("use_fp16", True))
        return self.pipes["instruct"]

    def _get_addit_pipe(self):
        if self.pipes["addit"] is None:
            m = self.cfg["models"]["addit"]
            self.pipes["addit"] = load_addit(m["base_model"], m["controlnet"], self.device, m.get("use_fp16", True))
        return self.pipes["addit"]

    def apply_edit(self, img: Image.Image, plan, grounding_info) -> Image.Image:
        """
        grounding_info = locate_plan_aware(...) output
        """
        op = plan.ops[0].type if plan.ops else "unknown"
        
        # Determine the appropriate mask based on operation type
        mask = None
        
        if op == "add":
            # For ADD operations: find the reference object (not the object to add)
            # The first target is typically the object to add, find other targets for reference
            reference_mask = None
            if len(grounding_info["targets"]) > 1:
                # Use the second target (reference object like "truck")
                ref_target = grounding_info["targets"][1]
                if ref_target.get("masks") is not None and len(ref_target["masks"]) > 0:
                    reference_mask = ref_target["masks"][0]
            
            if reference_mask is not None:
                # Create an inverted mask: allow adding everywhere EXCEPT on the reference object
                mask = ~reference_mask
            else:
                # No reference object found - allow adding anywhere (full white mask will be created in run_addit)
                mask = None
                
        elif op == "remove":
            # For REMOVE operations: use the mask of the object to remove (first target)
            if grounding_info["targets"] and grounding_info["targets"][0].get("masks") is not None:
                mask = grounding_info["targets"][0]["masks"][0]
        
        elif op in ["recolor", "replace"]:
            # For RECOLOR/REPLACE: use the target object mask
            if grounding_info["targets"] and grounding_info["targets"][0].get("masks") is not None:
                mask = grounding_info["targets"][0]["masks"][0]

        # Decide model
        if self.mode == "instructpix2pix" or (self.mode == "auto" and op in ["recolor", "replace", "move"]):
            pipe = self._get_instruct_pipe()
            prompt = plan.instruction
            return run_instructpix2pix(pipe, img, prompt)
        elif self.mode == "addit" or (self.mode == "auto" and op in ["add", "remove"]):
            pipe = self._get_addit_pipe()
            prompt = plan.instruction
            return run_addit(pipe, img, mask, prompt)
        else:
            return img  # no-op fallback
