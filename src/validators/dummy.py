from typing import Dict, Any
from PIL import Image

def validate_dummy(img_before: Image.Image, img_after: Image.Image, plan) -> Dict[str, Any]:
    # Dummy metrics just to show the structure
    requested_count = plan.targets[0].count if plan.targets else 1
    return {
        "requested": {"count": requested_count, "op": plan.ops[0].type if plan.ops else "unknown"},
        "achieved": {"count": requested_count, "ok": True},
        "collateral": {"lpips_proxy": 0.05},  # placeholder
        "status": "ok"
    }
