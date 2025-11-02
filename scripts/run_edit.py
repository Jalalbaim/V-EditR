import argparse
from pathlib import Path
from PIL import Image

import sys
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.utils.run_io import make_run_dir, load_image, save_image, save_json
from src.grounding.locate import locate_plan_aware
from src.editors.dummy import apply_dummy
from src.validators.dummy import validate_dummy
from src.verifiers.dummy import verify_dummy

def _load_yaml(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        return {}
    import yaml
    return yaml.safe_load(open(p, "r", encoding="utf-8"))

def _get_planner(parse_mode: str, planner_cfg: dict):
    if parse_mode == "heuristic_v2":
        from src.planners.parse_v2 import parse as parse_plan_v2
        def _parse(instr):
            return parse_plan_v2(
                instr,
                assume_preserve_background=planner_cfg.get("assume_preserve_background", True),
                max_collateral=planner_cfg.get("max_collateral", 0.12),
            )
        return _parse
    else:
        from src.planners.parse import parse as parse_plan_v1
        return lambda instr: parse_plan_v1(instr)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="Path to input image")
    ap.add_argument("--instruction", required=True, help="Natural-language instruction")
    ap.add_argument("--tag", default="phase3")
    args = ap.parse_args()

    # Load configs
    cfg_planner = _load_yaml("configs/planner.yaml").get("planner", {"mode":"heuristic_v2"})
    cfg_ground  = _load_yaml("configs/grounding.yaml")

    run_dir = make_run_dir(tag=args.tag)
    art = run_dir / "artifacts"

    # 1) Load
    img = load_image(args.image)
    save_image(img, art / "input.jpg")

    # 2) Plan
    parse_plan = _get_planner(cfg_planner.get("mode", "heuristic_v2"), cfg_planner)
    plan = parse_plan(args.instruction)
    save_json(plan.model_dump(), art / "plan.json")

    # 3) Ground (real with fallback)
    g_out = locate_plan_aware(img, plan, cfg_ground, save_debug_dir=art)
    save_json({"meta": g_out.get("meta", {}), "targets": [
        {k:(v if k!='masks' else ('<bool array>' if v is not None else None)) for k,v in t.items()}
        for t in g_out["targets"]
    ]}, art / "grounding.json")

    # Choose a box/mask for the **primary** plan target (index 0)
    primary_name = plan.targets[0].name if plan.targets else (g_out["targets"][0]["name"] if g_out["targets"] else "object")
    tgt0 = next((t for t in g_out["targets"] if t["name"] == primary_name), g_out["targets"][0])
    box0 = tgt0["boxes"][0] if len(tgt0["boxes"]) else [10,10,100,100]

    # 4) Edit (still dummy editor for this phase)
    color = plan.targets[0].attributes[-1] if plan.targets and plan.targets[0].attributes else None
    op_type = plan.ops[0].type if plan.ops else "unknown"
    edited = apply_dummy(img, op_type, box0, primary_name, color=color)
    save_image(edited, art / "edited.jpg")

    # 5) Validate
    report = validate_dummy(img, edited, plan)
    save_json(report, art / "validator.json")

    # 6) Verify
    verdict = verify_dummy(plan)
    save_json(verdict, art / "verifier.json")

    # 7) Index
    summary = {
        "image": str(Path(args.image).resolve()),
        "instruction": args.instruction,
        "run_dir": str(run_dir.resolve()),
        "artifacts": {
            "input": "artifacts/input.jpg",
            "plan": "artifacts/plan.json",
            "grounding": "artifacts/grounding.json",
            "boxes_*": "artifacts/boxes_*.jpg",
            "masks_*": "artifacts/masks_*.jpg",
            "edited": "artifacts/edited.jpg",
            "validator": "artifacts/validator.json",
            "verifier": "artifacts/verifier.json"
        },
        "configs": {"planner": cfg_planner, "grounding": cfg_ground}
    }
    save_json(summary, run_dir / "run_summary.json")
    print(f"\n✅ Phase 3 grounding done. See: {run_dir}\n")

if __name__ == "__main__":
    main()
