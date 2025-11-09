import argparse
from pathlib import Path
from PIL import Image

import sys
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.utils.run_io import make_run_dir, load_image, save_image, save_json
from src.grounding.locate import locate_plan_aware
from src.validators.dummy import validate_dummy
from src.verifiers.dummy import verify_dummy
from src.editors.edit_manager import EditManager

def _load_yaml(path: str) -> dict:
    import yaml
    p = Path(path)
    if not p.exists():
        return {}
    return yaml.safe_load(open(p, "r", encoding="utf-8"))

def _get_planner(parse_mode: str, planner_cfg: dict):
    if parse_mode == "heuristic_v2":
        from src.planners.parse_v2 import parse as parse_plan_v2
        return lambda instr: parse_plan_v2(instr,
            assume_preserve_background=planner_cfg.get("assume_preserve_background", True),
            max_collateral=planner_cfg.get("max_collateral", 0.12))
    else:
        from src.planners.parse import parse as parse_plan_v1
        return lambda instr: parse_plan_v1(instr)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--instruction", required=True)
    ap.add_argument("--tag", default="phase4")
    args = ap.parse_args()

    # Load configs
    cfg_planner = _load_yaml("configs/planner.yaml").get("planner", {"mode": "heuristic_v2"})
    cfg_ground  = _load_yaml("configs/grounding.yaml")
    cfg_models  = _load_yaml("configs/model.yaml")

    run_dir = make_run_dir(tag=args.tag)
    art = run_dir / "artifacts"

    # 1) Load image
    img = load_image(args.image)
    save_image(img, art / "input.jpg")

    # 2) Plan
    parse_plan = _get_planner(cfg_planner.get("mode", "heuristic_v2"), cfg_planner)
    plan = parse_plan(args.instruction)
    save_json(plan.model_dump(), art / "plan.json")

    # 3) Ground
    g_out = locate_plan_aware(img, plan, cfg_ground, save_debug_dir=art)
    save_json({"meta": g_out.get("meta", {}), "targets": [
        {k:(v if k!='masks' else '<bool array>') for k,v in t.items()}
        for t in g_out["targets"]
    ]}, art / "grounding.json")

    # 4) Edit (real)
    editor = EditManager(cfg_models)
    edited = editor.apply_edit(img, plan, g_out)
    save_image(edited, art / "edited.jpg")

    # 5) Validate + Verify (placeholders)
    report = validate_dummy(img, edited, plan)
    verdict = verify_dummy(plan)
    save_json(report, art / "validator.json")
    save_json(verdict, art / "verifier.json")

    # 6) Summary
    summary = {
        "image": str(Path(args.image).resolve()),
        "instruction": args.instruction,
        "run_dir": str(run_dir.resolve()),
        "artifacts": {
            "input": "artifacts/input.jpg",
            "plan": "artifacts/plan.json",
            "grounding": "artifacts/grounding.json",
            "edited": "artifacts/edited.jpg",
            "validator": "artifacts/validator.json",
            "verifier": "artifacts/verifier.json"
        },
        "configs": {"planner": cfg_planner, "grounding": cfg_ground, "models": cfg_models}
    }
    save_json(summary, run_dir / "run_summary.json")
    print(f"\n Editing done. See: {run_dir}\n")

if __name__ == "__main__":
    main()
