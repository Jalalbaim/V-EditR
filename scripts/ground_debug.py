# scripts/ground_debug.py
import argparse, json
from pathlib import Path
from src.utils.run_io import make_run_dir, load_image, save_image
from src.grounding.locate import locate_plan_aware

def _load_yaml(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        return {}
    import yaml
    return yaml.safe_load(open(p, "r", encoding="utf-8"))

def _load_plan_json(path: str):
    import json
    return json.load(open(path, "r", encoding="utf-8"))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--plan", required=True, help="path to artifacts/plan.json")
    ap.add_argument("--tag", default="ground_dbg")
    args = ap.parse_args()

    run_dir = make_run_dir(tag=args.tag)
    art = run_dir / "artifacts"

    img = load_image(args.image)
    save_image(img, art / "input.jpg")

    plan = _load_plan_json(args.plan)
    # convert dict → Plan (lazy import to avoid pydantic here)
    from src.planners.schema import Plan
    plan = Plan(**plan)

    g_cfg = _load_yaml("configs/grounding.yaml")
    out = locate_plan_aware(img, plan, g_cfg, save_debug_dir=art)

    (art / "grounding.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"\nSee artifacts in: {run_dir}\n")

if __name__ == "__main__":
    main()
