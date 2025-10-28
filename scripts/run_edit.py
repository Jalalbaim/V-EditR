import argparse
from pathlib import Path
from PIL import Image

import sys
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.utils.run_io import make_run_dir, load_image, save_image, save_json
from src.planners.parse import parse as parse_plan
from src.grounding.locate import locate_dummy, overlay_box
from src.editors.dummy import apply_dummy
from src.validators.dummy import validate_dummy
from src.verifiers.dummy import verify_dummy

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="Path to input image")
    ap.add_argument("--instruction", required=True, help="Natural-language instruction")
    ap.add_argument("--tag", default="phase1")
    args = ap.parse_args()

    run_dir = make_run_dir(tag=args.tag)
    art = run_dir / "artifacts"

    # 1) Load
    img = load_image(args.image)
    save_image(img, art / "input.jpg")

    # 2) Plan
    plan = parse_plan(args.instruction)
    save_json(plan.model_dump(), art / "plan.json")

    # 3) Ground (dummy)
    g = locate_dummy(img, plan.targets[0].name if plan.targets else "object")
    boxed = overlay_box(img, g["box"])
    save_image(boxed, art / "grounding_preview.jpg")

    # 4) Edit (dummy)
    color = plan.targets[0].attributes[0] if plan.targets and plan.targets[0].attributes else None
    edited = apply_dummy(img, plan.ops[0].type if plan.ops else "unknown", g["box"], plan.targets[0].name, color=color)
    save_image(edited, art / "edited.jpg")

    # 5) Validate (dummy)
    report = validate_dummy(img, edited, plan)
    save_json(report, art / "validator.json")

    # 6) Verify (dummy)
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
            "grounding_preview": "artifacts/grounding_preview.jpg",
            "edited": "artifacts/edited.jpg",
            "validator": "artifacts/validator.json",
            "verifier": "artifacts/verifier.json"
        }
    }
    save_json(summary, run_dir / "run_summary.json")
    print(f"\n Done. See: {run_dir}\n")

if __name__ == "__main__":
    main()
