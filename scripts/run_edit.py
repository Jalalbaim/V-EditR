import argparse
from pathlib import Path
from PIL import Image

import sys
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# --- Local imports ---
from src.utils.run_io import make_run_dir, save_image, save_json
from src.grounding.locate import locate_plan_aware
from src.validators.dummy import validate_dummy
from src.verifiers.dummy import verify_dummy
from src.editors.edit_manager import EditManager


# ---------------------- Helpers ----------------------

def _load_yaml(path: str) -> dict:
    """Load YAML configuration safely."""
    import yaml
    p = Path(path)
    if not p.exists():
        print(f"[WARN] Missing config: {p}")
        return {}
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _get_planner(parse_mode: str, planner_cfg: dict):
    """Select the parsing strategy for the planner."""
    if parse_mode == "llm":
        from src.planners.parse_v2 import parse as parse_plan_v2
        return lambda instr: parse_plan_v2(
            instr,
            assume_preserve_background=planner_cfg.get("assume_preserve_background", True),
            max_collateral=planner_cfg.get("max_collateral", 0.12),
        )
    else:
        from src.planners.parse import parse as parse_plan_v1
        return lambda instr: parse_plan_v1(instr)


# ---------------------- Main ----------------------

def main():
    ap = argparse.ArgumentParser(description="Run V-EditR editing pipeline (Phase 4)")
    ap.add_argument("--image", required=True, help="Path to the input image")
    ap.add_argument("--instruction", required=True, help="Text instruction (e.g. 'add a car next to the truck')")
    ap.add_argument("--tag", default="phase4", help="Name of the run folder")
    args = ap.parse_args()

    # --- Load configs ---
    cfg_planner = _load_yaml("configs/planner.yaml").get("planner", {"mode": "heuristic_v2"})
    cfg_ground = _load_yaml("configs/grounding.yaml")
    cfg_models = _load_yaml("configs/model.yaml")

    # --- Create run directory ---
    run_dir = make_run_dir(tag=args.tag)
    art = run_dir / "artifacts"

    # --- 1) Load image safely ---
    img_path = Path(args.image)
    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {img_path.resolve()}")
    try:
        img = Image.open(img_path).convert("RGB")
        print(f"[INFO] Loaded image: {img.size}, mode={img.mode}")
    except Exception as e:
        raise RuntimeError(f"Failed to open image: {img_path} — {e}")
    save_image(img, art / "input.jpg")

    # --- 2) Generate plan ---
    parse_plan = _get_planner(cfg_planner.get("mode", "heuristic_v2"), cfg_planner)
    plan = parse_plan(args.instruction)
    save_json(plan.model_dump(), art / "plan.json")
    print(f"[INFO] Plan generated for instruction: '{args.instruction}'")

    # --- 3) Ground objects (GroundingDINO + SAM) ---
    g_out = locate_plan_aware(img, plan, cfg_ground, save_debug_dir=art)
    save_json(
        {"meta": g_out.get("meta", {}), "targets": [
            {k: (v if k != "masks" else "<bool array>") for k, v in t.items()}
            for t in g_out["targets"]
        ]},
        art / "grounding.json",
    )
    print("[INFO] Grounding completed")

    # --- 4) Perform real editing (InstructPix2Pix / Add-It) ---
    editor = EditManager(cfg_models)
    edited = editor.apply_edit(img, plan, g_out)
    save_image(edited, art / "edited.jpg")
    print("[INFO] Image edited successfully")

    # --- 5) Validate and verify results (placeholders) ---
    report = validate_dummy(img, edited, plan)
    verdict = verify_dummy(plan)
    save_json(report, art / "validator.json")
    save_json(verdict, art / "verifier.json")

    # --- 6) Save summary ---
    summary = {
        "image": str(img_path.resolve()),
        "instruction": args.instruction,
        "run_dir": str(run_dir.resolve()),
        "artifacts": {
            "input": "artifacts/input.jpg",
            "plan": "artifacts/plan.json",
            "grounding": "artifacts/grounding.json",
            "edited": "artifacts/edited.jpg",
            "validator": "artifacts/validator.json",
            "verifier": "artifacts/verifier.json",
        },
        "configs": {
            "planner": cfg_planner,
            "grounding": cfg_ground,
            "models": cfg_models,
        },
    }
    save_json(summary, run_dir / "run_summary.json")

    print(f"\n Phase 4 editing done. See: {run_dir}\n")


# ---------------------- Entry point ----------------------

if __name__ == "__main__":
    main()
