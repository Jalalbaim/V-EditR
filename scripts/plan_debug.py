import argparse
import json
import yaml
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _load_planner_cfg():
    cfg_path = Path("configs/planner.yaml")
    if cfg_path.exists():
        return yaml.safe_load(open(cfg_path, "r", encoding="utf-8"))
    return {"planner":{"mode":"heuristic_v2","assume_preserve_background":True,"max_collateral":0.12}}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instruction", required=True)
    args = ap.parse_args()

    cfg = _load_planner_cfg()
    mode = cfg.get("planner",{}).get("mode","heuristic_v2")
    if mode == "heuristic_v2":
        from src.planners.parse_v2 import parse as parse_plan
        p = cfg.get("planner",{})
        plan = parse_plan(args.instruction,
                          assume_preserve_background=p.get("assume_preserve_background", True),
                          max_collateral=p.get("max_collateral", 0.12))
    else:
        from src.planners.parse import parse as parse_plan
        plan = parse_plan(args.instruction)

    print(json.dumps(plan.model_dump(), indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
