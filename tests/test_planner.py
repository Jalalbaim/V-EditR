# tests/test_planner_v2.py
from src.planners.parse_v2 import parse

def test_counts_and_colors():
    instr = "Add two red cars next to the blue truck"
    plan = parse(instr)
    assert plan.ops[0].type == "add"
    assert plan.targets[0].name in {"car","object"}
    assert any(t.count >= 2 for t in plan.targets)
    assert "red" in (plan.targets[0].attributes + plan.targets[-1].attributes)

def test_recolor():
    instr = "Make the jacket black"
    plan = parse(instr)
    assert plan.ops[0].type == "recolor"
    assert plan.targets[0].name in {"jacket","coat"}
