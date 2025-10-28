from typing import Dict, Any

def verify_dummy(plan) -> Dict[str, Any]:
    # Pretend the VLM looked and agreed
    return {
        "qa": [
            {"q": f"Did we {plan.ops[0].type} '{plan.targets[0].name}'?", "a": "Yes"}
        ],
        "verdict": "pass"
    }
