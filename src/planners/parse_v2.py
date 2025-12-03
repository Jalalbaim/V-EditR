
from __future__ import annotations
import json
import requests
from typing import Dict, Any
from .schema import Plan

OLLAMA_MODEL = "gemma3:4b"

PROMPT_TEMPLATE = """
You are an expert visual scene planner. 
Your job is to convert the user's instruction into a structured JSON plan for image editing.

STRICT RULES:
- Output ONLY valid JSON.
- No commentary, no backticks, no natural language.
- Follow EXACTLY this schema:

{{
  "instruction": "<original instruction>",
  "targets": [
    {{ "name": "object", "attributes": ["adj1", "adj2"], "count": 1 }}
  ],
  "relations": [
    {{ "subj": "object1", "rel": "relation", "obj": "object2" }}
  ],
  "ops": [
    {{ "type": "add|remove|recolor|replace", "target": "object", "params": {{}} }}
  ],
  "constraints": {{
    "preserve_background": true,
    "max_collateral": 0.10
  }}
}}

IMPORTANT CONSTRAINTS:
- "type" in ops MUST be one of: "add", "remove", "recolor", "replace" (NOT "move", "write", or anything else)
- "rel" in relations MUST be one of: "left_of", "right_of", "next_to", "behind", "in_front_of"
- If the instruction involves writing text, use type="add" with params containing the text
- Relations are optional - if no spatial relation is specified, use an empty relations array []
- For color changes, use type="recolor" with params={{"color": "colorname"}}

Convert the following instruction:

Instruction: "{instruction}"
"""
# ------------------------- API Call -----------------------------------
def _query_ollama(prompt: str) -> str:
    """
    Sends a prompt to Ollama and returns the raw text output.
    Ollama must be running locally: http://localhost:11434
    """
    url = "http://localhost:11434/api/generate"

    response = requests.post(
        url,
        json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
    )

    if response.status_code != 200:
        raise RuntimeError(f"Ollama error {response.status_code}: {response.text}")

    data = response.json()
    return data.get("response", "").strip()


def _sanitize_plan_dict(plan_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Cleans up plan dictionary to ensure it matches the expected schema.
    Filters out invalid relations and operation types.
    """
    # Valid values according to schema
    VALID_OPS = {"add", "remove", "recolor", "replace", "unknown"}
    VALID_RELS = {"left_of", "right_of", "next_to", "behind", "in_front_of"}
    
    # Clean operations
    if "ops" in plan_dict:
        cleaned_ops = []
        for op in plan_dict["ops"]:
            if op.get("type") not in VALID_OPS:
                # Try to map common variations
                op_type = op.get("type", "").lower()
                if "color" in op_type:
                    op["type"] = "recolor"
                elif "write" in op_type or "text" in op_type:
                    op["type"] = "add"
                elif "move" in op_type or "relocate" in op_type:
                    op["type"] = "replace"
                else:
                    op["type"] = "unknown"
            cleaned_ops.append(op)
        plan_dict["ops"] = cleaned_ops
    
    # Clean relations - filter out invalid ones
    if "relations" in plan_dict:
        cleaned_rels = []
        for rel in plan_dict["relations"]:
            if rel.get("rel") in VALID_RELS:
                cleaned_rels.append(rel)
        plan_dict["relations"] = cleaned_rels
    
    return plan_dict


def parse(instruction: str, assume_preserve_background: bool = True, max_collateral: float = 0.12) -> Plan:
    """
    Generates a structured editing plan from natural-language instruction using LLM.
    Returns a Plan object compatible with the rest of the pipeline.
    """
    prompt = PROMPT_TEMPLATE.format(instruction=instruction)

    # Call Ollama
    raw = _query_ollama(prompt)

    try:
        plan_dict = json.loads(raw)
    except Exception:
        cleaned = raw.strip()
        cleaned = cleaned[cleaned.find("{") : cleaned.rfind("}") + 1]
        plan_dict = json.loads(cleaned)

    # Ensure instruction is set
    plan_dict["instruction"] = instruction
    
    # Set constraints if not present or override with parameters
    if "constraints" not in plan_dict:
        plan_dict["constraints"] = {}
    plan_dict["constraints"]["preserve_background"] = assume_preserve_background
    plan_dict["constraints"]["max_collateral"] = max_collateral
    
    # Sanitize the plan dictionary to match schema
    plan_dict = _sanitize_plan_dict(plan_dict)

    # Convert dictionary to Plan object
    plan = Plan(**plan_dict)
    
    return plan


def generate_plan(instruction: str) -> Dict[str, Any]:
    """
    Legacy function for backward compatibility.
    Generates a structured editing plan and returns it as a dictionary.
    """
    plan = parse(instruction)
    return plan.model_dump()


if __name__ == "__main__":
    test = "Add two red cars next to the blue truck"
    result = generate_plan(test)
    print(json.dumps(result, indent=2))