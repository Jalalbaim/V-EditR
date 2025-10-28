import re
from typing import List
from .schema import Plan, Target, Relation, Operation, Constraints

_COLORS = ["red","green","blue","yellow","black","white","silver","gray","grey","brown","purple","orange","pink"]
_REL_MAP = {
    "left of": "left_of",
    "right of": "right_of",
    "next to": "next_to",
    "behind": "behind",
    "in front of": "in_front_of",
}

_WORD2NUM = {"one":1,"two":2,"three":3,"four":4,"five":5,"six":6,"seven":7,"eight":8,"nine":9,"ten":10}

def _extract_count(text: str) -> int:
    m = re.search(r"\b(\d+)\b", text)
    if m: return int(m.group(1))
    for w,n in _WORD2NUM.items():
        if re.search(rf"\b{w}\b", text): return n
    return 1

def _extract_color(text: str):
    for c in _COLORS:
        if re.search(rf"\b{c}\b", text): return c
    return None

def _extract_operation(text: str) -> str:
    if re.search(r"\b(add|insert|place)\b", text): return "add"
    if re.search(r"\b(remove|delete|erase)\b", text): return "remove"
    if re.search(r"\b(recolor|color|make .* (red|blue|green|black|white|yellow|silver|grey|gray))\b", text): return "recolor"
    return "unknown"

def _extract_relations(text: str) -> List[Relation]:
    rels = []
    for k,v in _REL_MAP.items():
        if k in text:
            # naive pattern: X <rel> Y  (e.g., "car left of truck")
            m = re.search(rf"\b([a-zA-Z]+)\s+{re.escape(k)}\s+([a-zA-Z]+)\b", text)
            if m:
                rels.append(Relation(subj=m.group(1), rel=v, obj=m.group(2)))
    return rels

def _guess_target_noun(text: str) -> str:
    # crude: last noun-like word (letters only)
    tokens = re.findall(r"[a-zA-Z]+", text.lower())
    if not tokens: return "object"
    # avoid common verbs
    blacklist = {"add","insert","place","remove","delete","make","color","recolor","change","the","a","an","to","of","in","on","next","left","right","behind","front"}
    for t in reversed(tokens):
        if t not in blacklist:
            return t
    return "object"

def parse(instruction: str) -> Plan:
    text = instruction.lower().strip()
    op = _extract_operation(text)
    count = _extract_count(text)
    color = _extract_color(text)
    relations = _extract_relations(text)

    target_name = _guess_target_noun(text)
    attributes = [color] if color else []

    targets = [Target(name=target_name, attributes=attributes, count=count)]
    ops = [Operation(type=op, target=target_name, params={"color": color} if color else {})]
    plan = Plan(
        instruction=instruction,
        targets=targets,
        relations=relations,
        ops=ops,
        constraints=Constraints()
    )
    return plan
