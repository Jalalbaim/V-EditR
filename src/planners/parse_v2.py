# src/planners/parse_v2.py
import re
from typing import List, Dict, Any, Tuple, Optional
from .schema import Plan, Target, Relation, Operation, Constraints
from .ontology import COLORS, RELATION_ALIASES, OBJECT_SYNONYMS

WORD2NUM = {
    "zero":0,"one":1,"two":2,"three":3,"four":4,"five":5,"six":6,"seven":7,"eight":8,"nine":9,"ten":10,
    "eleven":11,"twelve":12,"thirteen":13,"fourteen":14,"fifteen":15,"twenty":20,"thirty":30
}

OP_ALIASES = {
    "add": ["add","insert","place","put","create","spawn"],
    "remove": ["remove","delete","erase","take away"],
    "recolor": ["recolor","color","repaint","make","turn","change color","paint"],
    "replace": ["replace","swap"],
    "move": ["move","relocate","shift","put ... to"]
}

# ---------------------------
# Helpers
# ---------------------------

def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().strip())

def _noun_regex_from_ontology() -> str:
    alts = []
    for syns in OBJECT_SYNONYMS.values():
        for s in syns:
            alts.append(re.escape(s))
            if not s.endswith("s"):
                alts.append(re.escape(s + "s"))
    alts = sorted(set(alts), key=len, reverse=True)
    return r"(?:%s)" % "|".join(alts)

NOUN_RE = _noun_regex_from_ontology()

def _canonicalize(noun: str) -> str:
    n = noun.rstrip("s")
    for cano, syns in OBJECT_SYNONYMS.items():
        for s in syns:
            if n == s or noun == s or n == s.rstrip("s"):
                return cano
    return n

def _find_operation(text: str) -> Tuple[str, float, Optional[re.Match]]:
    for op, keys in [("replace", OP_ALIASES["replace"]),
                     ("add", OP_ALIASES["add"]),
                     ("remove", OP_ALIASES["remove"]),
                     ("recolor", OP_ALIASES["recolor"]),
                     ("move", OP_ALIASES["move"])]:
        for k in keys:
            m = re.search(rf"\b{k}\b", text)
            if m:
                return op, 0.9, m
    # weak guess (recolor) if pattern "make ... <color>"
    if re.search(r"\bmake .* (" + "|".join(COLORS) + r")\b", text):
        return "recolor", 0.6, None
    return "unknown", 0.3, None

def _num_from_token(tok: str) -> Optional[int]:
    if tok.isdigit():
        return int(tok)
    return WORD2NUM.get(tok)

def _colors_in_span(span: str) -> List[str]:
    found = []
    for c in COLORS:
        if re.search(rf"\b{c}\b", span):
            found.append(c)
    return found

# ---------------------------
# Parsing building blocks
# ---------------------------

def _extract_counts_bound_to_nouns(text: str) -> Dict[str,int]:
    """
    Extract counts tied to specific noun phrases:
      - 'two red cars' -> car:2
      - 'add 3 cones'  -> cone:3
    """
    counts: Dict[str, int] = {}
    # Pattern: (number) + optional adjectives + noun
    pattern = rf"\b((?:\d+)|(?:{'|'.join(map(re.escape, WORD2NUM.keys()))}))\s+(?:\w+\s+)*?({NOUN_RE})\b"
    for m in re.finditer(pattern, text):
        num_tok, noun_tok = m.group(1), m.group(2)
        n = _num_from_token(num_tok)
        if n is None:
            continue
        cano = _canonicalize(noun_tok)
        counts[cano] = max(counts.get(cano, 0), n)

    # Generic: "add two" without mentioning a noun immediately after
    generic = None
    for m in re.finditer(rf"\b(add|insert|place|put|create|spawn)\s+((?:\d+)|(?:{'|'.join(map(re.escape, WORD2NUM.keys()))}))\b", text):
        n = _num_from_token(m.group(2))
        if n is not None:
            generic = n
    if generic is not None and not counts:
        counts["__generic__"] = generic
    return counts

def _extract_relations_clean(text: str) -> List[Relation]:
    """
    Capture relations while ignoring articles/adjectives, e.g.:
      'cars next to the blue truck' -> subj=car, rel=next_to, obj=truck
    """
    rels: List[Relation] = []
    for phrase, rel_key in RELATION_ALIASES.items():
        # Allow: <noun-or-phrase> <phrase> (the|a|an)? <adjs>* <noun>
        pattern = rf"({NOUN_RE})(?:\s+\w+)*\s+{re.escape(phrase)}\s+(?:the|a|an)?\s*(?:\w+\s+)*({NOUN_RE})\b"
        for m in re.finditer(pattern, text):
            subj_raw, obj_raw = m.group(1), m.group(2)
            subj = _canonicalize(subj_raw)
            obj  = _canonicalize(obj_raw)
            rels.append(Relation(subj=subj, rel=rel_key, obj=obj))
    return rels

def _primary_targets_near_op(text: str, op_match: Optional[re.Match], counts: Dict[str,int]) -> List[Target]:
    """
    Identify primary target(s) for the operation by looking near the operation token.
    For 'add two red cars next to the blue truck':
      primary -> car (count=2, color=red)
      secondary (context) -> truck (count=1, color=blue)
    """
    tokens = text.split()
    # default window around op
    start_idx = 0 if op_match is None else max(0, op_match.start() - 40)
    end_idx   = len(text) if op_match is None else min(len(text), op_match.end() + 80)
    window = text[start_idx:end_idx]

    # Primary nouns near op
    primary_nouns = [ _canonicalize(n) for n in re.findall(NOUN_RE, window) ]
    # keep order of first appearance
    seen = set()
    ordered = []
    for n in primary_nouns:
        if n not in seen:
            seen.add(n)
            ordered.append(n)

    # Heuristic: first noun after op is primary
    targets: List[Target] = []
    if ordered:
        primary = ordered[0]
        # try to attach count and color to primary from local window chunk around the noun
        noun_win = re.search(rf"(?:\b\w+\b\s+){{0,4}}\b{re.escape(primary)}s?\b(?:\s+\b\w+\b){{0,4}}", window)
        attrs = _colors_in_span(noun_win.group(0)) if noun_win else []
        cnt = counts.get(primary, counts.get("__generic__", 1))
        targets.append(Target(name=primary, attributes=attrs, count=cnt))

        # Add a secondary contextual object if present (e.g., truck), count defaults to 1
        if len(ordered) > 1:
            secondary = ordered[1]
            noun2_win = re.search(rf"(?:\b\w+\b\s+){{0,4}}\b{re.escape(secondary)}s?\b(?:\s+\b\w+\b){{0,4}}", window)
            attrs2 = _colors_in_span(noun2_win.group(0)) if noun2_win else []
            cnt2 = counts.get(secondary, 1)
            targets.append(Target(name=secondary, attributes=attrs2, count=cnt2))
    else:
        # fallback
        generic_cnt = counts.get("__generic__", 1)
        targets = [Target(name="object", attributes=_colors_in_span(window), count=generic_cnt)]

    return targets

def _build_operations_for_primary(op_type: str, targets: List[Target], text: str) -> List[Operation]:
    """
    Create operations ONLY for the primary target (first target).
    For recolor, use the most specific color available (target-local > global).
    """
    if not targets:
        return [Operation(type=op_type, target="object", params={})]

    primary = targets[0]
    params: Dict[str, Any] = {}
    if op_type == "recolor":
        local_color = primary.attributes[-1] if primary.attributes else None
        global_colors = _colors_in_span(text)
        if local_color:
            params["color"] = local_color
        elif global_colors:
            params["color"] = global_colors[-1]

    return [Operation(type=op_type, target=primary.name, params=params)]

# ---------------------------
# Public parse()
# ---------------------------

def parse(instruction: str, assume_preserve_background=True, max_collateral=0.12) -> Plan:
    text = _norm(instruction)

    # 1) Operation + anchor
    op_type, op_conf, op_match = _find_operation(text)

    # 2) Counts tied to canonical nouns
    counts = _extract_counts_bound_to_nouns(text)

    # 3) Relations with article/adj cleanup
    relations = _extract_relations_clean(text)

    # 4) Targets near the op (primary first)
    targets = _primary_targets_near_op(text, op_match, counts)

    # 5) Build operation(s) only for primary target
    ops = _build_operations_for_primary(op_type, targets, text)

    # 6) Constraints
    constraints = Constraints(
        preserve_background=assume_preserve_background,
        max_collateral=max_collateral
    )

    plan = Plan(
        instruction=instruction,
        targets=targets,
        relations=relations,
        ops=ops,
        constraints=constraints
    )

    # annotate confidence hint
    if plan.ops:
        plan.ops[0].params["planner_confidence"] = round(op_conf, 2)

    return plan
