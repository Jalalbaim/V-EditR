# src/planners/ontology.py
from typing import Dict, List

COLORS: List[str] = [
    "red","green","blue","yellow","black","white","silver","gray","grey",
    "brown","purple","orange","pink","beige","gold","cyan","magenta"
]

RELATION_ALIASES: Dict[str, str] = {
    "left of": "left_of",
    "on the left of": "left_of",
    "to the left of": "left_of",
    "right of": "right_of",
    "on the right of": "right_of",
    "to the right of": "right_of",
    "next to": "next_to",
    "near": "next_to",
    "beside": "next_to",
    "behind": "behind",
    "in front of": "in_front_of",
    "front of": "in_front_of",
    "above": "above",
    "below": "below"
}

# Minimal, extend as you go. Keys are canonical names.
OBJECT_SYNONYMS: Dict[str, List[str]] = {
    "car": ["car","cars","vehicle","auto","sedan","hatchback","sports car"],
    "truck": ["truck","lorry","pickup","van"],
    "bus": ["bus","coach"],
    "bicycle": ["bicycle","bike","cycle"],
    "motorbike": ["motorbike","motorcycle","scooter"],
    "cone": ["cone","traffic cone","pylon"],
    "person": ["person","man","woman","boy","girl","people","guy","lady","kid","child"],
    "phone": ["phone","smartphone","cellphone","mobile"],
    "watch": ["watch","wristwatch"],
    "jacket": ["jacket","coat","blazer","parka","hoodie"],
    "shirt": ["shirt","t-shirt","tee"],
    "bag": ["bag","backpack","handbag","purse","pack"],
    "dog": ["dog","puppy","doggo"],
    "cat": ["cat","kitten"],
    "tree": ["tree"],
    "building": ["building","house","home"],
    "truck_trailer": ["trailer","semi","semi-truck"],
}
