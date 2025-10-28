from pydantic import BaseModel, Field
from typing import List, Optional, Literal, Dict, Any

class Target(BaseModel):
    name: str
    attributes: List[str] = []
    count: int = 1

class Relation(BaseModel):
    subj: str
    rel: Literal["left_of", "right_of", "next_to", "behind", "in_front_of"]
    obj: str

class Operation(BaseModel):
    type: Literal["add", "remove", "recolor", "unknown"] = "unknown"
    target: str
    params: Dict[str, Any] = {}

class Constraints(BaseModel):
    preserve_background: bool = True
    max_collateral: float = 0.15

class Plan(BaseModel):
    instruction: str
    targets: List[Target] = Field(default_factory=list)
    relations: List[Relation] = Field(default_factory=list)
    ops: List[Operation] = Field(default_factory=list)
    constraints: Constraints = Constraints()
