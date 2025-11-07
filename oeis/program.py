
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class Node:
    op: str
    args: List[int]=None
    kids: List["Node"]=None
    def __init__(self, op, args=None, kids=None):
        self.op = op
        self.args = list(args) if args is not None else []
        self.kids = list(kids) if kids is not None else []
    def to_tokens(self) -> List[str]:
        toks = [self.op]
        for a in self.args:
            toks.append(str(a))
        for ch in self.kids:
            toks += ch.to_tokens()
        return toks
    def __repr__(self):
        return " ".join(self.to_tokens())

@dataclass
class Program:
    root: Node
    def to_tokens(self) -> List[str]:
        return self.root.to_tokens()
