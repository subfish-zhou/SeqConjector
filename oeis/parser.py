
from typing import List, Tuple
from .program import Node, Program

OPS_SIG = {
    "A": (0,0),
    "SCALE": (1,1),
    "OFFSET": (1,1),
    "MAP_ABS": (0,1),
    "MAP_SGN": (0,1),
    "MAP_MOD": (1,1),
    "MAP_DIV": (1,1),
    "MAP_SQRT": (0,1),
    "SEQ_ADD": (0,2),
    "SEQ_SUB": (0,2),
    "SEQ_MUL": (0,2),
    "SEQ_DIV": (0,2),
    "SCAN_ADD": (0,1),
    "SCAN_MUL": (0,1),
    "DIFF_FWD": (1,1),
    "DIFF_BACK": (1,1),
    "CONV_FWD": (1,1),
    "CONV_BACK": (1,1),
    "REIDX_EVEN": (0,1),
    "REIDX_ODD": (0,1),
    "DROP1": (0,1),
    "DROP2": (0,1),
    "INSERT1": (0,1),
    "INSERT2": (0,1),
    "PRED_POS": (0,1),
    "PRED_NEG": (0,1),
    "PRED_IS_EVEN_N": (0,0),
}

class ParseError(Exception): pass

def parse_prefix(tokens: List[str]) -> Program:
    idx = 0
    toks = tokens
    def need() -> str:
        nonlocal idx
        if idx >= len(toks): raise ParseError("Unexpected end")
        s = toks[idx]; idx += 1; return s
    def parse_node():
        nonlocal idx
        op = need()
        if op not in OPS_SIG: raise ParseError(f"Unknown op {op}")
        args_arity, kids_arity = OPS_SIG[op]
        args = []
        
        # Standard args parsing
        if args_arity is not None:
            for _ in range(args_arity):
                try: args.append(int(need()))
                except: raise ParseError(f"{op} arg must be int")
        
        kids = []
        for _ in range(kids_arity):
            child, _ = parse_node()
            kids.append(child)
        return Node(op, args, kids), idx
    node, _ = parse_node()
    if idx != len(toks): raise ParseError(f"Unparsed tokens: {toks[idx:]}")
    return Program(node)
