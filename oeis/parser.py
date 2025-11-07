
from typing import List, Tuple
from .program import Node, Program

OPS_SIG = {
    "A": (0,0),
    "SCALE": (1,1),
    "OFFSET": (1,1),
    "MAP_ABS": (0,1),
    "MAP_SGN": (0,1),
    "MAP_MOD": (1,1),
    "MAP_TAU": (0,1),
    "MAP_SIGMA": (0,1),
    "MAP_PHI": (0,1),
    "MAP_MU": (0,1),
    "MAP_OMEGA": (0,1),
    "MAP_BIGOMEGA": (0,1),
    "SCAN_ADD": (0,1),
    "SCAN_MUL": (0,1),
    "CUMMAX": (0,1),
    "CUMMIN": (0,1),
    "DIFF_FWD": (1,1),
    "DIFF_BACK": (1,1),
    "ZIP": (3,1),
    "CONV": (1,1),
    "POLY": (1,1),
    "REIDX": (2,1),
    "SUBSAMPLE": (1,1),
    "REPEAT": (1,1),
    "INDEXBY": (0,1),
    "BINOM": (0,1),
    "IBINOM": (0,1),
    "EULER": (0,1),
    "DROP": (1,1),
    "INSERT1": (1,1),
    "INSERT2": (1,1),
    "PRED_POS": (0,1),
    "PRED_NEG": (0,1),
    "PRED_IS_EVEN_N": (0,0),
    "PRED_EQ_CONST": (1,1),
    "PRED_GT_CONST": (1,1),
    "PRED_LT_CONST": (1,1),
    "PRED_NOT": (0,1),
    "PRED_AND": (0,2),
    "PRED_OR": (0,2),
    "COND": (0,3),
}

BINOPS = {"ADD","SUB","MUL","MIN","MAX"}

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
        if op == "ZIP":
            bop = need()
            if bop not in BINOPS: raise ParseError("ZIP first arg must be binop")
            args.append(bop)
            try:
                args.append(int(need())); args.append(int(need()))
            except: raise ParseError("ZIP k1/k2 must be int")
        elif op == "CONV":
            Ls = need()
            try:
                L = int(Ls)
            except:
                raise ParseError("CONV length int")
            if not (1 <= L <= 5): raise ParseError("CONV length 1..5")
            args.append(L)
            for _ in range(L):
                try: args.append(int(need()))
                except: raise ParseError("CONV weights int")
        elif op == "POLY":
            Ds = need()
            try: D = int(Ds)
            except: raise ParseError("POLY degree int")
            if not (0 <= D <= 4): raise ParseError("POLY degree 0..4")
            args.append(D)
            for _ in range(D+1):
                try: args.append(int(need()))
                except: raise ParseError("POLY coeffs int")
        else:
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
