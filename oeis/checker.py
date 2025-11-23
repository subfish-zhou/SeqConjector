
from dataclasses import dataclass, replace
from typing import List, Optional
import math
from .parser import parse_prefix
from .interpreter import Interpreter, ExecConfig

@dataclass
class CheckReport:
    ok: bool
    stageA_ok: bool
    stageB_ok: bool
    reason: str
    prog_tokens: List[str]
    used_cost: int
    budget_total: int
    mismatch_pos: Optional[int]=None

def check_program_on_pair(tokens: List[str], A_full: List[int], B_full: List[int], n_in: int, n_chk: int) -> CheckReport:
    if n_in + n_chk > min(len(A_full), len(B_full)):
        return CheckReport(False, False, False, f"prefix+chk={n_in+n_chk} exceeds min length {min(len(A_full), len(B_full))}", tokens, 0, 0, None)
    inter_loose = Interpreter(ExecConfig(strict=False, t0=10, t_step=3))
    inter_strict = Interpreter(ExecConfig(strict=True, t0=10, t_step=3))
    try:
        P = parse_prefix(tokens)
    except Exception as e:
        return CheckReport(False, False, False, f"parse_error:{e}", tokens, 0, 0, None)
    # 检查程序是否包含 A
    if not P.contains_A():
        return CheckReport(False, False, False, "program_must_contain_A", tokens, 0, 0, None)
    # Stage A
    A_vis = A_full[:n_in]; B_vis = B_full[:n_in]
    rA = inter_loose.execute(P, A_vis, B_vis)
    if (not rA.ok) or (rA.seq is None) or (len(rA.seq) < n_in):
        return CheckReport(False, False, False, f"stageA_exec_fail:{rA.reason}", tokens, rA.used_cost, rA.budget_total, None)
    for i,(x,y) in enumerate(zip(rA.seq[:n_in], B_vis)):
        if x != y:
            return CheckReport(False, False, False, f"stageA_mismatch@{i}:{x}!={y}", tokens, rA.used_cost, rA.budget_total, i)
    # Stage B
    N2 = n_in + n_chk
    rB = inter_strict.execute(P, A_full[:N2], B_full[:N2])
    if (not rB.ok) or (rB.seq is None) or (len(rB.seq) < N2):
        return CheckReport(False, True, False, f"stageB_exec_fail:{rB.reason}", tokens, rB.used_cost, rB.budget_total, None)
    for i,(x,y) in enumerate(zip(rB.seq[:N2], B_full[:N2])):
        if x != y:
            return CheckReport(False, True, False, f"stageB_mismatch@{i}:{x}!={y}", tokens, rB.used_cost, rB.budget_total, i)
    return CheckReport(True, True, True, "ok", tokens, rB.used_cost, rB.budget_total, None)

def check_program_moonshine(tokens: List[str], A_full: List[int], B_full: List[int], n_in: int, n_chk: int,
                            k_strict:int=3, tau0:float=2e-3, tau1:float=1e-3, use_log:bool=True) -> CheckReport:
    if n_in + n_chk > min(len(A_full), len(B_full)):
        return CheckReport(False, False, False, f"prefix+chk={n_in+n_chk} exceeds min length {min(len(A_full), len(B_full))}", tokens, 0, 0, None)
    N2 = n_in + n_chk
    try:
        P = parse_prefix(tokens)
    except Exception as e:
        return CheckReport(False, False, False, f"parse_error:{e}", tokens, 0, 0, None)
    # 检查程序是否包含 A
    if not P.contains_A():
        return CheckReport(False, False, False, "program_must_contain_A", tokens, 0, 0, None)
    inter = Interpreter(ExecConfig(strict=True, t0=10, t_step=3))
    r = inter.execute(P, A_full[:N2], B_full[:N2])
    if (not r.ok) or (r.seq is None) or (len(r.seq) < N2):
        return CheckReport(False, False, False, f"exec_fail:{r.reason}", tokens, r.used_cost, r.budget_total, None)
    Bh = r.seq[:N2]; Bt = B_full[:N2]
    K = min(k_strict, N2)
    for i in range(K):
        if Bh[i] != Bt[i]:
            return CheckReport(False, False, False, f"head_strict_fail@{i}:{Bh[i]}!={Bt[i]}", tokens, r.used_cost, r.budget_total, i)
    ok = True
    errs = []
    for i in range(K, N2):
        if Bt[i] == 0: 
            continue
        ratio = Bh[i] / Bt[i]
        e = abs(math.log(max(ratio, 1e-18))) if use_log else abs(1.0 - ratio)
        thr = tau0 + tau1 * i
        errs.append(e)
        if e > thr:
            ok = False
            break
    if ok:
        return CheckReport(True, True, False, "moonshine_accept", tokens, r.used_cost, r.budget_total, None)
    return CheckReport(False, False, False, f"moonshine_tail_exceed:{max(errs) if errs else 0.0}", tokens, r.used_cost, r.budget_total, None)
