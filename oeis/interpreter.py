
from dataclasses import dataclass
from typing import List, Optional

from .program import Node, Program

@dataclass
class ExecConfig:
    strict: bool = True
    t0: int = 10
    t_step: int = 3

@dataclass
class ExecResult:
    ok: bool
    seq: Optional[List[int]]
    used_cost: int
    budget_total: int
    reason: str = ""

class Budget:
    def __init__(self, t0: int, t_step: int):
        self.t0=t0; self.t_step=t_step
        self.allowed=0; self.used=0
    def add_for(self, n:int):
        self.allowed += self.t0 + self.t_step*n
    def charge(self, c:int):
        self.used += c
        if self.used > self.allowed:
            raise RuntimeError("budget_exceeded")

class Interpreter:
    def __init__(self, cfg: ExecConfig=ExecConfig()):
        self.cfg=cfg
        self.ops = {
            "A": self._op_A,
            "SCALE": self._op_SCALE, "OFFSET": self._op_OFFSET,
            "MAP_ABS": self._op_MAP_ABS, "MAP_SGN": self._op_MAP_SGN,
            "MAP_MOD": self._op_MAP_MOD, "MAP_DIV": self._op_MAP_DIV, "MAP_SQRT": self._op_MAP_SQRT,
            "SEQ_ADD": self._op_SEQ_ADD, "SEQ_SUB": self._op_SEQ_SUB,
            "SEQ_MUL": self._op_SEQ_MUL, "SEQ_DIV": self._op_SEQ_DIV,
            "SCAN_ADD": self._op_SCAN_ADD, "SCAN_MUL": self._op_SCAN_MUL,
            "DIFF_FWD": self._op_DIFF_FWD, "DIFF_BACK": self._op_DIFF_BACK,
            "CONV_FWD": self._op_CONV_FWD, "CONV_BACK": self._op_CONV_BACK,
            "REIDX_EVEN": self._op_REIDX_EVEN, "REIDX_ODD": self._op_REIDX_ODD,
            "DROP1": self._op_DROP1, "DROP2": self._op_DROP2,
            "INSERT1": self._op_INSERT1, "INSERT2": self._op_INSERT2,
            "PRED_POS": self._op_PRED_POS, "PRED_NEG": self._op_PRED_NEG,
            "PRED_IS_EVEN_N": self._op_PRED_IS_EVEN_N
        }

    def execute(self, program: Program, A: List[int], B: Optional[List[int]] = None) -> ExecResult:
        try:
            budget = Budget(self.cfg.t0, self.cfg.t_step)
            seq = self._eval(program.root, A, budget, B)
            return ExecResult(True, seq, budget.used, budget.allowed)
        except Exception as e:
            return ExecResult(False, None, 0, 0, str(e))

    def _eval(self, node: Node, A: List[int], budget: Budget, B: Optional[List[int]] = None) -> List[int]:
        handler = self.ops.get(node.op)
        if handler:
            return handler(node, A, budget, B)
        raise RuntimeError(f"unknown_op:{node.op}")

    def _get_X(self, node: Node, A: List[int], budget: Budget, B: Optional[List[int]]):
        return self._eval(node.kids[0], A, budget, B) if node.kids else A

    def _op_A(self, node, A, budget, B): return A

    def _op_SCALE(self, node, A, budget, B):
        X=self._get_X(node, A, budget, B); c=node.args[0]
        return [budget.add_for(i) or (budget.charge(1) or 0, c*X[i])[1] for i in range(len(X))]

    def _op_OFFSET(self, node, A, budget, B):
        X=self._get_X(node, A, budget, B); c=node.args[0]
        return [budget.add_for(i) or (budget.charge(1) or 0, X[i]+c)[1] for i in range(len(X))]

    def _op_MAP_ABS(self, node, A, budget, B):
        X=self._get_X(node, A, budget, B)
        return [budget.add_for(i) or (budget.charge(1) or 0, abs(X[i]))[1] for i in range(len(X))]

    def _op_MAP_SGN(self, node, A, budget, B):
        X=self._get_X(node, A, budget, B)
        return [budget.add_for(i) or (budget.charge(1) or 0, (1 if X[i]>0 else (0 if X[i]==0 else -1)))[1] for i in range(len(X))]

    def _op_MAP_MOD(self, node, A, budget, B):
        X=self._get_X(node, A, budget, B); m=node.args[0]
        if m==0: 
            if self.cfg.strict: raise RuntimeError("mod_by_zero")
            return [0]*len(X)
        return [budget.add_for(i) or (budget.charge(5) or 0, X[i]%m)[1] for i in range(len(X))]

    def _op_MAP_DIV(self, node, A, budget, B):
        X=self._get_X(node, A, budget, B); c=node.args[0]
        if c==0:
            if self.cfg.strict: raise RuntimeError("div_by_zero")
            return [0]*len(X)
        return [budget.add_for(i) or (budget.charge(5) or 0, X[i]//c)[1] for i in range(len(X))]

    def _op_MAP_SQRT(self, node, A, budget, B):
        import math
        X=self._get_X(node, A, budget, B)
        def isqrt_safe(n):
            if n<0: 
                if self.cfg.strict: raise RuntimeError("sqrt_negative")
                return 0
            return math.isqrt(n)
        return [budget.add_for(i) or (budget.charge(5) or 0, isqrt_safe(X[i]))[1] for i in range(len(X))]

    def _op_SEQ_ADD(self, node, A, budget, B):
        L=self._eval(node.kids[0], A, budget, B); R=self._eval(node.kids[1], A, budget, B)
        return [budget.add_for(i) or (budget.charge(1) or 0, L[i]+R[i])[1] for i in range(len(L))]

    def _op_SEQ_SUB(self, node, A, budget, B):
        L=self._eval(node.kids[0], A, budget, B); R=self._eval(node.kids[1], A, budget, B)
        return [budget.add_for(i) or (budget.charge(1) or 0, L[i]-R[i])[1] for i in range(len(L))]

    def _op_SEQ_MUL(self, node, A, budget, B):
        L=self._eval(node.kids[0], A, budget, B); R=self._eval(node.kids[1], A, budget, B)
        return [budget.add_for(i) or (budget.charge(1) or 0, L[i]*R[i])[1] for i in range(len(L))]

    def _op_SEQ_DIV(self, node, A, budget, B):
        L=self._eval(node.kids[0], A, budget, B); R=self._eval(node.kids[1], A, budget, B)
        def safe_div(a,b):
            if b==0:
                if self.cfg.strict: raise RuntimeError("seq_div_by_zero")
                else: return 0
            return a//b
        return [budget.add_for(i) or (budget.charge(1) or 0, safe_div(L[i],R[i]))[1] for i in range(len(L))]

    def _op_SCAN_ADD(self, node, A, budget, B):
        X=self._get_X(node, A, budget, B); s=0; Res=[]
        for i,v in enumerate(X):
            budget.add_for(i); budget.charge(1); s+=v; Res.append(s)
        return Res

    def _op_SCAN_MUL(self, node, A, budget, B):
        X=self._get_X(node, A, budget, B); p=1; Res=[]
        for i,v in enumerate(X):
            budget.add_for(i); budget.charge(1); p*=v; Res.append(p)
        return Res

    def _op_DIFF_FWD(self, node, A, budget, B):
        X=self._get_X(node, A, budget, B); k=node.args[0]
        if k<0: raise RuntimeError("diff_fwd_negative_k")
        N=len(X); Res=[0]*N
        upto = N-k if k<=N else 0
        for i in range(upto):
            budget.add_for(i); budget.charge(1); Res[i]=X[i+k]-X[i]
        for i in range(upto, N):
            budget.add_for(i); budget.charge(1); Res[i]=X[i]
        return Res

    def _op_DIFF_BACK(self, node, A, budget, B):
        X=self._get_X(node, A, budget, B); k=node.args[0]
        if k<0: raise RuntimeError("diff_back_negative_k")
        N=len(X); Res=[0]*N
        for i in range(min(k, N)):
            budget.add_for(i); budget.charge(1); Res[i]=X[i]
        for i in range(k, N):
            budget.add_for(i); budget.charge(1); Res[i]=X[i]-X[i-k]
        return Res

    def _op_CONV_FWD(self, node, A, budget, B):
        X=self._get_X(node, A, budget, B); k=node.args[0]
        if k<0: raise RuntimeError("conv_fwd_negative_k")
        N=len(X); Res=[0]*N
        upto = N-k if k<=N else 0
        for i in range(upto):
            budget.add_for(i); budget.charge(1); Res[i]=X[i]+X[i+k]
        for i in range(upto, N):
            budget.add_for(i); budget.charge(1); Res[i]=X[i]
        return Res

    def _op_CONV_BACK(self, node, A, budget, B):
        X=self._get_X(node, A, budget, B); k=node.args[0]
        if k<0: raise RuntimeError("conv_back_negative_k")
        N=len(X); Res=[0]*N
        for i in range(min(k, N)):
            budget.add_for(i); budget.charge(1); Res[i]=X[i]
        for i in range(k, N):
            budget.add_for(i); budget.charge(1); Res[i]=X[i]+X[i-k]
        return Res

    def _op_REIDX_EVEN(self, node, A, budget, B):
        """从第一项开始跳一个选一个 (索引 0,2,4,6...)"""
        X=self._get_X(node, A, budget, B); N=len(X)
        if N==0: return []
        Res=[]
        for i in range(0, N, 2):
            budget.add_for(len(Res)); budget.charge(1)
            Res.append(X[i])
        return Res

    def _op_REIDX_ODD(self, node, A, budget, B):
        """从第二项开始跳一个选一个 (索引 1,3,5,7...)"""
        X=self._get_X(node, A, budget, B); N=len(X)
        if N==0: return []
        Res=[]
        for i in range(1, N, 2):
            budget.add_for(len(Res)); budget.charge(1)
            Res.append(X[i])
        return Res

    def _op_DROP1(self, node, A, budget, B):
        X=self._get_X(node, A, budget, B)
        if len(X)<=1: return []
        Res=[]
        for i in range(len(X)-1):
            budget.add_for(i); budget.charge(1); Res.append(X[i+1])
        return Res

    def _op_DROP2(self, node, A, budget, B):
        X=self._get_X(node, A, budget, B); N=len(X)
        if N<=1: return X
        Res=[]
        for i in range(N):
            if i==1: continue
            budget.add_for(len(Res)); budget.charge(1); Res.append(X[i])
        return Res

    def _op_INSERT1(self, node, A, budget, B):
        X=self._get_X(node, A, budget, B); N=len(X)
        if N==0: return []
        c = 0
        if B is not None and len(B)>1: c=B[1]
        elif self.cfg.strict: raise RuntimeError("insert1_no_target")
        
        Y=[0]*N
        for i in range(N):
            budget.add_for(i); budget.charge(1)
            if i==0: Y[i]=X[0]
            elif i==1: Y[i]=c
            else: Y[i]=X[i-1]
        return Y

    def _op_INSERT2(self, node, A, budget, B):
        X=self._get_X(node, A, budget, B); N=len(X)
        if N==0: return []
        c = 0
        if B is not None and len(B)>2: c=B[2]
        elif self.cfg.strict: raise RuntimeError("insert2_no_target")
        
        Y=[0]*N
        for i in range(N):
            budget.add_for(i); budget.charge(1)
            if i<2: Y[i]=X[i]
            elif i==2: Y[i]=c
            else: Y[i]=X[i-1]
        return Y

    def _op_PRED_POS(self, node, A, budget, B):
        X=self._get_X(node, A, budget, B)
        return [budget.add_for(i) or (budget.charge(1) or 0, 1 if X[i]>0 else 0)[1] for i in range(len(X))]

    def _op_PRED_NEG(self, node, A, budget, B):
        X=self._get_X(node, A, budget, B)
        return [budget.add_for(i) or (budget.charge(1) or 0, 1 if X[i]<0 else 0)[1] for i in range(len(X))]

    def _op_PRED_IS_EVEN_N(self, node, A, budget, B):
        N=len(A)
        return [ (budget.add_for(i) or (budget.charge(1) or 0, 1 if (i%2==0) else 0)[1]) for i in range(N) ]
