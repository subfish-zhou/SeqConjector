
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

def tau(n:int)->int:
    if n==0: return 0
    x=abs(n); c=1; p=2
    while p*p<=x:
        e=0
        while x%p==0: x//=p; e+=1
        if e>0: c*=e+1
        p+=1
    if x>1: c*=2
    return c

def phi(n:int)->int:
    if n==0: return 0
    x=abs(n); r=x; p=2
    while p*p<=x:
        if x%p==0:
            while x%p==0: x//=p
            r-=r//p
        p+=1
    if x>1: r-=r//x
    return r if n>=0 else -r

def sigma(n:int)->int:
    if n==0: return 0
    s=0; i=1; x=abs(n)
    while i*i<=x:
        if x%i==0:
            j=x//i; s+=i; 
            if j!=i: s+=j
        i+=1
    return s

def mu(n:int)->int:
    if n==0: return 0
    x=abs(n); m=1; p=2
    while p*p<=x:
        if x%p==0:
            x//=p
            if x%p==0: return 0
            m=-m
        p+=1
    if x>1: m=-m
    return m

def omega(n:int, big=False)->int:
    if n==0: return 0
    x=abs(n); c=0; p=2
    while p*p<=x:
        if x%p==0:
            c+=1
            if big:
                while x%p==0: x//=p; c+=1
            else:
                while x%p==0: x//=p
        p+=1
    if x>1: c+=1
    return c

class Interpreter:
    def __init__(self, cfg: ExecConfig=ExecConfig()):
        self.cfg=cfg

    def execute(self, program: Program, A: List[int]) -> ExecResult:
        try:
            budget = Budget(self.cfg.t0, self.cfg.t_step)
            seq = self._eval(program.root, A, budget)
            return ExecResult(True, seq, budget.used, budget.allowed)
        except Exception as e:
            return ExecResult(False, None, 0, 0, str(e))

    def _eval(self, node: Node, A: List[int], budget: Budget) -> List[int]:
        op=node.op
        if op=="A": return A
        X = self._eval(node.kids[0], A, budget) if node.kids else A
        if op=="SCALE":
            c=node.args[0]; return [budget.add_for(i) or (budget.charge(1) or 0, c*X[i])[1] for i in range(len(X))]
        if op=="OFFSET":
            c=node.args[0]; return [budget.add_for(i) or (budget.charge(1) or 0, X[i]+c)[1] for i in range(len(X))]
        if op=="MAP_ABS":
            return [budget.add_for(i) or (budget.charge(1) or 0, abs(X[i]))[1] for i in range(len(X))]
        if op=="MAP_SGN":
            return [budget.add_for(i) or (budget.charge(1) or 0, (1 if X[i]>0 else (0 if X[i]==0 else -1)))[1] for i in range(len(X))]
        if op=="MAP_MOD":
            m=node.args[0]; return [budget.add_for(i) or (budget.charge(5) or 0, X[i]%m)[1] for i in range(len(X))]
        if op=="MAP_TAU":   return [budget.add_for(i) or (budget.charge(50) or 0, tau(X[i]))[1] for i in range(len(X))]
        if op=="MAP_SIGMA": return [budget.add_for(i) or (budget.charge(80) or 0, sigma(X[i]))[1] for i in range(len(X))]
        if op=="MAP_PHI":   return [budget.add_for(i) or (budget.charge(50) or 0, phi(X[i]))[1] for i in range(len(X))]
        if op=="MAP_MU":    return [budget.add_for(i) or (budget.charge(50) or 0, mu(X[i]))[1] for i in range(len(X))]
        if op=="MAP_OMEGA": return [budget.add_for(i) or (budget.charge(50) or 0, omega(X[i], False))[1] for i in range(len(X))]
        if op=="MAP_BIGOMEGA": return [budget.add_for(i) or (budget.charge(50) or 0, omega(X[i], True))[1] for i in range(len(X))]
        if op=="SCAN_ADD":
            s=0; B=[]
            for i,v in enumerate(X):
                budget.add_for(i); budget.charge(1); s+=v; B.append(s)
            return B
        if op=="SCAN_MUL":
            p=1; B=[]
            for i,v in enumerate(X):
                budget.add_for(i); budget.charge(1); p*=v; B.append(p)
            return B
        if op=="CUMMAX":
            m=-10**18; B=[]
            for i,v in enumerate(X):
                budget.add_for(i); budget.charge(1); 
                if v>m: m=v
                B.append(m)
            return B
        if op=="CUMMIN":
            m=10**18; B=[]
            for i,v in enumerate(X):
                budget.add_for(i); budget.charge(1); 
                if v<m: m=v
                B.append(m)
            return B
        if op=="DIFF_FWD":
            k=node.args[0];
            if k<0: raise RuntimeError("diff_fwd_negative_k")
            N=len(X); B=[0]*N
            upto = N-k if k<=N else 0
            for i in range(upto):
                budget.add_for(i); budget.charge(1); B[i]=X[i+k]-X[i]
            if self.cfg.strict and k>0:
                raise RuntimeError("diff_fwd_tail_undefined")
            return B
        if op=="DIFF_BACK":
            k=node.args[0];
            if k<0: raise RuntimeError("diff_back_negative_k")
            N=len(X); B=[0]*N
            for i in range(N):
                if i-k<0:
                    if self.cfg.strict: raise RuntimeError("diff_back_head_undefined")
                    else: B[i]=0; continue
                budget.add_for(i); budget.charge(1); B[i]=X[i]-X[i-k]
            return B
        if op=="ZIP":
            bop, k1, k2 = node.args
            if k1<0 or k2<0:
                if self.cfg.strict: raise RuntimeError("zip_negative_delay")
                else: return [0]*len(X)
            N=len(X); B=[0]*N
            for i in range(N):
                j1=i-k1; j2=i-k2
                if j1<0 or j2<0 or j1>=N or j2>=N:
                    if self.cfg.strict: raise RuntimeError("zip_oob")
                    else: B[i]=0; continue
                budget.add_for(i); budget.charge(1)
                x=X[j1]; y=X[j2]
                if bop=="ADD": B[i]=x+y
                elif bop=="SUB": B[i]=x-y
                elif bop=="MUL": B[i]=x*y
                elif bop=="MIN": B[i]=x if x<y else y
                elif bop=="MAX": B[i]=x if x>y else y
            return B
        if op=="CONV":
            L=node.args[0]; w=node.args[1:1+L]; N=len(X); B=[0]*N
            for i in range(N):
                s=0
                for j in range(min(L, i+1)):
                    budget.add_for(i); budget.charge(2)
                    s += w[j]*X[i-j]
                B[i]=s
            return B
        if op=="POLY":
            D=node.args[0]; coeffs=node.args[1:1+D+1]
            L=len(coeffs); N=len(X); B=[0]*N
            for i in range(N):
                s=0
                for j in range(min(L, i+1)):
                    budget.add_for(i); budget.charge(2)
                    s += coeffs[j]*X[i-j]
                B[i]=s
            return B
        if op=="REIDX":
            k,b = node.args; N=len(X)
            if k<0: raise RuntimeError("reidx_negative_k")
            if N==0: return []
            # Ensure all indices valid for i in [0..N-1]
            if (k*(N-1)+b) >= N or (0*b) < 0 or (k*0+b) < 0:
                raise RuntimeError("reidx_oob")
            return [budget.add_for(i) or (budget.charge(1) or 0, X[k*i+b])[1] for i in range(N)]
        if op=="SUBSAMPLE":
            k=node.args[0]
            if k<=0: raise RuntimeError("subsample_nonpositive_k")
            return [budget.add_for(i) or (budget.charge(3) or 0, X[i*k])[1] for i in range(len(X)//k)]
        if op=="REPEAT":
            k=node.args[0];
            if k<=0: raise RuntimeError("repeat_nonpositive_k")
            B=[]
            for i in range(len(X)*k):
                budget.add_for(i); budget.charge(3); B.append(X[i//k])
            return B
        if op=="INDEXBY":
            N=len(X); B=[0]*N
            if max(X) >= N or min(X)<0: raise RuntimeError("indexby_oob")
            for i in range(N):
                budget.add_for(i); budget.charge(1); B[i]=X[X[i]]
            return B
        if op=="DROP":
            k=node.args[0]; 
            if len(X)<=k: return []
            B=[]
            for i in range(len(X)-k):
                budget.add_for(i); budget.charge(1); B.append(X[i+k])
            return B
        if op=="INSERT1":
            c=node.args[0]; N=len(X); 
            if N==0: return []
            Y=[0]*N
            for i in range(N):
                budget.add_for(i); budget.charge(1)
                if i==0: Y[i]=X[0]
                elif i==1: Y[i]=c
                else: Y[i]=X[i-1]
            return Y
        if op=="INSERT2":
            c=node.args[0]; N=len(X); 
            if N==0: return []
            Y=[0]*N
            for i in range(N):
                budget.add_for(i); budget.charge(1)
                if i<2: Y[i]=X[i]
                elif i==2: Y[i]=c
                else: Y[i]=X[i-1]
            return Y
        if op=="BINOM":
            N=len(X); B=[0]*N
            row=[0]*(N+1); row[0]=1
            B[0]=row[0]*X[0]
            for n in range(1,N):
                for k in range(n,0,-1):
                    row[k]+=row[k-1]
                s=0
                for k in range(n+1):
                    budget.add_for(n); budget.charge(2); s+=row[k]*X[k]
                B[n]=s
            return B
        if op=="IBINOM":
            N=len(X); B=[0]*N
            row=[0]*(N+1); row[0]=1
            B[0]=row[0]*X[0]
            for n in range(1,N):
                for k in range(n,0,-1):
                    row[k]+=row[k-1]
                s=0
                for k in range(n+1):
                    budget.add_for(n); budget.charge(2)
                    s += ((1 if (n-k)%2==0 else -1) * row[k] * X[k])
                B[n]=s
            return B
        if op=="EULER":
            N=len(X); 
            if N==0: return []
            B=[0]*N; B[0]=1; c=[0]*N
            for d in range(1,N):
                if X[d]==0: continue
                mul = d*X[d]
                for m in range(d, N, d):
                    budget.add_for(m); budget.charge(2)
                    c[m]+=mul
            for n in range(1,N):
                s=0
                for k in range(1,n+1):
                    budget.add_for(n); budget.charge(2)
                    s += c[k]*B[n-k]
                if s % n != 0 and self.cfg.strict:
                    raise RuntimeError("euler_non_integer")
                B[n] = s//n
            return B
        if op=="PRED_POS":
            return [budget.add_for(i) or (budget.charge(1) or 0, 1 if X[i]>0 else 0)[1] for i in range(len(X))]
        if op=="PRED_NEG":
            return [budget.add_for(i) or (budget.charge(1) or 0, 1 if X[i]<0 else 0)[1] for i in range(len(X))]
        if op=="PRED_IS_EVEN_N":
            N=len(A)
            return [ (budget.add_for(i) or (budget.charge(1) or 0, 1 if (i%2==0) else 0)[1]) for i in range(N) ]
        if op=="PRED_EQ_CONST":
            c=node.args[0]; return [budget.add_for(i) or (budget.charge(1) or 0, 1 if X[i]==c else 0)[1] for i in range(len(X))]
        if op=="PRED_GT_CONST":
            c=node.args[0]; return [budget.add_for(i) or (budget.charge(1) or 0, 1 if X[i]>c else 0)[1] for i in range(len(X))]
        if op=="PRED_LT_CONST":
            c=node.args[0]; return [budget.add_for(i) or (budget.charge(1) or 0, 1 if X[i]<c else 0)[1] for i in range(len(X))]
        if op=="PRED_NOT":
            P=X; return [budget.add_for(i) or (budget.charge(1) or 0, 0 if P[i]!=0 else 1)[1] for i in range(len(P))]
        if op=="PRED_AND":
            L=self._eval(node.kids[0], A, budget); R=self._eval(node.kids[1], A, budget)
            return [budget.add_for(i) or (budget.charge(1) or 0, 1 if (L[i]!=0 and R[i]!=0) else 0)[1] for i in range(len(L))]
        if op=="PRED_OR":
            L=self._eval(node.kids[0], A, budget); R=self._eval(node.kids[1], A, budget)
            return [budget.add_for(i) or (budget.charge(1) or 0, 1 if (L[i]!=0 or R[i]!=0) else 0)[1] for i in range(len(L))]
        if op=="COND":
            P=self._eval(node.kids[0], A, budget); T=self._eval(node.kids[1], A, budget); E=self._eval(node.kids[2], A, budget)
            N=len(P); 
            if not (len(T)==N and len(E)==N): raise RuntimeError("cond_len_mismatch")
            B=[0]*N
            for i in range(N):
                budget.add_for(i); budget.charge(1); B[i]=T[i] if P[i]!=0 else E[i]
            return B
        raise RuntimeError(f"unknown_op:{op}")
