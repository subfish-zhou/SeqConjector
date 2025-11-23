
from typing import List
import numpy as np, math, time
from .parser import OPS_SIG, parse_prefix
from .interpreter import Interpreter, ExecConfig
from .torch_model import stoi, TOKENS

class GrammarState:
    def __init__(self):
        self.stack=[]; self.expect_op=True
    def clone(self):
        g=GrammarState(); g.stack=[f.copy() for f in self.stack]; g.expect_op=self.expect_op; return g
    def push(self, op:str):
        args,kids = OPS_SIG[op]
        fr={"op":op,"need_args":args,"kids":kids,"mode":None}
        if op=="MAP_DIV": fr["need_args"]=1; fr["mode"]="MAP_DIV_ARG"
        self.stack.append(fr); self.expect_op=(fr["need_args"]==0 and fr["kids"]>0)
    def feed(self, tok:str)->bool:
        if self.expect_op:
            if tok not in OPS_SIG: return False
            self.push(tok)
        else:
            if not self.stack: return False
            fr=self.stack[-1]
            if fr["need_args"]<=0: return False
            try: v=int(tok)
            except: return False
            if fr["op"]=="MAP_DIV" and fr["mode"]=="MAP_DIV_ARG":
                if v==0: return False
                fr["need_args"]=0; self.expect_op=False
                if fr["kids"]>0: self.expect_op=True
                return True
            fr["need_args"]-=1; self.expect_op=(fr["need_args"]==0 and fr["kids"]>0)
        changed=True
        while changed and self.stack:
            changed=False
            fr=self.stack[-1]
            if fr["need_args"]==0:
                if fr["kids"]>0:
                    fr["kids"]-=1; self.expect_op=True
                    if fr["kids"]==0:
                        self.stack.pop(); changed=True
                else:
                    self.stack.pop(); changed=True
        if not self.stack: self.expect_op=True
        return True
    def allowed(self) -> List[str]:
        """
        语法感知的候选集生成：
        - 期待 op 时：仅返回 DSL 操作符；
        - 其他需要整数的地方：默认 {-16..16}。
        """
        if self.expect_op:
            from .parser import OPS_SIG
            return list(OPS_SIG.keys())

        if not self.stack:
            return []

        fr = self.stack[-1]

        # MAP_DIV: 除数不能为 0
        if fr["op"] == "MAP_DIV" and fr.get("mode") == "MAP_DIV_ARG" and fr["need_args"] == 1:
            return [str(i) for i in range(-16, 17) if i!=0]

        # 默认整数常量范围
        return [str(i) for i in range(-16, 17)]

    def complete(self)->bool: return len(self.stack)==0

def longest_prefix_parse(tokens: List[str]) -> int:
    for i in range(len(tokens),0,-1):
        try: parse_prefix(tokens[:i]); return i
        except: pass
    return 0

def egd_beam_search(model, A_vis, B_vis, feat, beam=256, max_steps=96,
                    use_ratio=True, k_strict=3,
                    err_thr_lo=0.02, err_thr_hi=0.10,
                    time_limit=10.0):
    """
    改动要点：
    1) 严格/宽松双执行器：宽松用于估计误差，严格用于提前剪掉"越界/未定义"的候选。
    2) 记录 best_by_err（rmse 最小的可解析前缀），作为最终兜底，避免回退到 `A`。
    """
    import time, math, numpy as np

    inter_loose   = Interpreter(ExecConfig(strict=False, t0=10, t_step=3))
    inter_strict  = Interpreter(ExecConfig(strict=True,  t0=10, t_step=3))

    class Hyp:
        __slots__ = ("toks","logp","state")
        def __init__(self, toks, logp, state):
            self.toks  = toks
            self.logp  = logp
            self.state = state

    class MWrap:
        def __init__(self, m):
            self.m = m
            self.ctx_len = getattr(getattr(m, "cfg", None), "ctx_len", 64)
        def logits(self, ctx_ids, feat_vec):
            if hasattr(self.m, "predict_logits"):
                return self.m.predict_logits(ctx_ids, feat_vec)
            import torch
            x = torch.tensor([ctx_ids], dtype=torch.long, device=getattr(self.m, "device", "cpu"))
            f = torch.tensor([feat_vec.tolist() if hasattr(feat_vec, "tolist") else feat_vec],
                             dtype=torch.float32, device=getattr(self.m, "device", "cpu"))
            with torch.no_grad():
                out = self.m.model(x, f)[0, -1, :].detach().cpu().numpy()
            return out

    def prefix_error(y_hat, y_true):
        K = min(k_strict, len(y_true))
        ok_head = (len(y_hat) >= K) and all(y_hat[i] == y_true[i] for i in range(K))
        errs = []
        for i, (x, y) in enumerate(zip(y_hat[:len(y_true)], y_true)):
            if i < K or y == 0:
                continue
            r = x / y
            e = abs(math.log(max(r, 1e-18))) if use_ratio else abs(1.0 - r)
            errs.append(e)
        rmse = (sum(e * e for e in errs) / len(errs)) ** 0.5 if errs else 0.0
        thr  = err_thr_hi if ok_head else err_thr_lo
        return ok_head, rmse, thr

    mw = MWrap(model)

    hyps = [Hyp([], 0.0, GrammarState())]
    finished = []
    best_by_err = None  # (rmse, toks)
    t0 = time.time()

    for step in range(max_steps):
        if time.time() - t0 > time_limit:
            break
        new = []
        for h in hyps:
            lp = longest_prefix_parse(h.toks)
            if lp > 0:
                try:
                    prog = parse_prefix(h.toks[:lp])
                    # 检查程序是否包含 A
                    if not prog.contains_A():
                        continue
                    rL = inter_loose.execute(prog, A_vis, B_vis)
                    if rL.ok and rL.seq is not None and len(rL.seq) >= len(B_vis):
                        # 严格前缀校验：提前淘汰越界/未定义
                        rS = inter_strict.execute(prog, A_vis, B_vis)
                        if not (rS.ok and rS.seq is not None and len(rS.seq) >= len(B_vis)):
                            continue
                        ok_head, rmse, thr = prefix_error(rL.seq, B_vis)
                        if (best_by_err is None) or (rmse < best_by_err[0]):
                            best_by_err = (rmse, h.toks[:lp])
                        # 完整可解析 + 误差在阈内 => 直接收敛
                        if lp == len(h.toks) and ok_head and rmse <= thr:
                            finished.append(h)
                            continue
                        # 误差超过阈值 => 剪枝
                        if rmse > thr:
                            continue
                except:
                    pass

            allowed = h.state.allowed()
            ctx = ["<BOS>"] + h.toks
            ctx_ids = [stoi.get(t, 0) for t in ctx][-64:]
            logits = mw.logits(ctx_ids, feat)

            # 只开放 allowed token
            mask = np.full_like(logits, -1e9, dtype=float)
            for tok in allowed:
                if tok in stoi:
                    mask[stoi[tok]] = 0.0
            masked = logits + mask
            topk = min(beam, len(allowed)) if allowed else 0
            if topk == 0:
                continue
            idx = np.argpartition(-masked, topk - 1)[:topk]
            vals = masked[idx]
            for v, i in zip(vals, idx):
                tok = TOKENS[int(i)]
                st2 = h.state.clone()
                if not st2.feed(tok):
                    continue
                new.append(Hyp(h.toks + [tok], h.logp + float(v), st2))

        if not new:
            break
        # 归一化长度，避免极短串吃到不公平优势
        new.sort(key=lambda x: x.logp / max(1, len(x.toks)), reverse=True)
        hyps = new[:beam]
        if finished:
            break

    # 选择返回：优先 finished；否则误差最小；再否则最优得分的可解析前缀；最后兜底 ["A"]
    def as_tokens(h):
        return h.toks[:longest_prefix_parse(h.toks)]

    if finished:
        return as_tokens(finished[0])
    if best_by_err is not None:
        return best_by_err[1]
    if hyps:
        return as_tokens(hyps[0]) or ["A"]
    return ["A"]
