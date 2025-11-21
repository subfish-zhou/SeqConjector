
import json, argparse, numpy as np, torch, math, time
from oeis.program import Node, Program
from oeis.parser import parse_prefix
from oeis.interpreter import Interpreter, ExecConfig
from oeis.checker import check_program_on_pair, check_program_moonshine
from oeis.beam_egd import egd_beam_search, longest_prefix_parse
from oeis.torch_model import Cfg, TransDecoder, stoi, itos, TOKENS, cheap_features
from oeis.split_utils import compute_split

def try_templates_moonshine(A, B, n_in, n_chk, k_strict=3, tau0=2e-3, tau1=1e-3):
    """
    模板集合：SCAN_ADD A；INSERT1 B[1] (SCAN_ADD A)；INSERT2 B[2] (SCAN_ADD A)
    逻辑：只要有模板通过 moonshine 检验，立刻返回；
         否则返回“尾部误差最小”的那个模板（兜底），其 rep.ok=False，供上层继续走 beam。
    """
    N2 = n_in + n_chk
    cands = []

    toks1 = ["SCAN_ADD", "A"]
    cands.append(("SCAN_ADD", toks1))

    if N2 >= 2 and len(B) >= 2:
        c = B[1]
        toks2 = ["INSERT1", str(c), "SCAN_ADD", "A"]
        cands.append(("INS1", toks2))

    if N2 >= 3 and len(B) >= 3:
        c = B[2]
        toks3 = ["INSERT2", str(c), "SCAN_ADD", "A"]
        cands.append(("INS2", toks3))

    from oeis.checker import check_program_moonshine
    best = None  # (rmse, toks, rep)
    import math

    for tag, toks in cands:
        rep = check_program_moonshine(
            toks, A_full=A, B_full=B,
            n_in=n_in, n_chk=n_chk,
            k_strict=k_strict, tau0=tau0, tau1=tau1
        )
        if rep.ok:
            return toks, rep  # 直接命中
        # 估算 tail 误差大小（用严格头部 + log 相对误差的 RMSE）
        # 这里重用 checker 的评价：rep.reason 里若包含 'tail_exceed'，说明误差较大
        # 为简单起见，按“尾部严格失败”的固定打分，否则给一个较小的分数。
        score = 1e9
        if isinstance(rep.reason, str) and "moonshine_tail_exceed" in rep.reason:
            try:
                val = float(rep.reason.split(":")[-1])
                score = val
            except:
                score = 1e6
        else:
            score = 1e3
        if (best is None) or (score < best[0]):
            best = (score, toks, rep)

    # 没有模板通过：返回“最接近”的模板，rep.ok=False，供上层继续 beam
    return (best[1], best[2]) if best is not None else ([], None)



class RandomAdapter:
    def __init__(self, seed=0):
        self.rng = np.random.RandomState(seed)
        class C: pass
        self.cfg = C(); self.cfg.ctx_len=64
    def predict_logits(self, ctx_ids, feat):
        V = len(TOKENS)
        return self.rng.randn(V).astype(np.float32)*0.01

class TorchAdapter:
    def __init__(self, ckpt_path: str, device: str="cpu"):
        ck = torch.load(ckpt_path, map_location=device)
        cfg = Cfg(**ck["cfg"])
        self.model = TransDecoder(cfg, vocab=len(TOKENS)).to(device)
        self.model.load_state_dict(ck["model"]); self.model.eval()
        self.device = device
        self.cfg = type("C", (), {"ctx_len": cfg.ctx_len})
    def predict_logits(self, ctx_ids, feat_vec):
        x = torch.tensor([ctx_ids], dtype=torch.long, device=self.device)
        f = torch.tensor([feat_vec.tolist() if hasattr(feat_vec,"tolist") else feat_vec], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            logits = self.model(x, f)[0, -1, :].detach().cpu().numpy()
        return logits

def cmd_eval(args):
    A = json.load(open(args.A)); B = json.load(open(args.B))
    toks = args.program.strip().split()
    
    # Auto-compute split if not provided
    if args.n_in is None or args.n_chk is None:
        min_len = min(len(A), len(B))
        n_in, n_chk = compute_split(min_len)
        if args.n_in is None: args.n_in = n_in
        if args.n_chk is None: args.n_chk = n_chk
    
    if args.moonshine:
        rep = check_program_moonshine(toks, A_full=A, B_full=B, n_in=args.n_in, n_chk=args.n_chk,
                                      k_strict=args.k_strict, tau0=args.relerr0, tau1=args.relerr_step)
    else:
        rep = check_program_on_pair(toks, A_full=A, B_full=B, n_in=args.n_in, n_chk=args.n_chk)
    print(rep)

def cmd_beam(args):
    # 这些工具通常在本文件已有全局导入；若没有则取消下面两行注释
    # from oeis.checker import check_program_on_pair, check_program_moonshine
    # from oeis.beam_egd import egd_beam_search
    # from oeis.torch_model import TorchAdapter, RandomAdapter

    A = json.load(open(args.A)); B = json.load(open(args.B))
    
    # Auto-compute split if not provided
    if args.n_in is None or args.n_chk is None:
        min_len = min(len(A), len(B))
        n_in_auto, n_chk_auto = compute_split(min_len)
        if args.n_in is None: args.n_in = n_in_auto
        if args.n_chk is None: args.n_chk = n_chk_auto
    
    n_in, n_chk = args.n_in, args.n_chk
    if n_in + n_chk > min(len(A), len(B)):
        print(f"[WARN] n_in+n_chk={n_in+n_chk} exceeds min length {min(len(A),len(B))}, shrinking n_chk")
        n_chk = max(0, min(len(A), len(B)) - n_in)

    # 1) 模板快速路径（Moonshine）：总是打印模板耗时；仅在通过时 return
    if args.moonshine:
        t_tpl0 = time.time()
        toks0, rep0 = try_templates_moonshine(
            A, B, n_in, n_chk,
            k_strict=args.k_strict, tau0=args.relerr0, tau1=args.relerr_step
        )
        t_tpl = time.time() - t_tpl0
        if toks0:
            print("PRED(TPL):", " ".join(toks0))
            print("CHECK(TPL):", rep0)
            print(f"TIME tpl={t_tpl:.3f}s")  # 无论 rep0.ok 与否都打印
            if rep0 and getattr(rep0, "ok", False):
                return  # 模板直接命中则退出；否则继续进入束搜

    # 2) 束搜（EGD）
    A_vis, B_vis = A[:n_in], B[:n_in]
    if args.ckpt:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = TorchAdapter(args.ckpt, device=device)
    else:
        model = RandomAdapter(seed=0)

    t_beam0 = time.time()
    toks = egd_beam_search(
        model, A_vis, B_vis,
        beam=args.beam,
        max_steps=getattr(args, "max_steps", 96),
        use_ratio=True,
        k_strict=args.k_strict,
        err_thr_lo=args.relerr0,
        err_thr_hi=args.relerr_hi,
        time_limit=args.time_limit
    )
    t_beam = time.time() - t_beam0
    print("PRED:", " ".join(toks))

    # 3) 校验 + 打印耗时（无论通过与否）
    t_chk0 = time.time()
    if args.moonshine:
        rep = check_program_moonshine(
            toks, A_full=A, B_full=B,
            n_in=n_in, n_chk=n_chk,
            k_strict=args.k_strict, tau0=args.relerr0, tau1=args.relerr_step
        )
    else:
        rep = check_program_on_pair(
            toks, A_full=A, B_full=B,
            n_in=n_in, n_chk=n_chk
        )
    t_check = time.time() - t_chk0
    print("CHECK:", rep)
    print(f"TIME beam={t_beam:.3f}s check={t_check:.3f}s total={(t_beam+t_check):.3f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers()

    ap_eval = sub.add_parser("eval")
    ap_eval.add_argument("--A", required=True)
    ap_eval.add_argument("--B", required=True)
    ap_eval.add_argument("--program", required=True)
    ap_eval.add_argument("--n_in", type=int, default=None, help="Auto-compute if not provided")
    ap_eval.add_argument("--n_chk", type=int, default=None, help="Auto-compute if not provided")
    ap_eval.add_argument("--moonshine", action="store_true")
    ap_eval.add_argument("--k_strict", type=int, default=3)
    ap_eval.add_argument("--relerr0", type=float, default=2e-3)
    ap_eval.add_argument("--relerr_step", type=float, default=1e-3)
    ap_eval.set_defaults(func=cmd_eval)

    ap_beam = sub.add_parser("beam")
    ap_beam.add_argument("--A", required=True)
    ap_beam.add_argument("--B", required=True)
    ap_beam.add_argument("--n_in", type=int, default=None, help="Auto-compute if not provided")
    ap_beam.add_argument("--n_chk", type=int, default=None, help="Auto-compute if not provided")
    ap_beam.add_argument("--beam", type=int, default=256)
    ap_beam.add_argument("--max_steps", type=int, default=96)
    ap_beam.add_argument("--ckpt", default="")
    ap_beam.add_argument("--moonshine", action="store_true")
    ap_beam.add_argument("--k_strict", type=int, default=3)
    ap_beam.add_argument("--relerr0", type=float, default=2e-3)
    ap_beam.add_argument("--relerr_step", type=float, default=1e-3)
    ap_beam.add_argument("--relerr_hi", type=float, default=0.10)
    ap_beam.add_argument("--time_limit", type=float, default=10.0)
    ap_beam.set_defaults(func=cmd_beam)

    args = ap.parse_args()
    args.func(args)
