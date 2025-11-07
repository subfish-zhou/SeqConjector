
import argparse, random, json, torch, math
from torch.utils.data import Dataset, DataLoader
from torch.amp import GradScaler
from oeis.program import Node, Program
from oeis.interpreter import Interpreter, ExecConfig
from oeis.parser import parse_prefix
from oeis.torch_model import Cfg, TransDecoder, stoi, TOKENS, cheap_features

def gen_A(N):
    kind = random.choice(["nat","squares","randwalk","smallpos"])
    if kind=="nat": return [i+1 for i in range(N)]
    if kind=="squares": return [(i+1)*(i+1) for i in range(N)]
    if kind=="randwalk":
        cur=0; out=[]
        for _ in range(N):
            cur += random.randint(-3,3); out.append(cur)
        return out
    return [random.randint(1,9) for _ in range(N)]


def choose_prog():
    ops=[]
    # basic atoms
    k = random.randint(-3,3)
    if k == 0: k = 2
    ops.append(Program(Node("SCALE",[k],[Node("A")])))
    ops.append(Program(Node("OFFSET",[random.randint(-4,4)],[Node("A")])))
    ops.append(Program(Node("SCAN_ADD",[],[Node("A")])))
    ops.append(Program(Node("DIFF_BACK",[random.choice([1,2,3])],[Node("A")])))
    ops.append(Program(Node("ZIP",[random.choice(["ADD","MUL"]), random.randint(0,2), random.randint(0,2)],[Node("A")])))
    L = random.choice([1,2,3]); w=[random.randint(-3,3) for _ in range(L)]
    ops.append(Program(Node("CONV",[L]+w,[Node("A")])))

    # optional INSERT at 1 or 2 with bounded constant in vocab
    if random.random()<0.2:
        base = Program(Node("SCAN_ADD",[],[Node("A")]))
        if random.random()<0.5:
            c = random.randint(-8, 8)
            ops.append(Program(Node("INSERT1",[c],[base.root])))
        else:
            c = random.randint(-8, 8)
            ops.append(Program(Node("INSERT2",[c],[base.root])))

    P = random.choice(ops)
    wrap = random.choice(["ID","MAP_ABS","CUMMAX"])
    if wrap!="ID":
        P = Program(Node(wrap,[],[P.root]))
    return P

class ShortDataset(Dataset):
    def __init__(self, M=200000, seed=0, moonshine_prob=0.1):
        super().__init__(); self.M=M; self.seed=seed; self.moonshine_prob=moonshine_prob
        self.samples=[self._gen(i) for i in range(M)]
    def _gen(self, i):
        random.seed(self.seed+i*7919)
        N_full = random.randint(12,20)
        n_in = random.randint(5,10)
        n_chk = min(random.randint(5,10), N_full-n_in)
        A_full = gen_A(N_full)
        P = choose_prog()
        inter = Interpreter(ExecConfig(strict=False, t0=10, t_step=3))
        r = inter.execute(P, A_full)
        if (not r.ok) or r.seq is None or len(r.seq)!=N_full:
            return self._gen(i+1234)
        B_full = r.seq
        is_moon = (random.random() < self.moonshine_prob)
        if is_moon and len(B_full) >= 8:
            s = random.randint(3,6)
            gamma = random.uniform(1e-3, 4e-3)
            B_mod = B_full[:]
            for t in range(s, len(B_mod)):
                fac = math.exp(gamma * (t - s))
                B_mod[t] = int(round(B_mod[t] * fac))
            B_full = B_mod
        toks = P.to_tokens()
        return (A_full[:n_in], B_full[:n_in], toks, is_moon)
    def __len__(self): return self.M
    def __getitem__(self, idx):
        A,B,toks,is_moon = self.samples[idx]
        feat = cheap_features(A,B)
        ctx=[stoi["<BOS>"]]; x=[]; y=[]
        for t in toks + ["<EOS>"]:
            x.append(ctx[-1]); y.append(stoi[t]); ctx.append(stoi[t])
        return {"x": torch.tensor(x, dtype=torch.long), "y": torch.tensor(y, dtype=torch.long),
                "feat": feat, "A": torch.tensor(A, dtype=torch.long), "B": torch.tensor(B, dtype=torch.long),
                "is_moon": torch.tensor(1 if is_moon else 0, dtype=torch.long)}


def exec_loss_moon(inter, toks, A_vis, B_vis, k_strict=3, tau0=2e-3, tau1=1e-3, use_log=True):
    from oeis.parser import parse_prefix as parse2
    # Ensure Python lists
    A_list = A_vis.tolist() if hasattr(A_vis, "tolist") else list(A_vis)
    B_list = B_vis.tolist() if hasattr(B_vis, "tolist") else list(B_vis)
    P = parse2([t for t in toks])
    r = inter.execute(P, A_list)
    if (not r.ok) or (r.seq is None) or len(r.seq)<len(A_list):
        return torch.tensor(0.0)
    Bh = r.seq[:len(A_list)]
    Bt = B_list[:len(A_list)]
    # head strict: binary 0/1 loss
    K = min(k_strict, len(Bt))
    head_err = sum(1.0 for i in range(K) if Bh[i] != Bt[i]) / max(1, K)
    # tail relaxed: deadzone L1 in log ratio
    tail_es = []
    import math
    eps = 1e-12
    for i in range(K, len(Bt)):
        y = Bt[i]
        if y == 0: continue
        x = Bh[i]
        ratio = (abs(x)+eps)/(abs(y)+eps)
        e = abs(math.log(ratio)) if use_log else abs(1.0 - x/(y+1e-12))
        thr = tau0 + tau1 * i
        tail_es.append(max(0.0, e - thr))
    tail_err = sum(tail_es)/max(1, len(tail_es))
    return torch.tensor(head_err + tail_err, dtype=torch.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="ckpt.pt")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--m", type=int, default=200000)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--moonshine_prob", type=float, default=0.1)
    ap.add_argument("--lambda_exec", type=float, default=0.1)
    ap.add_argument("--k_strict", type=int, default=3)
    ap.add_argument("--tau0", type=float, default=2e-3)
    ap.add_argument("--tau1", type=float, default=1e-3)
    args = ap.parse_args()

    ds = ShortDataset(M=args.m, seed=0, moonshine_prob=args.moonshine_prob)
    dl = DataLoader(ds, batch_size=args.bs, shuffle=True, num_workers=2, collate_fn=lambda b: {
        "x": torch.nn.utils.rnn.pad_sequence([v["x"] for v in b], batch_first=True, padding_value=0),
        "y": torch.nn.utils.rnn.pad_sequence([v["y"] for v in b], batch_first=True, padding_value=-100),
        "feat": torch.stack([v["feat"] for v in b], dim=0),
        "A": torch.nn.utils.rnn.pad_sequence([v["A"] for v in b], batch_first=True, padding_value=0),
        "B": torch.nn.utils.rnn.pad_sequence([v["B"] for v in b], batch_first=True, padding_value=0),
        "len": torch.tensor([len(v["y"]) for v in b], dtype=torch.long),
        "is_moon": torch.stack([v["is_moon"] for v in b], dim=0),
    })

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TransDecoder(Cfg(), vocab=len(TOKENS)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.CrossEntropyLoss()
    scaler = GradScaler(enabled=(args.amp and device=="cuda"))
    inter = Interpreter(ExecConfig(strict=True, t0=10, t_step=3))

    step=0
    for batch in dl:
        x = batch["x"].to(device); y=batch["y"].to(device); feat=batch["feat"].to(device)
        A=batch["A"].to("cpu"); B=batch["B"].to("cpu"); is_moon=batch["is_moon"].to("cpu")
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=("cuda" if device=="cuda" else "cpu"), enabled=(args.amp and device=="cuda")):
            logits = model(x, feat)
            loss_ce = loss_fn(logits.reshape(-1, logits.size(-1)), y.view(-1))
        exec_losses=[]
        toks_batch = []
        for row in y:
            toks = []
            for idx in row.tolist():
                if idx == -100: break
                tok = TOKENS[idx]
                if tok == "<EOS>": break
                toks.append(tok)
            toks_batch.append(toks)
        take = min(16, len(toks_batch))
        import random as _r
        idxs = _r.sample(range(len(toks_batch)), take)
        for j in idxs:
            if is_moon[j].item() == 1:
                Aj = [int(v) for v in A[j].tolist()]
                Bj = [int(v) for v in B[j].tolist()]
                toks = toks_batch[j]
                if len(Aj)>0 and len(Bj)>0 and len(toks)>0:
                    el = exec_loss_moon(inter, toks, Aj, Bj, k_strict=args.k_strict, tau0=args.tau0, tau1=args.tau1)
                    exec_losses.append(el)
        if exec_losses:
            loss_exec = torch.stack(exec_losses).mean().to(device)
            loss = loss_ce + args.lambda_exec * loss_exec
        else:
            loss = loss_ce
        scaler.scale(loss).backward()
        scaler.step(opt); scaler.update()
        if step % 100 == 0:
            print(f"step {step} loss {float(loss):.4f} (ce {float(loss_ce):.4f})")
        step += 1
        if step >= args.steps: break
    torch.save({"cfg":model.cfg.__dict__, "model":model.state_dict()}, args.out)
    print("saved", args.out)

if __name__ == "__main__":
    main()
