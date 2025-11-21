import argparse, random, json, torch, math, glob, os
from torch.utils.data import Dataset, DataLoader, IterableDataset
from torch.amp import GradScaler
from oeis.program import Node, Program
from oeis.interpreter import Interpreter, ExecConfig
from oeis.parser import parse_prefix
from oeis.torch_model import Cfg, TransDecoder, stoi, TOKENS, enhanced_features

# ==========================================
# 1. Pre-generated Dataset
# ==========================================
class PreGeneratedDataset(IterableDataset):
    def __init__(self, data_dir, file_pattern="*.jsonl", cycle=True):
        super().__init__()
        self.files = glob.glob(os.path.join(data_dir, file_pattern))
        if not self.files:
            raise ValueError(f"No files found in {data_dir} matching {file_pattern}")
        print(f"Found {len(self.files)} data files.")
        self.cycle = cycle

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            # Single process
            my_files = self.files
        else:
            # Split files among workers
            per_worker = int(math.ceil(len(self.files) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            start = worker_id * per_worker
            end = min(start + per_worker, len(self.files))
            my_files = self.files[start:end]
        
        if not my_files:
            return

        while True:
            random.shuffle(my_files)
            for fpath in my_files:
                with open(fpath, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            item = json.loads(line)
                            yield self._process_item(item)
                        except: continue
            if not self.cycle:
                break

    def _process_item(self, item):
        A = item["A"]
        B = item["B"]
        toks = item["toks"]
        is_moon = item["is_moon"]
        
        feat = enhanced_features(A, B)
        ctx = [stoi["<BOS>"]]
        x_toks = []; y_toks = []
        for t in toks + ["<EOS>"]:
            x_toks.append(ctx[-1])
            y_toks.append(stoi[t])
            ctx.append(stoi[t])
        
        return {
            "x": torch.tensor(x_toks, dtype=torch.long), 
            "y": torch.tensor(y_toks, dtype=torch.long),
            "feat": feat, 
            "A": torch.tensor(A, dtype=torch.long), 
            "B": torch.tensor(B, dtype=torch.long),
            "is_moon": torch.tensor(1 if is_moon else 0, dtype=torch.long)
        }

# Keep helper functions
def exec_loss_moon(inter, toks, A_vis, B_vis, k_strict=3, tau0=2e-3, tau1=1e-3, use_log=True):
    from oeis.parser import parse_prefix as parse2
    A_list = A_vis.tolist() if hasattr(A_vis, "tolist") else list(A_vis)
    B_list = B_vis.tolist() if hasattr(B_vis, "tolist") else list(B_vis)
    try:
        P = parse2([t for t in toks])
        r = inter.execute(P, A_list)
    except: return torch.tensor(0.0)

    if (not r.ok) or (r.seq is None) or len(r.seq)<len(A_list):
        return torch.tensor(0.0)
    
    Bh = r.seq[:len(A_list)]
    Bt = B_list[:len(A_list)]
    K = min(k_strict, len(Bt))
    head_err = sum(1.0 for i in range(K) if Bh[i] != Bt[i]) / max(1, K)
    tail_es = []
    eps = 1e-12
    for i in range(K, len(Bt)):
        y = Bt[i]; x = Bh[i]
        if y == 0: continue
        ratio = (abs(x)+eps)/(abs(y)+eps)
        e = abs(math.log(ratio)) if use_log else abs(1.0 - x/(y+eps))
        thr = tau0 + tau1 * i
        tail_es.append(max(0.0, e - thr))
    tail_err = sum(tail_es)/max(1, len(tail_es))
    return torch.tensor(head_err + tail_err, dtype=torch.float32)

def collate_batch(batch):
    return {
        "x": torch.nn.utils.rnn.pad_sequence([v["x"] for v in batch], batch_first=True, padding_value=0),
        "y": torch.nn.utils.rnn.pad_sequence([v["y"] for v in batch], batch_first=True, padding_value=-100),
        "feat": torch.stack([v["feat"] for v in batch], dim=0),
        "A": torch.nn.utils.rnn.pad_sequence([v["A"] for v in batch], batch_first=True, padding_value=0),
        "B": torch.nn.utils.rnn.pad_sequence([v["B"] for v in batch], batch_first=True, padding_value=0),
        "len": torch.tensor([len(v["y"]) for v in batch], dtype=torch.long),
        "is_moon": torch.stack([v["is_moon"] for v in batch], dim=0),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="ckpt.pt")
    ap.add_argument("--data_dir", default="data_gen", help="Directory containing .jsonl data files")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--lambda_exec", type=float, default=0.1)
    args = ap.parse_args()

    # Check if data exists
    if not os.path.exists(args.data_dir) or not glob.glob(os.path.join(args.data_dir, "*.jsonl")):
        print(f"Error: No data found in {args.data_dir}. Please run generate_data.py first.")
        return

    ds = PreGeneratedDataset(data_dir=args.data_dir)
    dl = DataLoader(ds, batch_size=args.bs, num_workers=2, collate_fn=collate_batch)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TransDecoder(Cfg(), vocab=len(TOKENS)).to(device)
    
    if os.path.exists(args.out):
        print(f"Resuming from {args.out}...")
        ckpt = torch.load(args.out, map_location=device)
        model.load_state_dict(ckpt["model"])
    
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.CrossEntropyLoss()
    scaler = GradScaler(enabled=(args.amp and device=="cuda"))
    inter = Interpreter(ExecConfig(strict=True, t0=10, t_step=3))

    step=0
    model.train()
    print(f"Start training on data from {args.data_dir}...")

    # Use iterator for infinite loop control
    data_iter = iter(dl)

    while step < args.steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dl)
            batch = next(data_iter)

        x = batch["x"].to(device); y=batch["y"].to(device); feat=batch["feat"].to(device)
        A = batch["A"].to("cpu"); B = batch["B"].to("cpu"); is_moon = batch["is_moon"].to("cpu")
        
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=("cuda" if device=="cuda" else "cpu"), enabled=(args.amp and device=="cuda")):
            logits = model(x, feat)
            loss_ce = loss_fn(logits.reshape(-1, logits.size(-1)), y.view(-1))
        
        loss = loss_ce
        if args.lambda_exec > 0:
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
            take = min(8, len(toks_batch))
            idxs = random.sample(range(len(toks_batch)), take)
            for j in idxs:
                if is_moon[j].item() == 1:
                    Aj = [int(v) for v in A[j].tolist() if v!=0]
                    Bj = [int(v) for v in B[j].tolist() if v!=0]
                    toks = toks_batch[j]
                    if len(toks)>0:
                        el = exec_loss_moon(inter, toks, Aj, Bj)
                        exec_losses.append(el)
            if exec_losses:
                loss_exec = torch.stack(exec_losses).mean().to(device)
                loss = loss_ce + args.lambda_exec * loss_exec

        scaler.scale(loss).backward()
        scaler.step(opt); scaler.update()
        
        if step % 50 == 0:
            print(f"step {step}/{args.steps} loss {loss.item():.4f} (ce {loss_ce.item():.4f})")
        
        step += 1
        if step >= args.steps: break
            
    torch.save({"cfg":model.cfg.__dict__, "model":model.state_dict()}, args.out)
    print("saved", args.out)

if __name__ == "__main__":
    main()
