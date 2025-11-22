import argparse, random, json, torch, math, glob, os
from torch.utils.data import Dataset, DataLoader, IterableDataset
from torch.amp import GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR, LambdaLR
from oeis.program import Node, Program
from oeis.interpreter import Interpreter, ExecConfig
from oeis.parser import parse_prefix
from oeis.torch_model import Cfg, TransDecoder, stoi, TOKENS, enhanced_features
from oeis.logging_config import setup_logger
from oeis.config import Config

# 设置日志
logger = setup_logger("train", level="INFO")

# ==========================================
# 1. Pre-generated Dataset
# ==========================================
class PreGeneratedDataset(IterableDataset):
    def __init__(self, data_dir, file_pattern="*.jsonl", cycle=True):
        super().__init__()
        self.files = glob.glob(os.path.join(data_dir, file_pattern))
        if not self.files:
            raise ValueError(f"No files found in {data_dir} matching {file_pattern}")
        logger.info(f"Found {len(self.files)} data files in {data_dir}")
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

        try:
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
        except GeneratorExit:
            # 优雅地处理生成器关闭，避免警告
            pass

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
    ap.add_argument("--steps", type=int, default=None, help="Training steps (default: from Config)")
    ap.add_argument("--bs", type=int, default=None, help="Batch size (default: from Config)")
    ap.add_argument("--lr", type=float, default=None, help="Learning rate (default: from Config)")
    ap.add_argument("--amp", action="store_true", help="Enable mixed precision training")
    ap.add_argument("--grad_accum", type=int, default=None, help="Gradient accumulation steps (default: from Config)")
    ap.add_argument("--warmup_steps", type=int, default=None, help="Warmup steps (default: from Config)")
    ap.add_argument("--save_every", type=int, default=5000, help="Save checkpoint every N steps")
    args = ap.parse_args()

    # 使用Config的默认值
    steps = args.steps if args.steps is not None else Config.DEFAULT_STEPS
    batch_size = args.bs if args.bs is not None else Config.DEFAULT_BATCH_SIZE
    lr = args.lr if args.lr is not None else Config.DEFAULT_LR
    grad_accum = args.grad_accum if args.grad_accum is not None else Config.GRADIENT_ACCUMULATION
    warmup_steps = args.warmup_steps if args.warmup_steps is not None else Config.WARMUP_STEPS

    # Check if data exists
    if not os.path.exists(args.data_dir) or not glob.glob(os.path.join(args.data_dir, "*.jsonl")):
        logger.error(f"No data found in {args.data_dir}. Please run generate_data.py first.")
        return

    ds = PreGeneratedDataset(data_dir=args.data_dir)
    dl = DataLoader(ds, batch_size=batch_size, num_workers=4, collate_fn=collate_batch)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 使用Config中的模型配置
    model_cfg_dict = Config.get_model_config()
    model_cfg = Cfg(**model_cfg_dict)
    model = TransDecoder(model_cfg, vocab=len(TOKENS)).to(device)
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    # 恢复检查点
    start_step = 0
    if os.path.exists(args.out):
        logger.info(f"Resuming from checkpoint {args.out}")
        ckpt = torch.load(args.out, map_location=device)
        model.load_state_dict(ckpt["model"])
        if "step" in ckpt:
            start_step = ckpt["step"]
            logger.info(f"Resuming from step {start_step}")
    
    # 优化器和学习率调度
    opt = torch.optim.AdamW(
        model.parameters(), 
        lr=lr,
        weight_decay=Config.WEIGHT_DECAY,
        betas=(0.9, 0.999)
    )
    loss_fn = torch.nn.CrossEntropyLoss()
    scaler = GradScaler(enabled=(args.amp and device=="cuda"))
    # inter = Interpreter(Config.get_interpreter_config(strict=True)) # 移除解释器
    
    # 学习率调度：warmup + cosine annealing
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            # Warmup阶段：线性增长
            return float(current_step) / float(max(1, warmup_steps))
        else:
            # Cosine annealing阶段
            progress = float(current_step - warmup_steps) / float(max(1, steps - warmup_steps))
            return max(Config.MIN_LR / lr, 0.5 * (1.0 + math.cos(math.pi * progress)))
    
    scheduler = LambdaLR(opt, lr_lambda)

    step = start_step
    model.train()
    effective_batch_size = batch_size * grad_accum
    logger.info(f"Starting training:")
    logger.info(f"  Steps: {steps}")
    logger.info(f"  Batch size: {batch_size} (effective: {effective_batch_size} with grad_accum={grad_accum})")
    logger.info(f"  Learning rate: {lr}")
    logger.info(f"  Warmup steps: {warmup_steps}")
    logger.info(f"  Device: {device}")
    logger.info(f"  Mixed precision: {args.amp}")
    logger.info(f"  Data directory: {args.data_dir}")

    # Use iterator for infinite loop control
    data_iter = iter(dl)
    accumulated_loss = 0.0
    
    while step < steps:
        # 梯度累积循环
        opt.zero_grad(set_to_none=True)
        batch_loss = 0.0
        
        for accum_step in range(grad_accum):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dl)
                batch = next(data_iter)

            x = batch["x"].to(device)
            y = batch["y"].to(device)
            feat = batch["feat"].to(device)
            
            with torch.amp.autocast(device_type=("cuda" if device=="cuda" else "cpu"), enabled=(args.amp and device=="cuda")):
                logits = model(x, feat)
                loss = loss_fn(logits.reshape(-1, logits.size(-1)), y.view(-1))
            
            # 梯度累积：除以累积步数
            scaled_loss = loss / grad_accum
            scaler.scale(scaled_loss).backward()
            
            batch_loss += loss.item()
        
        # 更新参数
        scaler.step(opt)
        scaler.update()
        scheduler.step()
        
        accumulated_loss += batch_loss / grad_accum
        
        # 日志输出
        if step % 50 == 0:
            avg_loss = accumulated_loss / min(50, step + 1)
            current_lr = scheduler.get_last_lr()[0]
            logger.info(f"step {step}/{steps} | loss {avg_loss:.4f} | lr {current_lr:.6f}")
            accumulated_loss = 0.0
        
        # 定期保存检查点
        if step > 0 and step % args.save_every == 0:
            checkpoint_path = args.out.replace(".pt", f"_step{step}.pt")
            torch.save({
                "cfg": model.cfg.__dict__,
                "model": model.state_dict(),
                "step": step,
                "optimizer": opt.state_dict(),
                "scheduler": scheduler.state_dict()
            }, checkpoint_path)
            logger.info(f"Checkpoint saved to {checkpoint_path}")
        
        step += 1
        if step >= steps:
            break
            
    # 保存最终模型
    torch.save({
        "cfg": model.cfg.__dict__,
        "model": model.state_dict(),
        "step": step,
        "optimizer": opt.state_dict(),
        "scheduler": scheduler.state_dict()
    }, args.out)
    logger.info(f"Training completed. Model saved to {args.out}")

if __name__ == "__main__":
    main()
