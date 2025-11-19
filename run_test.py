import json
import time
import argparse
import multiprocessing
import os
import torch
import sys

# Add current directory to path so we can import oeis modules
sys.path.append(os.getcwd())

from oeis.program import Program
from oeis.checker import check_program_on_pair, check_program_moonshine
from oeis.beam_egd import egd_beam_search
from main import try_templates_moonshine, TorchAdapter, RandomAdapter

# Global model for workers
worker_model = None

def init_worker(ckpt_path, device):
    global worker_model
    # Load model once per worker
    if ckpt_path:
        worker_model = TorchAdapter(ckpt_path, device=device)
    else:
        worker_model = RandomAdapter(seed=os.getpid())

def solve_pair(task):
    """
    task: (A_seq, B_seq, A_id, B_id, mode, n_in, n_chk, timeout)
    """
    A, B, A_id, B_id, mode, n_in, n_chk, timeout = task
    
    # Adjust n_chk if sequence is too short
    min_len = min(len(A), len(B))
    if n_in + n_chk > min_len:
        local_n_chk = max(0, min_len - n_in)
    else:
        local_n_chk = n_chk

    A_vis = A[:n_in]
    B_vis = B[:n_in]
    
    start_time = time.time()
    toks = []
    result_status = "fail"
    result_reason = ""
    
    try:
        # Logic copied/adapted from main.py cmd_beam
        
        # 1. Moonshine fast path (only if in moonshine mode)
        if mode == "moonshine":
            # Parameters taken from main.py defaults
            k_strict = 3
            tau0 = 2e-3
            tau1 = 1e-3
            
            toks0, rep0 = try_templates_moonshine(
                A, B, n_in, local_n_chk,
                k_strict=k_strict, tau0=tau0, tau1=tau1
            )
            
            if rep0 and getattr(rep0, "ok", False):
                return {
                    "A_id": A_id, "B_id": B_id, 
                    "mode": mode, "status": "success", 
                    "program": " ".join(toks0), 
                    "time": time.time() - start_time,
                    "reason": "template_hit"
                }

        # 2. Beam Search
        # Constraints
        k_strict = 3
        relerr0 = 2e-3
        relerr_hi = 0.10
        
        toks = egd_beam_search(
            worker_model, A_vis, B_vis,
            beam=256, # Standard beam size
            max_steps=96,
            use_ratio=True,
            k_strict=k_strict,
            err_thr_lo=relerr0,
            err_thr_hi=relerr_hi,
            time_limit=timeout
        )

        # 3. Check result
        if mode == "moonshine":
            rep = check_program_moonshine(
                toks, A_full=A, B_full=B,
                n_in=n_in, n_chk=local_n_chk,
                k_strict=k_strict, tau0=2e-3, tau1=1e-3
            )
        else:
            # Exact mode
            rep = check_program_on_pair(
                toks, A_full=A, B_full=B,
                n_in=n_in, n_chk=local_n_chk
            )
            
        if rep.ok:
            result_status = "success"
        else:
            result_status = "fail"
            result_reason = str(rep)

    except Exception as e:
        result_status = "error"
        result_reason = str(e)
        toks = []

    return {
        "A_id": A_id, 
        "B_id": B_id, 
        "mode": mode, 
        "status": result_status, 
        "program": " ".join(toks) if toks else "", 
        "time": time.time() - start_time,
        "reason": result_reason
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="oeis_seq_labeled/formula_true/trivial.jsonl")
    parser.add_argument("--ckpt", default="ckpt.pt")
    parser.add_argument("--output", default="results.jsonl")
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args()

    # 1. Load Data
    print(f"Loading data from {args.data}...")
    seq_map = {}
    with open(args.data, 'r') as f:
        for line in f:
            item = json.loads(line)
            seq_map[item['id']] = item

    # 2. Prepare Tasks
    tasks = []
    # We need to run both modes
    modes = ["exact", "moonshine"]
    
    print("Generating task list...")
    for b_id, item in seq_map.items():
        if "related_seq" not in item:
            continue
            
        b_seq = item['seq']
        for a_id in item['related_seq']:
            if a_id in seq_map:
                a_seq = seq_map[a_id]['seq']
                # Skip if sequences are empty or too short
                if not a_seq or not b_seq:
                    continue
                    
                # Add tasks for both modes
                for mode in modes:
                    # task: (A_seq, B_seq, A_id, B_id, mode, n_in, n_chk, timeout)
                    tasks.append((a_seq, b_seq, a_id, b_id, mode, 8, 8, 1.0))

    print(f"Total tasks: {len(tasks)}")

    # 3. Run Parallel
    num_workers = args.workers if args.workers > 0 else os.cpu_count()
    print(f"Running with {num_workers} workers...")
    
    # Check device for worker init
    device = "cpu" # Force CPU as requested for parallelism on non-GPU server
    
    results = []
    count = 0
    
    # Initialize pool with model loading
    with multiprocessing.Pool(processes=num_workers, initializer=init_worker, initargs=(args.ckpt, device)) as pool:
        for res in pool.imap_unordered(solve_pair, tasks, chunksize=10):
            results.append(res)
            count += 1
            if count % 100 == 0:
                print(f"Processed {count}/{len(tasks)}", end='\r')

    print(f"\nFinished. Saving to {args.output}...")
    
    with open(args.output, 'w') as f:
        for res in results:
            f.write(json.dumps(res) + "\n")

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True) # Safer for PyTorch
    main()

