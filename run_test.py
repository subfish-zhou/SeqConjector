import json
import time
import argparse
import multiprocessing
import os
import torch
import sys
import logging

# Add current directory to path so we can import oeis modules
sys.path.append(os.getcwd())

from oeis.program import Program
from oeis.checker import check_program_on_pair, check_program_moonshine
from oeis.beam_egd import egd_beam_search
from main import try_templates_moonshine, TorchAdapter, RandomAdapter

logger = logging.getLogger(__name__)

# Global model for workers
worker_model = None

def init_worker(ckpt_path, device):
    global worker_model
    # Load model once per worker
    if ckpt_path:
        logger.info(f"[pid={os.getpid()}] Loading checkpoint from {ckpt_path} on {device}")
        worker_model = TorchAdapter(ckpt_path, device=device)
    else:
        logger.info(f"[pid={os.getpid()}] Using RandomAdapter")
        worker_model = RandomAdapter(seed=os.getpid())

def solve_pair(task):
    """
    task: (A_seq, B_seq, A_id, B_id, mode, n_in, n_chk, timeout)
    """
    A, B, A_id, B_id, mode, n_in, n_chk, timeout = task
    
    # Adjust n_in / n_chk if sequence is too short
    min_len = min(len(A), len(B))
    local_n_in = min(n_in, min_len)
    remaining = max(0, min_len - local_n_in)
    local_n_chk = min(n_chk, remaining)

    A_vis = A[:local_n_in]
    B_vis = B[:local_n_in]
    
    start_time = time.time()
    toks = []
    result_status = "fail"
    result_reason = ""
    logger.debug(f"Task start A:{A_id} B:{B_id} mode:{mode} n_in:{n_in} n_chk:{local_n_chk}")
    
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
                duration = time.time() - start_time
                logger.info(f"Template hit A:{A_id} B:{B_id} mode:{mode} time:{duration:.3f}s")
                return {
                    "A_id": A_id, "B_id": B_id, 
                    "mode": mode, "status": "success", 
                    "program": " ".join(toks0), 
                    "time": duration,
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
            logger.info(f"Solved A:{A_id} B:{B_id} mode:{mode} len:{len(toks)} time:{time.time()-start_time:.3f}s")
        else:
            result_status = "fail"
            result_reason = str(rep)
            logger.debug(f"Validation failed A:{A_id} B:{B_id} mode:{mode} reason:{result_reason}")

    except Exception as e:
        result_status = "error"
        result_reason = str(e)
        toks = []
        logger.exception(f"Error solving A:{A_id} B:{B_id} mode:{mode}: {e}")

    return {
        "A_id": A_id, 
        "B_id": B_id, 
        "mode": mode, 
        "status": result_status, 
        "program": " ".join(toks) if toks else "", 
        "time": time.time() - start_time,
        "reason": result_reason
    }

def setup_logging(level: str):
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s [pid=%(process)d] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger.info(f"Logger initialized at level {level.upper()}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="oeis_seq_labeled/formula_true/trivial.jsonl")
    parser.add_argument("--ckpt", default="ckpt.pt")
    parser.add_argument("--output", default="results.jsonl")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--log-level", default="INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR)")
    parser.add_argument("--chunk-size", type=int, default=1, help="Task chunk size for multiprocessing (smaller=better load balance)")
    args = parser.parse_args()

    setup_logging(args.log_level)
    # 1. Load Data
    logger.info(f"Loading data from {args.data}...")
    seq_map = {}
    with open(args.data, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            seq_map[item['id']] = item

    # 2. Prepare Tasks
    tasks = []
    # We need to run both modes
    modes = ["exact", "moonshine"]
    
    logger.info("Generating task list...")
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
                    
                # Add tasks for both modes with auto-computed split
                from oeis.split_utils import compute_split
                min_len = min(len(a_seq), len(b_seq))
                n_in, n_chk = compute_split(min_len)
                
                for mode in modes:
                    # task: (A_seq, B_seq, A_id, B_id, mode, n_in, n_chk, timeout)
                    tasks.append((a_seq, b_seq, a_id, b_id, mode, n_in, n_chk, 1.0))

    logger.info(f"Total tasks: {len(tasks)}")

    # 3. Run Parallel
    num_workers = args.workers if args.workers > 0 else os.cpu_count()
    logger.info(f"Running with {num_workers} workers...")
    
    # Check device for worker init
    device = "cpu" # Force CPU as requested for parallelism on non-GPU server
    
    results = []
    count = 0
    
    # Initialize pool with model loading
    summary = {"success": 0, "fail": 0, "error": 0}
    chunk_size = max(1, args.chunk_size)
    total_task_time = 0.0

    with multiprocessing.Pool(processes=num_workers, initializer=init_worker, initargs=(args.ckpt, device)) as pool:
        for res in pool.imap_unordered(solve_pair, tasks, chunksize=chunk_size):
            results.append(res)
            summary[res["status"]] = summary.get(res["status"], 0) + 1
            count += 1
            total_task_time += res.get("time", 0.0)
            if count % 50 == 0:
                avg_task_time = total_task_time / max(count, 1)
                throughput = 1.0 / max(avg_task_time, 1e-6)
                logger.info(
                    f"Processed {count}/{len(tasks)} ({count/len(tasks)*100:.1f}%) | "
                    f"avg task {avg_task_time:.3f}s | throughput ~{throughput:.2f} tasks/s"
                )

    logger.info(f"Finished processing {count} tasks. Summary: {summary}")
    logger.info(f"Saving to {args.output}...")
    
    with open(args.output, 'w') as f:
        for res in results:
            f.write(json.dumps(res) + "\n")

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True) # Safer for PyTorch
    main()

