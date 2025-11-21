import argparse, random, json, math, glob, os, time, multiprocessing
from oeis.program import Node, Program
from oeis.config import Config
from oeis.split_utils import compute_split

# ==========================================
# 1. Real Sequence Pool (On-demand)
# ==========================================
class RealSequencePool:
    def __init__(self, data_dir="oeis_seq_labeled/formula_true"):
        self.files = glob.glob(os.path.join(data_dir, "*.jsonl"))
        if len(self.files) == 0:
            print(f"Warning: No OEIS files found in {data_dir}")
        # Pre-compute file sizes for weighted sampling
        self.file_lines = []
        for fpath in self.files:
            try:
                with open(fpath, 'r', encoding='utf-8') as f:
                    num_lines = sum(1 for _ in f)
                    self.file_lines.append((fpath, num_lines))
            except:
                continue

    def get_random(self, min_len=7, max_attempts=20):
        """Extract a random sequence on-demand without caching"""
        if not self.file_lines:
            return [i+1 for i in range(30)]  # Fallback
        
        for _ in range(max_attempts):
            # Pick a random file
            fpath, num_lines = random.choice(self.file_lines)
            
            # Pick a random line number
            target_line = random.randint(0, num_lines - 1)
            
            try:
                with open(fpath, 'r', encoding='utf-8') as f:
                    # Skip to target line
                    for i, line in enumerate(f):
                        if i == target_line:
                            try:
                                entry = json.loads(line)
                                seq = entry.get("seq", [])
                                
                                # Filter: min length + no huge values (no max length limit)
                                if (len(seq) >= min_len and 
                                    all(isinstance(x, int) and abs(x) < 1e12 for x in seq)):
                                    
                                    # Optionally extract a substring (for very long sequences)
                                    if len(seq) > 50 and random.random() < 0.3:
                                        start = random.randint(0, len(seq) - min_len)
                                        end = start + random.randint(min_len, min(len(seq) - start, 100))
                                        return seq[start:end]
                                    else:
                                        return seq  # Return full sequence (no truncation)
                            except:
                                break
                            break
            except:
                continue
        
        # Fallback if all attempts failed
        return [i+1 for i in range(min_len)]

# ==========================================
# 2. Program Generator (New DSL)
# ==========================================
class ProgramGenerator:
    def __init__(self):
        # Group ops by arity (number of children nodes)
        # 0-arity (Leaves)
        self.terminals = ["A"] 
        
        # 1-arity (Unary)
        self.unary_arith = ["SCALE", "OFFSET", "MAP_ABS", "MAP_SGN", "MAP_MOD", "MAP_DIV", "MAP_SQRT"]
        self.unary_scan = ["SCAN_ADD", "SCAN_MUL", "DIFF_FWD", "DIFF_BACK"]
        self.unary_trans = ["CONV_FWD", "CONV_BACK", "POLY"]  # Removed BINOM, IBINOM, EULER (too slow)
        self.unary_idx = ["SHIFT", "REIDX", "SUBSAMPLE", "REPEAT", "DROP", "DROP_AT_2", "INSERT1", "INSERT2"]
        # Removed all number theory functions (MAP_TAU, MAP_SIGMA, etc.) - too slow on large integers
        self.unary_pred = ["PRED_POS", "PRED_NEG", "PRED_IS_EVEN_N", "PRED_EQ_CONST", "PRED_GT_CONST", "PRED_LT_CONST", "PRED_NOT"]
        
        self.unary_ops = self.unary_arith + self.unary_scan + self.unary_trans + self.unary_idx + self.unary_pred

        # 2-arity (Binary)
        self.binary_ops = ["SEQ_ADD", "SEQ_SUB", "SEQ_MUL", "SEQ_DIV", "PRED_AND", "PRED_OR"]
        
        # 3-arity (Ternary)
        self.ternary_ops = ["COND"]

    def _random_args(self, op):
        # Helper to generate arguments for ops that require them
        if op in ["SCALE", "OFFSET"]: return [random.choice([-2,-1,2,3,4,5,10])]
        if op in ["MAP_MOD", "MAP_DIV"]: return [random.choice([2,3,4,5,10])]
        if op in ["DIFF_FWD", "DIFF_BACK", "CONV_FWD", "CONV_BACK"]: return [random.choice([1,1,1,2,3])]
        if op == "SHIFT": return [random.randint(1, 4)]
        if op == "SUBSAMPLE": return [random.randint(2, 4)]
        if op == "REPEAT": return [random.randint(2, 3)]
        if op == "DROP": return [random.randint(1, 5)]
        if op == "INSERT1" or op == "INSERT2": return [random.randint(-5, 5)]
        if op == "REIDX": return [random.choice([2,3]), random.choice([0,1])] # k, b
        if op == "POLY":
            return [random.randint(-2,2) for _ in range(3)] # a,b,c
        if "CONST" in op: return [random.randint(0, 5)]
        return []

    def generate(self, max_depth=3, max_len=8):
        # Recursive generation with strict length budgeting
        self.nodes_count = 0
        
        def _gen(current_depth, force_op=False):
            # If max depth reached or max len nearly reached, return terminal
            if current_depth >= max_depth or self.nodes_count >= max_len - 1:
                self.nodes_count += 1
                return Node("A")
            
            # Decide branching factor based on remaining budget
            # If we have little budget, prefer unary
            remaining = max_len - self.nodes_count
            
            valid_arities = [0]
            if remaining >= 2: valid_arities.append(1)
            if remaining >= 3: valid_arities.append(2)
            if remaining >= 4: valid_arities.append(3)
            
            # Bias towards growth if depth is low
            if current_depth < 2 and remaining > 4:
                weights = [0.1, 0.6, 0.2, 0.1] # 0, 1, 2, 3
            elif current_depth < max_depth - 1 and remaining > 2:
                weights = [0.3, 0.5, 0.2, 0.0]
            else:
                weights = [0.8, 0.2, 0.0, 0.0]
            
            # Force at least one operation at root (avoid bare "A")
            if force_op:
                weights[0] = 0.0  # Disable arity 0 (terminal)
                
            # Normalize weights for valid arities
            probs = [weights[i] if i in valid_arities else 0 for i in range(4)]
            total = sum(probs)
            if total == 0: arity = 0
            else: arity = random.choices([0,1,2,3], weights=probs)[0]
            
            self.nodes_count += 1
            
            if arity == 0:
                return Node("A")
            elif arity == 1:
                op = random.choice(self.unary_ops)
                return Node(op, self._random_args(op), [_gen(current_depth+1)])
            elif arity == 2:
                op = random.choice(self.binary_ops)
                return Node(op, [], [_gen(current_depth+1), _gen(current_depth+1)])
            elif arity == 3:
                op = "COND"
                return Node(op, [], [_gen(current_depth+1), _gen(current_depth+1), _gen(current_depth+1)])
        
        # With 99.99% probability, force at least one operation
        force_meaningful = (random.random() > 0.0001)
        root = _gen(0, force_op=force_meaningful)
        return Program(root)


# ==========================================
# 3. Parallel Worker
# ==========================================
def worker_generate(job_id, num_samples, seed, out_file, moonshine_prob, difficulty, queue):
    try:
        # Re-init random seed for this process
        random.seed(seed + job_id * 9999)
        
        # Re-instantiate locally to avoid sharing issues
        pool = RealSequencePool()
        prog_gen = ProgramGenerator()
        # Use loose budget for data generation (allows more complex programs)
        inter_config = Config.get_interpreter_config(strict=False, loose_budget=True)
        from oeis.interpreter import Interpreter
        inter = Interpreter(inter_config)
        
        prob_long = 0.1 + 0.6 * difficulty 

        generated_count = 0
        buffer = []
        
        # Increase attempt limit significantly to avoid early exit on hard difficulties
        attempts = 0
        max_attempts = num_samples * 50  # Reduce from 100 to avoid extreme cases 
        
        with open(out_file, 'w', encoding='utf-8') as f:
            while generated_count < num_samples and attempts < max_attempts:
                attempts += 1
                
                # 1. Gen A
                if random.random() < 0.6:
                    # Extract from real OEIS data (min_len=7, no upper limit)
                    base = pool.get_random(min_len=7)
                    if random.random() < 0.3: 
                        k = random.randint(1, 3); b = random.randint(-5, 5)
                        base = [x*k + b for x in base]
                    A_full = base
                else:
                    # Synthesize simple sequence (length 7-30)
                    N = random.randint(7, 30)
                    kind = random.choice(["nat", "squares", "randwalk", "const"])
                    if kind=="nat": A_full = [i+1 for i in range(N)]
                    elif kind=="squares": A_full = [(i+1)**2 for i in range(N)]
                    elif kind=="const": c=random.randint(1,5); A_full = [c]*N
                    else:
                        cur=0; out=[]
                        for _ in range(N):
                            cur+=random.randint(-3,3); out.append(cur)
                        A_full = out
                
                # No truncation - allow full length sequences

                # 2. Gen Program
                if random.random() < prob_long:
                     max_d = random.randint(3, 6)
                     max_l = random.randint(4, 8)
                else:
                     max_d = random.randint(1, 3)
                     max_l = random.randint(2, 4)
                     
                P = prog_gen.generate(max_depth=max_d, max_len=max_l)
                
                # Check that program contains A (required by system design)
                if not P.contains_A():
                    continue
                
                # 3. Execute
                r = inter.execute(P, A_full)
                if (not r.ok) or r.seq is None or len(r.seq) < 10:
                    continue

                B_full = r.seq
                
                # 4. Moonshine
                is_moon = (random.random() < moonshine_prob)
                if is_moon and len(B_full) >= 12:
                    s = random.randint(4, 8)
                    gamma = random.uniform(1e-3, 5e-3)
                    for t in range(s, len(B_full)):
                        try:
                            fac = math.exp(gamma * (t - s))
                            val = B_full[t] * fac
                            if abs(val) < 1e18:
                                B_full[t] = int(round(val))
                        except: pass
                
                # Determine train/validation split using unified rule
                n_in, n_validate = compute_split(len(A_full))
                
                toks = P.to_tokens()
                
                # Store minimal info
                item = {
                    "A": A_full[:n_in],
                    "B": B_full[:n_in],
                    "toks": toks,
                    "is_moon": is_moon
                }
                buffer.append(json.dumps(item))
                generated_count += 1
                
                # Flush buffer every 200 items (reduce I/O and queue overhead)
                if len(buffer) >= 200:
                    f.write("\n".join(buffer) + "\n")
                    f.flush()  # Ensure data is written
                    queue.put(len(buffer))
                    buffer = []

            # Flush remaining
            if buffer:
                f.write("\n".join(buffer) + "\n")
                f.flush()
                queue.put(len(buffer))
        
        return generated_count
    except Exception as e:
        print(f"Worker {job_id} failed: {e}")
        return 0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="data_gen")
    ap.add_argument("--total_samples", type=int, default=200000)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--difficulty", type=float, default=0.5, help="0.0=easy(short), 1.0=hard(mixed)")
    ap.add_argument("--moonshine_prob", type=float, default=0.1)
    ap.add_argument("--prefix", default="data", help="Prefix for output filenames")
    ap.add_argument("--seed", type=int, default=0, help="Random seed base")
    args = ap.parse_args()
    
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
        
    samples_per_worker = args.total_samples // args.workers
    
    # Use Manager Queue for progress tracking
    manager = multiprocessing.Manager()
    queue = manager.Queue()
    
    pool_args = []
    for i in range(args.workers):
        out_file = os.path.join(args.out_dir, f"{args.prefix}_part_{i:03d}.jsonl")
        pool_args.append((i, samples_per_worker, args.seed, out_file, args.moonshine_prob, args.difficulty, queue))
        
    print(f"Generating {args.total_samples} samples with {args.workers} workers...")
    print(f"Difficulty level: {args.difficulty}")
    
    t0 = time.time()
    
    import tqdm
    
    # Start pool asynchronously
    with multiprocessing.Pool(args.workers) as p:
        # Use map_async so we can monitor queue while workers run
        result_async = p.starmap_async(worker_generate, pool_args)
        
        # Monitor queue with reduced update frequency
        with tqdm.tqdm(total=args.total_samples, unit="sample", miniters=100, mininterval=0.5) as pbar:
            accumulated = 0
            while not result_async.ready():
                # Read from queue
                try:
                    # Drain queue and batch update
                    batch_count = 0
                    while not queue.empty():
                        n = queue.get_nowait()
                        batch_count += n
                    if batch_count > 0:
                        pbar.update(batch_count)
                        accumulated += batch_count
                except: pass
                time.sleep(0.5)  # Check less frequently
            
            # Final drain
            batch_count = 0
            while not queue.empty():
                n = queue.get_nowait()
                batch_count += n
            if batch_count > 0:
                pbar.update(batch_count)
                
        # Get final results (to propagate exceptions if any)
        results = result_async.get()
        
    total_gen = sum(results)
    dt = time.time() - t0
    print(f"Done. Generated {total_gen} samples in {dt:.2f}s. Saved to {args.out_dir}")


if __name__ == "__main__":
    main()

