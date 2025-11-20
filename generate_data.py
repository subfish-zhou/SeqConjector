import argparse, random, json, torch, math, glob, os, time, multiprocessing
from oeis.program import Node, Program
from oeis.interpreter import Interpreter, ExecConfig
from oeis.torch_model import stoi, TOKENS, cheap_features

# ==========================================
# 1. Real Sequence Pool (Reusable)
# ==========================================
class RealSequencePool:
    def __init__(self, data_dir="oeis_seq_labeled/formula_true", max_cache=10000):
        self.files = glob.glob(os.path.join(data_dir, "*.jsonl"))
        self.cache = []
        self.max_cache = max_cache
        if len(self.files) > 0:
            print(f"[Gen] Found {len(self.files)} source files in {data_dir}")
            self._fill_cache()
        else:
            print(f"[Gen] Warning: no source files found in {data_dir}")

    def _fill_cache(self):
        # Load a random subset of real sequences
        attempts = 0
        while len(self.cache) < self.max_cache and attempts < 100:
            attempts += 1
            fpath = random.choice(self.files)
            try:
                with open(fpath, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    # Sample up to 200 lines per file to spread diversity
                    sample_lines = random.sample(lines, min(len(lines), 200))
                    for line in sample_lines:
                        try:
                            entry = json.loads(line)
                            seq = entry.get("seq", [])
                            # Only keep reasonably long integer sequences
                            if len(seq) > 10 and all(isinstance(x, int) for x in seq):
                                self.cache.append(seq)
                        except: continue
            except: continue
            if len(self.cache) >= self.max_cache: break
        print(f"[Gen] Loaded {len(self.cache)} real sequences into cache.")

    def get_random(self, min_len=15):
        if not self.cache: return [i+1 for i in range(30)]
        for _ in range(10):
            seq = random.choice(self.cache)
            if len(seq) >= min_len:
                start = 0
                if len(seq) > min_len + 5 and random.random() < 0.4:
                    start = random.randint(0, len(seq) - min_len)
                return seq[start:start+random.randint(min_len, min(len(seq)-start, 100))]
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
        self.unary_trans = ["BINOM", "IBINOM", "EULER", "CONV", "POLY"]
        self.unary_idx = ["SHIFT", "REIDX", "SUBSAMPLE", "REPEAT", "DROP", "DROP_AT_2", "INSERT1", "INSERT2"]
        self.unary_nt = ["MAP_TAU", "MAP_SIGMA", "MAP_PHI", "MAP_MU", "MAP_OMEGA", "MAP_BIGOMEGA"]
        self.unary_pred = ["PRED_POS", "PRED_NEG", "PRED_IS_EVEN_N", "PRED_EQ_CONST", "PRED_GT_CONST", "PRED_LT_CONST", "PRED_NOT"]
        
        self.unary_ops = self.unary_arith + self.unary_scan + self.unary_trans + self.unary_idx + self.unary_nt + self.unary_pred

        # 2-arity (Binary)
        self.binary_ops = ["SEQ_ADD", "SEQ_SUB", "SEQ_MUL", "SEQ_DIV", "PRED_AND", "PRED_OR"]
        
        # 3-arity (Ternary)
        self.ternary_ops = ["COND"]

    def _random_args(self, op):
        # Helper to generate arguments for ops that require them
        if op in ["SCALE", "OFFSET"]: return [random.choice([-2,-1,2,3,4,5,10])]
        if op in ["MAP_MOD", "MAP_DIV"]: return [random.choice([2,3,4,5,10])]
        if op == "DIFF_FWD" or op == "DIFF_BACK": return [random.choice([1,1,1,2,3])]
        if op == "SHIFT": return [random.randint(1, 4)]
        if op == "SUBSAMPLE": return [random.randint(2, 4)]
        if op == "REPEAT": return [random.randint(2, 3)]
        if op == "DROP": return [random.randint(1, 5)]
        if op == "INSERT1" or op == "INSERT2": return [random.randint(-5, 5)]
        if op == "REIDX": return [random.choice([2,3]), random.choice([0,1])] # k, b
        if op == "CONV": 
            L = random.randint(1, 3)
            return [L] + [random.randint(-3, 3) for _ in range(L)]
        if op == "POLY":
            return [random.randint(-2,2) for _ in range(3)] # a,b,c
        if "CONST" in op: return [random.randint(0, 5)]
        return []

    def generate(self, max_depth=3, max_len=8):
        # Recursive generation with strict length budgeting
        self.nodes_count = 0
        
        def _gen(current_depth):
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
        
        root = _gen(0)
        return Program(root)


# ==========================================
# 3. Parallel Worker
# ==========================================
def worker_generate(job_id, num_samples, seed, out_file, moonshine_prob, difficulty):
    # Re-init random seed for this process
    random.seed(seed + job_id * 9999)
    
    pool = RealSequencePool()
    prog_gen = ProgramGenerator()
    inter = Interpreter(ExecConfig(strict=False, t0=20, t_step=5))
    
    # Difficulty config
    # difficulty 0.0 -> mostly short
    # difficulty 1.0 -> mixture
    # To pre-generate a curriculum, we can generate chunks with different difficulties
    # Or generate a balanced mix and filter later.
    # Let's assume we generate a mix based on the requested 'difficulty' parameter for this chunk.
    
    prob_long = 0.1 + 0.6 * difficulty 

    generated_count = 0
    data = []
    
    # Limit attempts to avoid infinite loops if generation is hard
    attempts = 0
    while generated_count < num_samples and attempts < num_samples * 5:
        attempts += 1
        
        # 1. Gen A
        if random.random() < 0.6:
            base = pool.get_random(min_len=15)
            if random.random() < 0.3: 
                k = random.randint(1, 3); b = random.randint(-5, 5)
                base = [x*k + b for x in base]
            A_full = base
        else:
            N = random.randint(15, 25)
            kind = random.choice(["nat", "squares", "randwalk", "const"])
            if kind=="nat": A_full = [i+1 for i in range(N)]
            elif kind=="squares": A_full = [(i+1)**2 for i in range(N)]
            elif kind=="const": c=random.randint(1,5); A_full = [c]*N
            else:
                cur=0; out=[]
                for _ in range(N):
                    cur+=random.randint(-3,3); out.append(cur)
                A_full = out
        
        if len(A_full) > 40: A_full = A_full[:40]

        # 2. Gen Program
        if random.random() < prob_long:
             max_d = random.randint(3, 6)
             max_l = random.randint(4, 8)
        else:
             max_d = random.randint(1, 3)
             max_l = random.randint(2, 4)
             
        P = prog_gen.generate(max_depth=max_d, max_len=max_l)
        
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
        
        n_in = random.randint(5, min(12, len(A_full)-1))
        toks = P.to_tokens()
        
        # Store minimal info
        # A, B, toks, is_moon
        item = {
            "A": A_full[:n_in],
            "B": B_full[:n_in],
            "toks": toks,
            "is_moon": is_moon
        }
        data.append(json.dumps(item))
        generated_count += 1

    # Write to file
    with open(out_file, 'w', encoding='utf-8') as f:
        for line in data:
            f.write(line + "\n")
    
    return generated_count

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
    
    pool_args = []
    for i in range(args.workers):
        out_file = os.path.join(args.out_dir, f"{args.prefix}_part_{i:03d}.jsonl")
        # worker_id, num_samples, seed, out_file, moonshine_prob, difficulty
        pool_args.append((i, samples_per_worker, args.seed, out_file, args.moonshine_prob, args.difficulty))
        
    print(f"Generating {args.total_samples} samples with {args.workers} workers...")
    print(f"Difficulty level: {args.difficulty}")
    
    t0 = time.time()
    with multiprocessing.Pool(args.workers) as p:
        results = p.starmap(worker_generate, pool_args)
        
    total = sum(results)
    dt = time.time() - t0
    print(f"Done. Generated {total} samples in {dt:.2f}s. Saved to {args.out_dir}")

if __name__ == "__main__":
    main()

