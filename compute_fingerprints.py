import json
import numpy as np
import math
import zlib
from collections import Counter

def berlekamp_massey(sequence, p):
    """
    Computes the length of the shortest linear recurrence for the sequence over GF(p).
    """
    N = len(sequence)
    if N == 0: return 0
    
    # Convert to field elements
    s = np.array(sequence) % p
    
    # Initialize
    b = np.zeros(N, dtype=int) # correction polynomial
    c = np.zeros(N, dtype=int) # connection polynomial
    b[0] = 1
    c[0] = 1
    
    L = 0
    m = -1
    d_prev = 1
    
    for n in range(N):
        # Calculate discrepancy
        d = s[n]
        for i in range(1, L + 1):
            d = (d + c[i] * s[n - i]) % p
            
        if d == 0:
            continue
            
        # If d != 0
        t = np.copy(c)
        # c(x) = c(x) - d * d_prev^-1 * x^(n-m) * b(x)
        
        # inverse of d_prev in GF(p) using fermat's little theorem: a^(p-2)
        inv_d_prev = pow(int(d_prev), p - 2, p)
        coef = (d * inv_d_prev) % p
        
        # shift b by (n-m)
        shift = n - m
        
        # We need to handle arrays carefully. 
        # c is size N. b is size N.
        # term to subtract is b shifted by 'shift' multiplied by coef
        
        # Create shifted b
        b_shifted = np.zeros(N, dtype=int)
        if shift < N:
            b_shifted[shift:] = b[:N-shift]
            
        c = (c - coef * b_shifted) % p
        
        if 2 * L <= n:
            L = n + 1 - L
            m = n
            b = t
            d_prev = d
            
    return int(L)

def lempel_ziv_complexity(binary_string):
    """
    Simple Lempel-Ziv complexity measure.
    Counts number of new patterns in the sequence.
    """
    n = len(binary_string)
    if n == 0: return 0
    
    i = 0
    c = 1
    u = 1
    v = 1
    v_max = v
    
    while u + v <= n:
        if u + v < n and binary_string[i + v] == binary_string[u + v]:
            v += 1
        else:
            v_max = max(v, v_max)
            i += 1
            if i == u:
                c += 1
                u += v_max
                v = 1
                i = 0
                v_max = v
            else:
                v = 1
                
    return c

def compute_sign_zero_stats(arr):
    if len(arr) == 0:
        return {}
    
    pos = np.sum(arr > 0)
    neg = np.sum(arr < 0)
    zeros = np.sum(arr == 0)
    total = len(arr)
    
    # Sign changes
    signs = np.sign(arr)
    # filter out zeros for sign changes?? Usually strictly different sign.
    # User said: sign(a_n) != sign(a_{n+1}). sign(0)=0. 
    # So 1 -> 0 is a change. 0 -> 1 is a change.
    if total > 1:
        changes = np.sum(signs[:-1] != signs[1:])
    else:
        changes = 0
        
    # Zero structure
    zero_indices = np.where(arr == 0)[0]
    has_zeros = len(zero_indices) > 0
    first_zero = int(zero_indices[0]) if has_zeros else -1
    
    zero_intervals_mean = 0.0
    if len(zero_indices) > 1:
        intervals = np.diff(zero_indices)
        zero_intervals_mean = float(np.mean(intervals))
        
    return {
        "p_pos": float(pos / total),
        "p_neg": float(neg / total),
        "p_zero": float(zeros / total),
        "sign_changes": int(changes),
        "has_zeros": bool(has_zeros),
        "first_zero_idx": first_zero,
        "zero_intervals_mean": zero_intervals_mean
    }

def compute_residue_stats(arr, mods=[2, 3, 4, 5, 7, 8, 9, 11]):
    if len(arr) == 0: return {}
    stats = {}
    
    for m in mods:
        rems = arr % m
        counts = Counter(rems)
        # Normalize
        total = len(rems)
        dist = {r: c/total for r, c in counts.items()}
        
        # Missing residues
        present = set(counts.keys())
        missing = [r for r in range(m) if r not in present]
        
        # Entropy might be a good scalar summary
        entropy = -sum(p * math.log(p) for p in dist.values())
        
        stats[f"mod_{m}_entropy"] = entropy
        stats[f"mod_{m}_missing_count"] = len(missing)
        # Maybe store the most frequent residue and its freq?
        most_common = counts.most_common(1)
        if most_common:
            stats[f"mod_{m}_max_freq"] = float(most_common[0][1] / total)
        
    return stats

def compute_diff_stats(arr):
    if len(arr) < 2: return {}
    
    # 1st diff
    d1 = np.diff(arr)
    mean_d1 = float(np.mean(d1))
    var_d1 = float(np.var(d1))
    
    signs_d1 = np.sign(d1)
    pos_d1 = np.sum(signs_d1 > 0) / len(d1)
    neg_d1 = np.sum(signs_d1 < 0) / len(d1)
    
    # Simple periodicity check (very rough) - check if d1 repeats a small pattern
    # Just checking if d1 is constant
    d1_is_const = bool(np.all(d1 == d1[0]))
    
    # 2nd diff
    d2 = np.diff(d1)
    if len(d2) > 0:
        var_d2 = float(np.var(d2))
        d2_is_const = bool(np.all(d2 == d2[0]))
    else:
        var_d2 = 0.0
        d2_is_const = True # trivially
        
    return {
        "d1_mean": mean_d1,
        "d1_var": var_d1,
        "d1_pos_frac": float(pos_d1),
        "d1_neg_frac": float(neg_d1),
        "d1_is_const": d1_is_const,
        "d2_var": var_d2,
        "d2_is_const": d2_is_const
    }

def compute_growth_stats(arr):
    # Avoid log(0) and negative issues
    abs_arr = np.abs(arr)
    n_vals = np.arange(1, len(arr) + 1)
    
    # g1 = log(|a_n|+1) / log(n)
    # We exclude n=1 (log(1)=0) if we divide
    mask = n_vals > 1
    
    g1_vals = np.zeros_like(abs_arr, dtype=float)
    if np.any(mask):
        numerator = np.log(abs_arr[mask] + 1)
        denominator = np.log(n_vals[mask])
        g1_vals[mask] = numerator / denominator
        
    # g2 = log(log(|a_n|+e)) / log(n)
    g2_vals = np.zeros_like(abs_arr, dtype=float)
    if np.any(mask):
        numerator = np.log(np.log(abs_arr[mask] + math.e))
        denominator = np.log(n_vals[mask])
        g2_vals[mask] = numerator / denominator
        
    # Sample few points
    indices = [19, 49, 99, 199] # 0-based for n=20, 50, 100, 200
    valid_indices = [i for i in indices if i < len(arr)]
    
    g1_samples = [float(g1_vals[i]) for i in valid_indices]
    g2_samples = [float(g2_vals[i]) for i in valid_indices]
    
    # Linear fit log(|a_n|) vs log(n) -> Power law exponent
    # Filter out zeros for log
    pos_mask = abs_arr > 0
    poly_exponent = 0.0
    exp_exponent = 0.0
    
    if np.sum(pos_mask) > 5:
        y = np.log(abs_arr[pos_mask])
        x = np.log(n_vals[pos_mask])
        if np.std(x) > 1e-6:
            slope, _ = np.polyfit(x, y, 1)
            poly_exponent = slope
            
        # Linear fit log(|a_n|) vs n -> Exponential base
        x_lin = n_vals[pos_mask]
        if np.std(x_lin) > 1e-6:
            slope, _ = np.polyfit(x_lin, y, 1)
            exp_exponent = slope # This is ln(alpha)
            
    return {
        "g1_samples": g1_samples,
        "g2_samples": g2_samples,
        "est_poly_exponent": float(poly_exponent),
        "est_exp_exponent": float(exp_exponent)
    }

def compute_fitting_stats(arr_float, arr_obj):
    n = len(arr_float)
    if n < 5: return {}
    
    x = np.arange(n)
    y = arr_float
    
    # Poly fit errors (RMSE)
    fits = {}
    
    # Filter out NaNs or Infs for fitting
    mask = np.isfinite(y)
    if np.sum(mask) > 5:
        x_fit = x[mask]
        y_fit = y[mask]
        
        for deg in [1, 2, 3]:
            try:
                coeffs = np.polyfit(x_fit, y_fit, deg)
                p = np.poly1d(coeffs)
                y_pred = p(x_fit)
                rmse = np.sqrt(np.mean((y_fit - y_pred)**2))
                fits[f"poly_deg{deg}_rmse"] = float(rmse)
            except:
                fits[f"poly_deg{deg}_rmse"] = -1.0
    else:
         for deg in [1, 2, 3]:
             fits[f"poly_deg{deg}_rmse"] = -1.0
            
    # BM Linear Complexity
    # Use integer array (arr_obj)
    bm_stats = {}
    # Ensure we pass python ints to BM
    # arr_obj might contain python ints
    for p in [2, 3, 5, 7]:
        try:
            L = berlekamp_massey(arr_obj, p)
            bm_stats[f"bm_L_{p}"] = L
        except:
            bm_stats[f"bm_L_{p}"] = -1
            
    return {**fits, **bm_stats}

def compute_complexity_stats(arr):
    if len(arr) == 0: return {}
    
    # Mod 2 complexity
    bin_seq = arr % 2
    bin_str = "".join(map(str, bin_seq))
    
    # Lempel-Ziv (rough)
    # using a python implementation or just compression as proxy
    # User asked for LZ or compression.
    
    # zlib compression ratio
    # convert to bytes
    data_bytes = str(arr.tolist()).encode('utf-8')
    compressed = zlib.compress(data_bytes)
    ratio = len(compressed) / len(data_bytes)
    
    # LZ on binary string
    # This is a custom implementation above
    # Use 0/1 string
    lz_val = lempel_ziv_complexity(bin_str)
    lz_norm = lz_val / (len(bin_str) / math.log2(len(bin_str) + 1)) if len(bin_str) > 1 else 0
    
    return {
        "zlib_ratio": float(ratio),
        "lz_complexity_mod2": int(lz_val),
        "lz_norm_mod2": float(lz_norm)
    }

def process_file(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as f_in, \
         open(output_path, 'w', encoding='utf-8') as f_out:
        
        for line in f_in:
            if not line.strip(): continue
            try:
                data = json.loads(line)
                seq = data.get("seq", [])
                if not isinstance(seq, list):
                    f_out.write(line)
                    continue
                    
                # Truncate if too long
                seq = seq[:1000]
                
                # Create object array for exact integer arithmetic (modulo)
                seq_arr_obj = np.array(seq, dtype=object)
                
                # Create float array for stats/fitting
                # Handle potential overflow for floats
                try:
                    seq_arr_float = np.array(seq, dtype=float)
                    # Replace Infs with NaN or huge number?
                    # Just keep them, check for validity later
                except:
                    # Fallback for extremely large numbers
                    seq_arr_float = np.array([float(x) if abs(x) < 1e300 else (1e300 if x > 0 else -1e300) for x in seq])

                fingerprints = {}
                
                # 1. Sign & Zero (use float or obj is fine, signs are same)
                fingerprints.update(compute_sign_zero_stats(seq_arr_float))
                
                # 2. Residue (use object array for correctness with big ints)
                fingerprints.update(compute_residue_stats(seq_arr_obj))
                
                # 3. Difference (use float for mean/var)
                fingerprints.update(compute_diff_stats(seq_arr_float))
                
                # 4. Growth (use float)
                fingerprints.update(compute_growth_stats(seq_arr_float))
                
                # 5. Fitting (use float)
                # For BM, we need integers. Use seq (list) or seq_arr_obj
                fingerprints.update(compute_fitting_stats(seq_arr_float, seq_arr_obj))
                
                # 6. Complexity (use obj/list)
                fingerprints.update(compute_complexity_stats(seq_arr_obj))
                
                data["fingerprints"] = fingerprints
                f_out.write(json.dumps(data) + "\n")
                
            except Exception as e:
                # In case of error, write original line? Or skip?
                # Write original line with error log maybe
                print(f"Error processing line: {e}")
                f_out.write(line)

if __name__ == "__main__":
    import sys
    import os
    import shutil
    import glob

    if len(sys.argv) < 2:
        print("Usage: python compute_fingerprints.py <file_or_dir> [--inplace]")
        sys.exit(1)

    target = sys.argv[1]
    inplace = "--inplace" in sys.argv

    if os.path.isfile(target):
        files = [target]
    elif os.path.isdir(target):
        files = glob.glob(os.path.join(target, "**", "*.jsonl"), recursive=True)
    else:
        print(f"Target {target} not found.")
        sys.exit(1)

    for f in files:
        print(f"Processing {f}...")
        temp_out = f + ".tmp"
        try:
            process_file(f, temp_out)
            if inplace:
                shutil.move(temp_out, f)
                print(f"Updated {f}")
            else:
                print(f"Written to {temp_out} (use --inplace to overwrite)")
        except Exception as e:
            print(f"Failed to process {f}: {e}")
            if os.path.exists(temp_out):
                os.remove(temp_out)

