
import torch, torch.nn as nn
from dataclasses import dataclass
from typing import List
import numpy as np
import warnings
warnings.filterwarnings('ignore')

OPS = [
    "A",
    "SCALE","OFFSET","MAP_ABS","MAP_SGN","MAP_MOD","MAP_DIV","MAP_SQRT",
    "SEQ_ADD","SEQ_SUB","SEQ_MUL","SEQ_DIV",
    "SCAN_ADD","SCAN_MUL","DIFF_FWD","DIFF_BACK",
    "CONV_FWD","CONV_BACK","POLY",
    "SHIFT","REIDX","SUBSAMPLE","REPEAT","DROP","DROP_AT_2","INSERT1","INSERT2",
    "PRED_POS","PRED_NEG","PRED_IS_EVEN_N","PRED_EQ_CONST","PRED_GT_CONST","PRED_LT_CONST",
    "PRED_NOT","PRED_AND","PRED_OR","COND",
    "<BOS>","<EOS>"
]
INTS = list(range(-16,17))
TOKENS = OPS + [str(x) for x in INTS]
stoi = {t:i for i,t in enumerate(TOKENS)}
itos = {i:t for t,i in stoi.items()}


# =============================================================================
# Statistical utility functions (to avoid scipy dependency)
# =============================================================================

def _skew(x):
    """Compute skewness"""
    x_mean = np.mean(x)
    x_std = np.std(x)
    if x_std < 1e-12:
        return 0.0
    return np.mean(((x - x_mean) / x_std) ** 3)

def _kurtosis(x):
    """Compute excess kurtosis"""
    x_mean = np.mean(x)
    x_std = np.std(x)
    if x_std < 1e-12:
        return 0.0
    return np.mean(((x - x_mean) / x_std) ** 4) - 3

def _spearmanr(x, y):
    """Spearman rank correlation coefficient"""
    rank_x = np.argsort(np.argsort(x))
    rank_y = np.argsort(np.argsort(y))
    return np.corrcoef(rank_x, rank_y)[0, 1]

def _kendalltau(x, y):
    """Kendall tau correlation coefficient (simplified version)"""
    n = len(x)
    concordant = 0
    discordant = 0
    for i in range(n):
        for j in range(i+1, n):
            if (x[i] - x[j]) * (y[i] - y[j]) > 0:
                concordant += 1
            elif (x[i] - x[j]) * (y[i] - y[j]) < 0:
                discordant += 1
    tau = (concordant - discordant) / (concordant + discordant + 1e-12)
    return tau


# =============================================================================
# Feature extraction
# =============================================================================

def _safe_ratio(x, y, default=0.0):
    """Safely compute ratio with clipping"""
    if abs(y) < 1e-12:
        return default
    return float(np.clip(x / y, -100, 100))

def _safe_log(x, default=0.0):
    """Safely compute log10 with clipping"""
    if x <= 0:
        return default
    return float(np.clip(np.log10(abs(x) + 1e-12), -10, 10))

def _r_squared(x, y):
    """Compute R-squared coefficient of determination"""
    try:
        if len(x) < 2:
            return 0.0
        corr = np.corrcoef(x, y)[0, 1]
        if np.isnan(corr):
            return 0.0
        return float(corr ** 2)
    except:
        return 0.0

def _autocorrelation(x, max_lag):
    """Compute autocorrelation"""
    result = []
    for lag in range(max_lag):
        if lag < len(x):
            c = np.corrcoef(x[:-lag or None], x[lag:])[0, 1]
            result.append(c if not np.isnan(c) else 0.0)
        else:
            result.append(0.0)
    return result

def _mutual_information(x, y, bins=10):
    """Compute mutual information"""
    try:
        x_discrete = np.digitize(x, np.linspace(np.min(x), np.max(x), bins))
        y_discrete = np.digitize(y, np.linspace(np.min(y), np.max(y), bins))
        
        xy_hist = np.histogram2d(x_discrete, y_discrete, bins=bins)[0]
        xy_hist = xy_hist / np.sum(xy_hist)
        
        x_hist = np.sum(xy_hist, axis=1)
        y_hist = np.sum(xy_hist, axis=0)
        
        mi = 0.0
        for i in range(bins):
            for j in range(bins):
                if xy_hist[i, j] > 0:
                    mi += xy_hist[i, j] * np.log(xy_hist[i, j] / (x_hist[i] * y_hist[j] + 1e-12))
        
        return float(np.clip(mi, 0, 10))
    except:
        return 0.0

def _dtw_distance(x, y):
    """Simplified Dynamic Time Warping distance"""
    n, m = len(x), len(y)
    if n == 0 or m == 0 or n > 20 or m > 20:
        return 0.0
    
    dtw = np.full((n+1, m+1), np.inf)
    dtw[0, 0] = 0
    
    for i in range(1, n+1):
        for j in range(1, m+1):
            cost = abs(x[i-1] - y[j-1])
            dtw[i, j] = cost + min(dtw[i-1, j], dtw[i, j-1], dtw[i-1, j-1])
    
    normalized = float(dtw[n, m] / (n + m))
    return np.clip(normalized / (np.std(x) + np.std(y) + 1e-12), 0, 10)


def enhanced_features(A: List[int], B: List[int]) -> torch.Tensor:
    """
    Extract enhanced 54-dimensional feature vector from sequences A and B.
    
    Feature allocation:
    - Basic statistics (8): length ratio, mean ratio, variance ratio, max ratio, range ratio, skewness diff, kurtosis diff, median ratio
    - Linear fit (4): log_k, k_raw, log_c, c_raw (solves clipping issue)
    - Nonlinear indicators (8): r2_square, r2_cube, r2_exp, r2_log, power exponent, nonlinearity, monotonicity, convexity
    - Periodicity (6): main frequency, frequency strength, period length, periodicity score, phase, num harmonics
    - Correlation (12): Pearson, Spearman, Kendall, mutual info, 6 offset correlations, DTW distance, cosine similarity
    - Segmented features (8): corr_q1, corr_h1, corr_h2, corr_q4, segment consistency, num jumps, local variance, slope diff
    - Special patterns (8): is_scan, is_diff, is_scale2, is_mod_like, is_polynomial, is_exponential, is_periodic, is_piecewise
    
    Total: 54 dimensions
    """
    if len(A) == 0 or len(B) == 0:
        return torch.zeros(54, dtype=torch.float32)
    
    a_full = np.array(A, dtype=float)
    b_full = np.array(B, dtype=float)
    
    min_len = min(len(A), len(B))
    a = a_full[:min_len]
    b = b_full[:min_len]
    
    features = []
    
    # 1. Basic statistics (8 dims)
    len_ratio = _safe_ratio(len(b_full), len(a_full), 1.0)
    features.append(len_ratio)
    
    mean_ratio = _safe_ratio(np.mean(b), np.mean(a), 0.0)
    features.append(mean_ratio)
    
    var_ratio = _safe_ratio(np.var(b), np.var(a), 0.0)
    features.append(_safe_log(var_ratio))
    
    max_ratio = _safe_ratio(np.max(b), np.max(a) if np.max(a) != 0 else 1, 0.0)
    features.append(_safe_log(max_ratio))
    
    range_ratio = _safe_ratio(np.ptp(b), np.ptp(a), 0.0)
    features.append(_safe_log(range_ratio))
    
    try:
        skew_diff = float(_skew(b) - _skew(a))
        kurt_diff = float(_kurtosis(b) - _kurtosis(a))
    except:
        skew_diff = 0.0
        kurt_diff = 0.0
    features.append(np.clip(skew_diff, -10, 10))
    features.append(np.clip(kurt_diff, -10, 10))
    
    median_ratio = _safe_ratio(np.median(b), np.median(a), 0.0)
    features.append(median_ratio)
    
    # 2. Linear fit (4 dims) - solves clipping issue
    if len(a) < 2 or np.all(a == a[0]):
        features.extend([0.0, 0.0, 0.0, 0.0])
    else:
        try:
            A_matrix = np.vstack([a, np.ones(len(a))]).T
            k, c = np.linalg.lstsq(A_matrix, b, rcond=None)[0]
            
            log_k = _safe_log(abs(k)) * np.sign(k)
            k_clipped = float(np.clip(k, -100, 100))
            log_c = _safe_log(abs(c)) * np.sign(c)
            c_clipped = float(np.clip(c, -1000, 1000))
            
            features.extend([log_k, k_clipped, log_c, c_clipped])
        except:
            features.extend([0.0, 0.0, 0.0, 0.0])
    
    # 3. Nonlinear indicators (8 dims)
    if len(a) < 3:
        features.extend([0.0] * 8)
    else:
        try:
            r2_square = _r_squared(a ** 2, b)
            r2_cube = _r_squared(a ** 3, b)
            
            r2_exp = _r_squared(a, np.log(b + 1e-12)) if np.all(b > 0) else 0.0
            r2_log = _r_squared(np.log(a + 1e-12), b) if np.all(a > 0) else 0.0
            
            if np.all(a > 0) and np.all(b > 0):
                try:
                    log_a = np.log(a + 1e-12)
                    log_b = np.log(b + 1e-12)
                    slope, _ = np.polyfit(log_a, log_b, 1)
                    power_exp = float(np.clip(slope, -10, 10))
                except:
                    power_exp = 0.0
            else:
                power_exp = 0.0
            
            r2_linear = _r_squared(a, b)
            nonlinearity = max(0, r2_square - r2_linear)
            
            monotonicity = 0.0
            if len(a) >= 3:
                try:
                    monotonicity = float(_spearmanr(a, b))
                    if np.isnan(monotonicity):
                        monotonicity = 0.0
                except:
                    pass
            
            if len(b) >= 3:
                diff2 = np.diff(np.diff(b))
                convexity = float(np.mean(diff2) / (np.std(diff2) + 1e-12))
                convexity = np.clip(convexity, -10, 10)
            else:
                convexity = 0.0
            
            features.extend([r2_square, r2_cube, r2_exp, r2_log, power_exp, nonlinearity, monotonicity, convexity])
        except:
            features.extend([0.0] * 8)
    
    # 4. Periodicity detection (6 dims)
    if len(b) < 8:
        features.extend([0.0] * 6)
    else:
        try:
            fft_b = np.fft.fft(b - np.mean(b))
            power = np.abs(fft_b[:len(b)//2])
            freqs = np.fft.fftfreq(len(b))[:len(b)//2]
            
            if len(power) > 1:
                main_freq_idx = np.argmax(power[1:]) + 1
                main_freq = float(freqs[main_freq_idx])
                freq_strength = float(power[main_freq_idx] / (np.sum(power) + 1e-12))
            else:
                main_freq = 0.0
                freq_strength = 0.0
            
            period_length = float(1.0 / main_freq) if main_freq > 1e-6 else 0.0
            period_length = np.clip(period_length, 0, len(b))
            
            if len(b) >= 4:
                acf = _autocorrelation(b - np.mean(b), max_lag=min(len(b)//2, 10))
                periodicity_score = float(np.max(acf[1:]) if len(acf) > 1 else 0.0)
            else:
                periodicity_score = 0.0
            
            phase = float(np.angle(fft_b[main_freq_idx]) if main_freq_idx < len(fft_b) else 0.0)
            phase = phase / np.pi
            
            threshold = np.max(power) * 0.1
            num_harmonics = float(np.sum(power > threshold)) / len(power)
            
            features.extend([main_freq, freq_strength, period_length, periodicity_score, phase, num_harmonics])
        except:
            features.extend([0.0] * 6)
    
    # 5. Correlation indicators (12 dims)
    if len(a) < 2:
        features.extend([0.0] * 12)
    else:
        try:
            pearson = float(np.corrcoef(a, b)[0, 1])
            features.append(0.0 if np.isnan(pearson) else pearson)
            
            try:
                spearman_val = float(_spearmanr(a, b))
            except:
                spearman_val = 0.0
            features.append(0.0 if np.isnan(spearman_val) else spearman_val)
            
            if len(a) <= 20:
                try:
                    kendall_val = float(_kendalltau(a, b))
                except:
                    kendall_val = 0.0
            else:
                kendall_val = 0.0
            features.append(0.0 if np.isnan(kendall_val) else kendall_val)
            
            try:
                mi = _mutual_information(a, b)
            except:
                mi = 0.0
            features.append(mi)
            
            for d in range(6):
                if d >= len(a) or (len(a) - d) < 2:
                    features.append(0.0)
                else:
                    va = a[d:]
                    vb = b[d:]
                    c = np.corrcoef(va, vb)[0, 1]
                    features.append(0.0 if np.isnan(c) else float(c))
            
            dtw_dist = _dtw_distance(a[:20], b[:20])
            features.append(dtw_dist)
            
            cos_sim = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
            features.append(cos_sim)
        except:
            features.extend([0.0] * 12)
    
    # 6. Segmented features (8 dims)
    if len(a) < 4:
        features.extend([0.0] * 8)
    else:
        try:
            n = len(a)
            q1 = n // 4
            h1 = n // 2
            q3 = 3 * n // 4
            
            corr_q1 = float(np.corrcoef(a[:q1], b[:q1])[0, 1] if q1 > 1 else 0.0)
            features.append(0.0 if np.isnan(corr_q1) else corr_q1)
            
            corr_h1 = float(np.corrcoef(a[:h1], b[:h1])[0, 1] if h1 > 1 else 0.0)
            features.append(0.0 if np.isnan(corr_h1) else corr_h1)
            
            corr_h2 = float(np.corrcoef(a[h1:], b[h1:])[0, 1] if n-h1 > 1 else 0.0)
            features.append(0.0 if np.isnan(corr_h2) else corr_h2)
            
            corr_q4 = float(np.corrcoef(a[q3:], b[q3:])[0, 1] if n-q3 > 1 else 0.0)
            features.append(0.0 if np.isnan(corr_q4) else corr_q4)
            
            segment_consistency = abs(corr_h1 - corr_h2)
            features.append(segment_consistency)
            
            if len(b) >= 3:
                diff_b = np.diff(b)
                diff2_b = np.diff(diff_b)
                threshold = np.std(diff2_b) * 2
                num_jumps = float(np.sum(np.abs(diff2_b) > threshold)) / len(diff2_b)
                features.append(num_jumps)
            else:
                features.append(0.0)
            
            window = max(3, n // 10)
            local_vars = []
            for i in range(len(b) - window + 1):
                local_vars.append(np.var(b[i:i+window]))
            variance_of_variance = float(np.var(local_vars) if local_vars else 0.0)
            features.append(np.clip(variance_of_variance / (np.var(b) + 1e-12), 0, 10))
            
            if h1 > 1 and n - h1 > 1:
                k1 = np.polyfit(np.arange(h1), b[:h1], 1)[0]
                k2 = np.polyfit(np.arange(n-h1), b[h1:], 1)[0]
                slope_diff = abs(k1 - k2) / (abs(k1) + abs(k2) + 1e-12)
                features.append(float(np.clip(slope_diff, 0, 10)))
            else:
                features.append(0.0)
        except:
            features.extend([0.0] * 8)
    
    # 7. Special pattern detection (8 dims)
    if len(a) < 2 or len(b) < 2:
        features.extend([0.0] * 8)
    else:
        try:
            db = np.zeros_like(b)
            db[0] = b[0]
            db[1:] = np.diff(b)
            is_scan = 1.0 if np.allclose(db, a, rtol=1e-2) else 0.0
            features.append(is_scan)
            
            is_diff = 1.0 if len(b) > 1 and np.allclose(b[1:], a[1:] - a[:-1], rtol=1e-2) and abs(b[0]) < 1e-6 else 0.0
            features.append(is_diff)
            
            is_scale2 = 1.0 if np.allclose(b, 2 * a, rtol=1e-2) else 0.0
            features.append(is_scale2)
            
            unique_vals = len(np.unique(b))
            value_range = np.ptp(b)
            mean_val = np.mean(b)
            is_mod_like = 1.0 if (unique_vals < len(b) * 0.5 and value_range < mean_val * 2) else 0.0
            features.append(is_mod_like)
            
            r2_linear = _r_squared(a, b)
            try:
                A_matrix = np.vstack([np.ones(len(a)), a, a**2]).T
                coeffs = np.linalg.lstsq(A_matrix, b, rcond=None)[0]
                b_pred = A_matrix @ coeffs
                r2_poly = 1 - np.sum((b - b_pred)**2) / (np.var(b) * len(b) + 1e-12)
                is_polynomial = 1.0 if (r2_poly - r2_linear) > 0.1 else 0.0
            except:
                is_polynomial = 0.0
            features.append(is_polynomial)
            
            if np.all(b > 0):
                r2_exp = _r_squared(a, np.log(b + 1e-12))
                is_exponential = 1.0 if r2_exp > 0.9 else 0.0
            else:
                is_exponential = 0.0
            features.append(is_exponential)
            
            if len(b) >= 8:
                acf = _autocorrelation(b - np.mean(b), max_lag=min(len(b)//2, 10))
                is_periodic = 1.0 if len(acf) > 1 and np.max(acf[1:]) > 0.7 else 0.0
            else:
                is_periodic = 0.0
            features.append(is_periodic)
            
            if len(a) >= 4:
                h = len(a) // 2
                corr_h1 = np.corrcoef(a[:h], b[:h])[0, 1] if h > 1 else 0.0
                corr_h2 = np.corrcoef(a[h:], b[h:])[0, 1] if len(a) - h > 1 else 0.0
                if not np.isnan(corr_h1) and not np.isnan(corr_h2):
                    is_piecewise = 1.0 if abs(corr_h1 - corr_h2) > 0.5 else 0.0
                else:
                    is_piecewise = 0.0
            else:
                is_piecewise = 0.0
            features.append(is_piecewise)
        except:
            features.extend([0.0] * 8)
    
    return torch.tensor(features, dtype=torch.float32)


def cheap_features(A: List[int], B: List[int]):
    """
    Legacy 11-dimensional feature extractor (for backward compatibility).
    
    Use enhanced_features() instead for better performance.
    """
    a = np.array(A, dtype=float)
    b = np.array(B, dtype=float)
    
    if len(a) == 0:
        return torch.zeros(11, dtype=torch.float32)
    
    if np.all(a == a[0]):
        k = 0.0
        c = float(np.mean(b)) if len(b) > 0 else 0.0
    else:
        A1 = np.vstack([a, np.ones(len(a))]).T
        try:
            k, c = np.linalg.lstsq(A1, b, rcond=None)[0]
        except (np.linalg.LinAlgError, ValueError, OverflowError):
            k = 0.0
            c = float(np.mean(b)) if len(b) > 0 else 0.0
    
    k = float(np.clip(k, -10, 10))
    c = float(np.clip(c, -100, 100))
    
    db = np.zeros_like(b)
    if len(b) > 0:
        db[0] = b[0]
    if len(b) > 1:
        db[1:] = np.diff(b)
    is_scan = 1.0 if len(b) > 0 and np.allclose(db, a) else 0.0
    
    is_diff1 = 1.0 if len(b) > 1 and np.allclose(b[1:], a[1:]-a[:-1]) and b[0]==0 else 0.0
    
    corr = []
    for d in range(6):
        if d >= len(a) or d >= len(b) or (len(a)-d) < 2:
            corr.append(0.0)
            continue
        va = a[d:]
        vb = b[d:]
        sa = float(np.std(va))
        sb = float(np.std(vb))
        if sa == 0.0 or sb == 0.0:
            corr.append(0.0)
        else:
            with np.errstate(invalid="ignore", divide="ignore"):
                C = np.corrcoef(va, vb)[0,1]
            corr.append(0.0 if (np.isnan(C) or np.isinf(C)) else float(C))
    
    scale2 = 1.0 if np.allclose(b, 2*a) else 0.0
    
    return torch.tensor([k, c, is_scan, is_diff1, scale2] + corr, dtype=torch.float32)


@dataclass
class Cfg:
    d_model:int=256; nhead:int=4; nlayers:int=4; d_ff:int=1024; dropout:float=0.1; ctx_len:int=64; feat_dim:int=54

class TransDecoder(nn.Module):
    def __init__(self, cfg: Cfg, vocab:int=len(TOKENS)):
        super().__init__()
        self.cfg=cfg
        self.tok = nn.Embedding(vocab, cfg.d_model)
        self.pos = nn.Embedding(cfg.ctx_len, cfg.d_model)
        self.fproj = nn.Linear(cfg.feat_dim, cfg.d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model, nhead=cfg.nhead, dim_feedforward=cfg.d_ff,
            dropout=cfg.dropout, batch_first=True, activation="gelu"
        )
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=cfg.nlayers)
        self.lm = nn.Linear(cfg.d_model, vocab)
    
    def forward(self, ctx_ids: torch.LongTensor, feat: torch.FloatTensor):
        B,T = ctx_ids.shape
        T = min(T, self.cfg.ctx_len)
        ctx_ids = ctx_ids[:, -T:]
        pos = torch.arange(T, device=ctx_ids.device).unsqueeze(0).expand(B,T)
        x = self.tok(ctx_ids) + self.pos(pos) + self.fproj(feat).unsqueeze(1)
        mask = torch.triu(torch.ones(T,T,device=x.device, dtype=torch.bool), diagonal=1)
        h = self.enc(x, mask)
        return self.lm(h)
