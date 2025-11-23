"""
Feature-driven template matching system.

Uses 54-dimensional features to intelligently generate template candidates,
dramatically reducing the need for expensive beam search.
"""

from typing import List, Tuple, Optional
import numpy as np
from oeis.smart_polynomial import (
    fit_quadratic_integer, 
    fit_linear_integer,
    fit_scale_integer,
    fit_offset_integer
)


def generate_templates_from_features(A: List[int], B: List[int], feat) -> List[Tuple[int, List[str], str]]:
    """
    Generate template candidates based on 54-dim enhanced features.
    
    Args:
        A, B: Input/output sequences
        feat: 54-dim feature vector (torch.Tensor or numpy array)
        
    Returns:
        List of (priority, template_tokens, description) tuples,
        sorted by priority (descending)
    """
    templates = []
    
    # Convert feat to numpy if needed
    if hasattr(feat, 'numpy'):
        feat = feat.detach().cpu().numpy()
    elif hasattr(feat, 'tolist'):
        feat = np.array(feat.tolist())
    else:
        feat = np.array(feat)
    
    # Extract key features (indices match enhanced_features in torch_model.py)
    len_ratio = float(feat[0])
    mean_ratio = float(feat[1])
    log_k = float(feat[8])
    k_raw = float(feat[9])
    log_c = float(feat[10])
    c_raw = float(feat[11])
    r2_square = float(feat[12])
    r2_cube = float(feat[13])
    power_exp = float(feat[16])
    periodicity_score = float(feat[23])
    period_length = float(feat[22])
    is_scan = float(feat[46])
    is_diff = float(feat[47])
    is_scale2 = float(feat[48])
    is_mod_like = float(feat[49])
    is_polynomial = float(feat[50])
    is_exponential = float(feat[51])
    is_periodic = float(feat[52])
    is_piecewise = float(feat[53])
    
    # ============================================================
    # 0. ULTRA HIGH PRIORITY: Smart polynomial fitting
    #    直接计算最佳系数，不受±16限制
    # ============================================================
    
    # 0a. 二次关系: B = a*A² + b*A + c
    quad_result = fit_quadratic_integer(A, B, r2_threshold=0.90, max_coeff=10000)
    if quad_result is not None:
        a, b, c = quad_result
        templates.append((100, 
                        ["POLY", str(a), str(b), str(c), "A"], 
                        f"Smart quad: {a}*x²{b:+d}*x{c:+d}"))
    
    # 0b. 线性关系: B = k*A + c  
    linear_result = fit_linear_integer(A, B, r2_threshold=0.95, max_coeff=10000)
    if linear_result is not None:
        k, c = linear_result
        if c == 0:
            # 纯缩放: B = k*A
            templates.append((99, 
                            ["SCALE", str(k), "A"], 
                            f"Smart scale: {k}*A"))
        elif k == 1:
            # 纯偏移: B = A + c
            templates.append((99, 
                            ["OFFSET", str(c), "A"], 
                            f"Smart offset: A{c:+d}"))
        else:
            # 线性组合: B = k*A + c
            templates.append((98, 
                            ["OFFSET", str(c), "SCALE", str(k), "A"], 
                            f"Smart linear: {k}*A{c:+d}"))
    
    # 0c. 纯缩放: B = k*A
    scale_result = fit_scale_integer(A, B, r2_threshold=0.98, max_coeff=10000)
    if scale_result is not None:
        k = scale_result
        templates.append((97, 
                        ["SCALE", str(k), "A"], 
                        f"Smart scale: {k}*A"))
    
    # 0d. 纯偏移: B = A + c
    offset_result = fit_offset_integer(A, B, r2_threshold=0.98, max_coeff=10000)
    if offset_result is not None:
        c = offset_result
        templates.append((96, 
                        ["OFFSET", str(c), "A"], 
                        f"Smart offset: A{c:+d}"))
    
    # ============================================================
    # 1. 常见模式的备用模板（如果智能拟合失败）
    # ============================================================
    # Direct check: does B look like (A-k)²?
    if len(A) >= 5 and len(B) >= 5:
        # Test if B ≈ (A-c)² for c in [-2, -1, 0, 1, 2]
        for c in [1, 2, 0, -1, -2]:
            # (x-c)² = x² - 2cx + c²
            a_coeff = 1
            b_coeff = -2 * c
            c_coeff = c * c
            templates.append((90 - abs(c), 
                            ["POLY", str(a_coeff), str(b_coeff), str(c_coeff), "A"], 
                            f"Common: (x{c:+d})²"))
    
    # Other common polynomials (backup)
    templates.append((85, ["POLY", "1", "1", "0", "A"], "Common: x²+x"))
    templates.append((84, ["POLY", "1", "-1", "0", "A"], "Common: x²-x"))
    templates.append((83, ["POLY", "1", "0", "0", "A"], "Common: x²"))
    templates.append((82, ["POLY", "1", "0", "1", "A"], "Common: x²+1"))
    templates.append((81, ["POLY", "1", "0", "-1", "A"], "Common: x²-1"))
    
    # 2. Direct boolean matches (priority 70-79)
    if is_scan > 0.5:
        templates.append((79, ["SCAN_ADD", "A"], "Cumulative sum (is_scan)"))
    
    if is_diff > 0.5:
        templates.append((79, ["DIFF_FWD", "1", "A"], "Forward difference (is_diff)"))
    
    if is_scale2 > 0.5:
        templates.append((79, ["SCALE", "2", "A"], "Scale by 2 (is_scale2)"))
    
    # 2. Power law relationships (80-90)
    if 1.8 < power_exp < 2.2 and r2_square > 0.90:
        # Square: B ≈ A²
        # Try POLY 1 0 0 (a*x² + b*x + c where a=1, b=0, c=0)
        templates.append((90, ["POLY", "1", "0", "0", "A"], f"Square (power_exp={power_exp:.2f})"))
    
    if 2.8 < power_exp < 3.2 and r2_cube > 0.90:
        # Cubic: B ≈ A³ (no direct primitive, but flag it)
        # Could try nested SCALE operations or leave for beam search
        templates.append((85, ["POLY", "1", "0", "0", "A"], f"Cubic hint (power_exp={power_exp:.2f})"))
    
    # 3. Simple linear scaling (70-80)
    if abs(log_k) > 0.1 and -16 <= k_raw <= 16:
        k_int = int(round(k_raw))
        if abs(k_int - k_raw) < 0.3 and k_int != 0:  # Allow some tolerance
            templates.append((80, ["SCALE", str(k_int), "A"], f"Scale by {k_int} (k_raw={k_raw:.2f})"))
    
    # 4. Modulo-like pattern (70-75)
    if is_mod_like > 0.5 and is_periodic > 0.5:
        # Infer modulo from period length
        if 2 <= period_length <= 10:
            mod_val = int(round(period_length))
            if 2 <= mod_val <= 10:
                templates.append((75, ["MAP_MOD", str(mod_val), "A"], 
                                f"Modulo {mod_val} (period={period_length:.1f})"))
    
    # 5. Offset patterns (65-70)
    if abs(c_raw) > 1 and -16 <= c_raw <= 16:
        c_int = int(round(c_raw))
        if abs(c_int - c_raw) < 0.3 and c_int != 0:
            templates.append((70, ["OFFSET", str(c_int), "A"], f"Offset by {c_int} (c_raw={c_raw:.2f})"))
    
    # 6. Linear combination: k*A + c (60-65)
    if (abs(log_k) > 0.1 and abs(c_raw) > 1 and 
        -16 <= k_raw <= 16 and -16 <= c_raw <= 16):
        k_int = int(round(k_raw))
        c_int = int(round(c_raw))
        if (abs(k_int - k_raw) < 0.3 and abs(c_int - c_raw) < 0.3 and 
            k_int != 0):
            # OFFSET c (SCALE k A)
            templates.append((65, 
                ["OFFSET", str(c_int), "SCALE", str(k_int), "A"], 
                f"Linear: {k_int}*A + {c_int}"))
    
    # 7. Polynomial (general case, lower priority 55-60)
    if is_polynomial > 0.5:
        # Try common polynomial patterns
        # POLY a b c: a*x² + b*x + c
        templates.append((60, ["POLY", "1", "2", "1", "A"], "Polynomial: x²+2x+1"))
        templates.append((58, ["POLY", "1", "1", "0", "A"], "Polynomial: x²+x"))
        templates.append((56, ["POLY", "1", "0", "1", "A"], "Polynomial: x²+1"))
    
    # 7b. Common polynomial shifts (HIGH PRIORITY!)
    # Removed duplicate - moved to section 0 above
    
    # 8. Moonshine-specific templates (50-55)
    # These use actual B values for INSERT constants
    if len(B) >= 2:
        templates.append((52, ["INSERT1", "SCAN_ADD", "A"], 
                        f"Moonshine INSERT1 (B[1])"))
    
    if len(B) >= 3:
        templates.append((51, ["INSERT2", "SCAN_ADD", "A"], 
                        f"Moonshine INSERT2 (B[2])"))
    
    # Always include basic moonshine template as last resort
    templates.append((50, ["SCAN_ADD", "A"], "Moonshine baseline"))
    
    # 9. Difference-based templates (45-50)
    if abs(mean_ratio - 1.0) < 0.1:  # Mean is similar
        # Try difference operations with different orders
        templates.append((48, ["DIFF_FWD", "1", "A"], "Difference order 1"))
        templates.append((46, ["DIFF_FWD", "2", "A"], "Difference order 2"))
    
    # 10. Scan multiply (40-45)
    if abs(power_exp - 1.0) < 0.2 and len(A) >= 3:
        # Check if B might be cumulative product
        templates.append((45, ["SCAN_MUL", "A"], "Cumulative product"))
    
    # Sort by priority (descending)
    templates.sort(key=lambda x: x[0], reverse=True)
    
    return templates


def try_feature_templates(A: List[int], B: List[int], feat, 
                         n_in: int, n_chk: int, 
                         checker_mode: str = "exact",
                         k_strict: int = 3,
                         tau0: float = 2e-3,
                         tau1: float = 1e-3,
                         max_templates: int = None) -> Tuple[Optional[List[str]], Optional[object]]:
    """
    Try feature-driven templates and return the first one that passes.
    
    Args:
        A, B: Sequences
        feat: 54-dim feature vector
        n_in, n_chk: Train/validation split
        checker_mode: "exact" or "moonshine"
        k_strict, tau0, tau1: Moonshine parameters
        max_templates: Maximum number of templates to try (None = try all)
        
    Returns:
        (best_tokens, best_report) if found, else (None, None)
    """
    from .checker import check_program_on_pair, check_program_moonshine
    
    templates = generate_templates_from_features(A, B, feat)
    
    # Limit the number of templates to try (performance)
    # If max_templates is None, try all templates
    if max_templates is not None:
        templates = templates[:max_templates]
    
    for priority, toks, desc in templates:
        try:
            if checker_mode == "moonshine":
                rep = check_program_moonshine(
                    toks, A_full=A, B_full=B,
                    n_in=n_in, n_chk=n_chk,
                    k_strict=k_strict, tau0=tau0, tau1=tau1
                )
            else:
                rep = check_program_on_pair(
                    toks, A_full=A, B_full=B,
                    n_in=n_in, n_chk=n_chk
                )
            
            if rep and rep.ok:
                # Success!
                return toks, rep
        except Exception as e:
            # Template failed, continue to next
            continue
    
    # No template worked
    return None, None


def get_template_statistics(A: List[int], B: List[int], feat) -> dict:
    """
    Get statistics about templates that would be generated.
    Useful for debugging and analysis.
    
    Returns:
        Dictionary with template statistics
    """
    templates = generate_templates_from_features(A, B, feat)
    
    stats = {
        "total_templates": len(templates),
        "templates_by_priority": {},
        "top_5_templates": [],
        "feature_flags": {}
    }
    
    # Convert feat to numpy if needed
    if hasattr(feat, 'numpy'):
        feat = feat.detach().cpu().numpy()
    elif hasattr(feat, 'tolist'):
        feat = np.array(feat.tolist())
    else:
        feat = np.array(feat)
    
    # Group by priority ranges
    for priority, toks, desc in templates:
        if priority >= 90:
            key = "90-100 (highest)"
        elif priority >= 70:
            key = "70-89 (high)"
        elif priority >= 50:
            key = "50-69 (medium)"
        else:
            key = "0-49 (low)"
        
        stats["templates_by_priority"][key] = stats["templates_by_priority"].get(key, 0) + 1
    
    # Top 5
    for priority, toks, desc in templates[:5]:
        stats["top_5_templates"].append({
            "priority": priority,
            "tokens": " ".join(toks),
            "description": desc
        })
    
    # Feature flags
    stats["feature_flags"] = {
        "is_scan": float(feat[46]) > 0.5,
        "is_diff": float(feat[47]) > 0.5,
        "is_scale2": float(feat[48]) > 0.5,
        "is_mod_like": float(feat[49]) > 0.5,
        "is_polynomial": float(feat[50]) > 0.5,
        "is_exponential": float(feat[51]) > 0.5,
        "is_periodic": float(feat[52]) > 0.5,
        "is_piecewise": float(feat[53]) > 0.5,
        "power_exp": float(feat[16]),
        "k_raw": float(feat[9]),
        "c_raw": float(feat[11]),
    }
    
    return stats


