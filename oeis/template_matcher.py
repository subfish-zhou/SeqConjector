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
    fit_offset_integer,
    detect_growth_pattern,
    fit_exponential_integer
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
    # 0. GROWTH PATTERN DETECTION & ADVANCED TEMPLATES
    #    检测B序列的增长模式，尝试预处理+POLY或指数拟合
    # ============================================================
    
    growth_pattern = detect_growth_pattern(B)
    
    # 尝试对B序列应用不同的预处理，看能否变成多项式或指数关系
    preprocessing_ops = [
        ("", []),  # 无预处理
        ("INSERT1", ["INSERT1"]),
        ("INSERT2", ["INSERT2"]),
        ("DROP1", ["DROP1"]),
        ("DROP2", ["DROP2"]),
        ("REIDX_EVEN", ["REIDX_EVEN"]),
        ("REIDX_ODD", ["REIDX_ODD"]),
    ]
    
    # 如果增长模式是多项式或指数，尝试预处理+拟合
    if growth_pattern in ["quadratic", "polynomial", "exponential", "super_exponential"]:
        for preproc_name, preproc_ops in preprocessing_ops:
            # 应用预处理到A序列
            try:
                A_processed = A[:]
                
                # 模拟预处理操作
                from oeis.interpreter import Interpreter, ExecConfig
                from oeis.program import Node, Program
                
                if len(preproc_ops) > 0:
                    op_name = preproc_ops[0]
                    node = Node(op_name, args=[], kids=[Node("A", args=[], kids=[])])
                    prog = Program(node)
                    
                    interp = Interpreter(ExecConfig(strict=False))
                    result = interp.execute(prog, A, B)
                    
                    if result.ok and len(result.seq) >= 3:
                        A_processed = result.seq
                    else:
                        continue
                
                # 尝试多项式拟合 -> 但不生成POLY模板（已从interpreter删除）
                if growth_pattern in ["quadratic", "polynomial"]:
                    # 线性拟合
                    linear_result = fit_linear_integer(A_processed, B, r2_threshold=0.95, max_coeff=10000)
                    if linear_result is not None and preproc_ops:
                        k, c = linear_result
                        if c == 0:
                            templates.append((104, 
                                            ["SCALE", str(k)] + preproc_ops + ["A"], 
                                            f"Advanced: {preproc_name} then scale {k}"))
                        elif k == 1:
                            templates.append((104, 
                                            ["OFFSET", str(c)] + preproc_ops + ["A"], 
                                            f"Advanced: {preproc_name} then offset {c:+d}"))
                        else:
                            templates.append((103, 
                                            ["OFFSET", str(c), "SCALE", str(k)] + preproc_ops + ["A"], 
                                            f"Advanced: {preproc_name} then {k}*x{c:+d}"))
                
                # 尝试指数拟合
                if growth_pattern in ["exponential", "super_exponential"]:
                    exp_result = fit_exponential_integer(B, r2_threshold=0.90, max_base=100)
                    if exp_result is not None:
                        a_exp, base = exp_result
                        # 指数序列: B[i] = a * base^i
                        # 这需要特殊处理，因为我们没有直接的EXP原语
                        # 但我们可以用SCAN_MUL和常数序列来模拟
                        if preproc_ops and a_exp == 1:
                            templates.append((106, 
                                            ["SCALE", str(base)] + preproc_ops + ["A"], 
                                            f"Exponential: {preproc_name} then base={base}"))
            
            except:
                continue
    
    # ============================================================
    # 1. ULTRA HIGH PRIORITY: Smart polynomial fitting (direct)
    #    直接计算最佳系数，不受±16限制
    # ============================================================
    
    # 1a. 二次关系: B = a*A² + b*A + c (POLY已从interpreter删除，不生成此类模板)
    
    # 1b. 线性关系: B = k*A + c (direct, no preprocessing)
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
    
    # 1c. 纯缩放: B = k*A
    scale_result = fit_scale_integer(A, B, r2_threshold=0.98, max_coeff=10000)
    if scale_result is not None:
        k = scale_result
        templates.append((97, 
                        ["SCALE", str(k), "A"], 
                        f"Smart scale: {k}*A"))
    
    # 1d. 纯偏移: B = A + c
    offset_result = fit_offset_integer(A, B, r2_threshold=0.98, max_coeff=10000)
    if offset_result is not None:
        c = offset_result
        templates.append((96, 
                        ["OFFSET", str(c), "A"], 
                        f"Smart offset: A{c:+d}"))
    
    # ============================================================
    # 2. 常见模式的备用模板（POLY相关已删除）
    # ============================================================
    
    # 3. Direct boolean matches (priority 70-79)
    if is_scan > 0.5:
        templates.append((79, ["SCAN_ADD", "A"], "Cumulative sum (is_scan)"))
    
    if is_diff > 0.5:
        templates.append((79, ["DIFF_FWD", "1", "A"], "Forward difference (is_diff)"))
    
    if is_scale2 > 0.5:
        templates.append((79, ["SCALE", "2", "A"], "Scale by 2 (is_scale2)"))
    
    # 4. Power law relationships (80-90) - POLY已删除，跳过多项式模式
    
    # 5. Simple linear scaling (70-80)
    if abs(log_k) > 0.1 and -16 <= k_raw <= 16:
        k_int = int(round(k_raw))
        if abs(k_int - k_raw) < 0.3 and k_int != 0:  # Allow some tolerance
            templates.append((80, ["SCALE", str(k_int), "A"], f"Scale by {k_int} (k_raw={k_raw:.2f})"))
    
    # 6. Modulo-like pattern (70-75)
    if is_mod_like > 0.5 and is_periodic > 0.5:
        # Infer modulo from period length
        if 2 <= period_length <= 10:
            mod_val = int(round(period_length))
            if 2 <= mod_val <= 10:
                templates.append((75, ["MAP_MOD", str(mod_val), "A"], 
                                f"Modulo {mod_val} (period={period_length:.1f})"))
    
    # 7. Offset patterns (65-70)
    if abs(c_raw) > 1 and -16 <= c_raw <= 16:
        c_int = int(round(c_raw))
        if abs(c_int - c_raw) < 0.3 and c_int != 0:
            templates.append((70, ["OFFSET", str(c_int), "A"], f"Offset by {c_int} (c_raw={c_raw:.2f})"))
    
    # 8. Linear combination: k*A + c (60-65)
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
    
    # 9. Polynomial (general case) - POLY已从interpreter删除，不生成模板
    
    # 10. Moonshine-specific templates (50-55)
    # These use actual B values for INSERT constants
    if len(B) >= 2:
        templates.append((52, ["INSERT1", "SCAN_ADD", "A"], 
                        f"Moonshine INSERT1 (B[1])"))
    
    if len(B) >= 3:
        templates.append((51, ["INSERT2", "SCAN_ADD", "A"], 
                        f"Moonshine INSERT2 (B[2])"))
    
    # Always include basic moonshine template as last resort
    templates.append((50, ["SCAN_ADD", "A"], "Moonshine baseline"))
    
    # 11. Difference-based templates (45-50)
    if abs(mean_ratio - 1.0) < 0.1:  # Mean is similar
        # Try difference operations with different orders
        templates.append((48, ["DIFF_FWD", "1", "A"], "Difference order 1"))
        templates.append((46, ["DIFF_FWD", "2", "A"], "Difference order 2"))
    
    # 12. Scan multiply (40-45)
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


