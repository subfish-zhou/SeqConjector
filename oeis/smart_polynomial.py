"""
智能计算多项式关系的系数

对于二次关系 B[i] = a*A[i]² + b*A[i] + c，使用最小二乘法直接求解
"""

import numpy as np
from typing import List, Tuple, Optional


def fit_quadratic(A: List[int], B: List[int]) -> Optional[Tuple[float, float, float, float]]:
    """
    拟合二次关系: B = a*A² + b*A + c
    
    返回: (a, b, c, r_squared) 如果拟合良好，否则 None
    """
    if len(A) < 3 or len(B) < 3:
        return None
    
    min_len = min(len(A), len(B))
    A_arr = np.array(A[:min_len], dtype=float)
    B_arr = np.array(B[:min_len], dtype=float)
    
    try:
        # 构建设计矩阵: [A², A, 1]
        X = np.column_stack([A_arr**2, A_arr, np.ones_like(A_arr)])
        
        # 最小二乘求解
        coeffs, residuals, rank, s = np.linalg.lstsq(X, B_arr, rcond=None)
        a, b, c = coeffs
        
        # 计算 R²
        B_pred = a * A_arr**2 + b * A_arr + c
        ss_res = np.sum((B_arr - B_pred)**2)
        ss_tot = np.sum((B_arr - np.mean(B_arr))**2)
        
        if ss_tot < 1e-10:
            r_squared = 1.0 if ss_res < 1e-10 else 0.0
        else:
            r_squared = 1.0 - (ss_res / ss_tot)
        
        return (a, b, c, r_squared)
    
    except:
        return None


def fit_quadratic_integer(A: List[int], B: List[int], 
                          r2_threshold: float = 0.95,
                          max_coeff: int = 1000,
                          integer_tolerance: float = 0.05) -> Optional[Tuple[int, int, int]]:
    """
    拟合二次关系并返回整数系数
    
    策略：
    1. 先用浮点数拟合得到候选系数
    2. 四舍五入到整数
    3. 用整数系数重新计算，验证是否真的匹配
    
    Args:
        A, B: 输入输出序列
        r2_threshold: R²阈值，低于此值认为不是二次关系
        max_coeff: 系数的最大绝对值（防止过大）
        integer_tolerance: 浮点系数与整数的最大偏差
    
    Returns:
        (a, b, c) 整数系数，如果拟合不好则返回 None
    """
    result = fit_quadratic(A, B)
    
    if result is None:
        return None
    
    a_float, b_float, c_float, r_squared = result
    
    # R²太低，不是二次关系
    if r_squared < r2_threshold:
        return None
    
    # 四舍五入到整数
    a_int = int(round(a_float))
    b_int = int(round(b_float))
    c_int = int(round(c_float))
    
    # 检查系数是否合理
    if (abs(a_int) > max_coeff or 
        abs(b_int) > max_coeff or 
        abs(c_int) > max_coeff):
        return None
    
    # 检查浮点系数是否接近整数（关键！）
    # 如果偏差太大，说明真实关系不是整数系数
    if abs(a_float - a_int) > integer_tolerance and abs(a_float) > 0.01:
        return None
    if abs(b_float - b_int) > integer_tolerance and abs(b_float) > 0.01:
        return None
    if abs(c_float - c_int) > integer_tolerance and abs(c_float) > 0.01:
        return None
    
    # 关键验证：用整数系数重新计算，看是否真的匹配
    min_len = min(len(A), len(B))
    A_arr = np.array(A[:min_len], dtype=int)
    B_arr = np.array(B[:min_len], dtype=int)
    
    # 用整数系数计算预测值
    B_pred_int = a_int * A_arr * A_arr + b_int * A_arr + c_int
    
    # 检查整数预测是否精确匹配
    # 对于整数关系，应该完全相等或误差极小
    max_error = np.max(np.abs(B_arr - B_pred_int))
    
    # 如果最大误差>0，说明不是精确的整数关系
    if max_error > 0:
        return None
    
    return (a_int, b_int, c_int)


def fit_linear_integer(A: List[int], B: List[int],
                       r2_threshold: float = 0.95,
                       max_coeff: int = 1000,
                       integer_tolerance: float = 0.05) -> Optional[Tuple[int, int]]:
    """
    拟合线性关系: B = k*A + c
    
    策略：
    1. 先用浮点数拟合
    2. 四舍五入到整数
    3. 用整数系数验证是否真的精确匹配
    
    Returns:
        (k, c) 整数系数，如果拟合不好则返回 None
    """
    if len(A) < 2 or len(B) < 2:
        return None
    
    min_len = min(len(A), len(B))
    A_arr = np.array(A[:min_len], dtype=float)
    B_arr = np.array(B[:min_len], dtype=float)
    
    try:
        # 构建设计矩阵: [A, 1]
        X = np.column_stack([A_arr, np.ones_like(A_arr)])
        
        # 最小二乘求解
        coeffs, residuals, rank, s = np.linalg.lstsq(X, B_arr, rcond=None)
        k_float, c_float = coeffs
        
        # 计算 R²
        B_pred = k_float * A_arr + c_float
        ss_res = np.sum((B_arr - B_pred)**2)
        ss_tot = np.sum((B_arr - np.mean(B_arr))**2)
        
        if ss_tot < 1e-10:
            r_squared = 1.0 if ss_res < 1e-10 else 0.0
        else:
            r_squared = 1.0 - (ss_res / ss_tot)
        
        # R²太低
        if r_squared < r2_threshold:
            return None
        
        # 四舍五入到整数
        k_int = int(round(k_float))
        c_int = int(round(c_float))
        
        # 检查系数是否合理
        if abs(k_int) > max_coeff or abs(c_int) > max_coeff:
            return None
        
        # 检查浮点系数是否接近整数
        if abs(k_float - k_int) > integer_tolerance and abs(k_float) > 0.01:
            return None
        if abs(c_float - c_int) > integer_tolerance and abs(c_float) > 0.01:
            return None
        
        # 关键验证：用整数系数重新计算，看是否精确匹配
        A_arr_int = np.array(A[:min_len], dtype=int)
        B_arr_int = np.array(B[:min_len], dtype=int)
        B_pred_int = k_int * A_arr_int + c_int
        
        # 检查整数预测是否精确匹配
        max_error = np.max(np.abs(B_arr_int - B_pred_int))
        
        if max_error > 0:
            return None
        
        return (k_int, c_int)
    
    except:
        return None


def fit_scale_integer(A: List[int], B: List[int],
                     r2_threshold: float = 0.98,
                     max_coeff: int = 1000,
                     integer_tolerance: float = 0.05) -> Optional[int]:
    """
    拟合纯缩放关系: B = k*A
    
    Returns:
        k (整数)，如果拟合不好则返回 None
    """
    if len(A) < 2 or len(B) < 2:
        return None
    
    min_len = min(len(A), len(B))
    A_arr = np.array(A[:min_len], dtype=float)
    B_arr = np.array(B[:min_len], dtype=float)
    
    # 避免除零
    if np.any(np.abs(A_arr) < 1e-10):
        return None
    
    try:
        # 最小二乘: k = (A·B) / (A·A)
        k_float = np.sum(A_arr * B_arr) / np.sum(A_arr * A_arr)
        
        # 计算 R²
        B_pred = k_float * A_arr
        ss_res = np.sum((B_arr - B_pred)**2)
        ss_tot = np.sum((B_arr - np.mean(B_arr))**2)
        
        if ss_tot < 1e-10:
            r_squared = 1.0 if ss_res < 1e-10 else 0.0
        else:
            r_squared = 1.0 - (ss_res / ss_tot)
        
        # R²太低
        if r_squared < r2_threshold:
            return None
        
        # 四舍五入到整数
        k_int = int(round(k_float))
        
        # 检查系数是否合理
        if abs(k_int) > max_coeff or k_int == 0:
            return None
        
        # 检查浮点系数是否接近整数
        if abs(k_float - k_int) > integer_tolerance:
            return None
        
        # 关键验证：用整数系数重新计算
        A_arr_int = np.array(A[:min_len], dtype=int)
        B_arr_int = np.array(B[:min_len], dtype=int)
        B_pred_int = k_int * A_arr_int
        
        max_error = np.max(np.abs(B_arr_int - B_pred_int))
        
        if max_error > 0:
            return None
        
        return k_int
    
    except:
        return None


def fit_offset_integer(A: List[int], B: List[int],
                      r2_threshold: float = 0.98,
                      max_coeff: int = 1000,
                      integer_tolerance: float = 0.05) -> Optional[int]:
    """
    拟合纯偏移关系: B = A + c
    
    Returns:
        c (整数)，如果拟合不好则返回 None
    """
    if len(A) < 2 or len(B) < 2:
        return None
    
    min_len = min(len(A), len(B))
    A_arr = np.array(A[:min_len], dtype=float)
    B_arr = np.array(B[:min_len], dtype=float)
    
    try:
        # 最优偏移量: c = mean(B - A)
        c_float = np.mean(B_arr - A_arr)
        
        # 计算 R²
        B_pred = A_arr + c_float
        ss_res = np.sum((B_arr - B_pred)**2)
        ss_tot = np.sum((B_arr - np.mean(B_arr))**2)
        
        if ss_tot < 1e-10:
            r_squared = 1.0 if ss_res < 1e-10 else 0.0
        else:
            r_squared = 1.0 - (ss_res / ss_tot)
        
        # R²太低
        if r_squared < r2_threshold:
            return None
        
        # 四舍五入到整数
        c_int = int(round(c_float))
        
        # 检查系数是否合理
        if abs(c_int) > max_coeff:
            return None
        
        # 检查浮点系数是否接近整数
        if abs(c_float - c_int) > integer_tolerance:
            return None
        
        # 关键验证：用整数系数重新计算
        A_arr_int = np.array(A[:min_len], dtype=int)
        B_arr_int = np.array(B[:min_len], dtype=int)
        B_pred_int = A_arr_int + c_int
        
        max_error = np.max(np.abs(B_arr_int - B_pred_int))
        
        if max_error > 0:
            return None
        
        return c_int
    
    except:
        return None

