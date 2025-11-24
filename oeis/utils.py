"""
共享工具函数模块
提供跨模块使用的通用工具函数
"""

import numpy as np
from typing import Optional
from .config import Config


def create_interpreter(strict: bool = True, loose_budget: bool = False):
    """
    创建标准配置的Interpreter实例
    
    Args:
        strict: 是否使用严格模式
        loose_budget: 是否使用宽松预算（用于数据生成）
    
    Returns:
        Interpreter实例
    
    Examples:
        >>> inter = create_interpreter(strict=True)
        >>> inter_loose = create_interpreter(strict=False, loose_budget=True)
    """
    from .interpreter import Interpreter
    return Interpreter(Config.get_interpreter_config(strict=strict, loose_budget=loose_budget))


def safe_corrcoef(a, b, default: float = 0.0) -> float:
    """
    安全计算相关系数，自动处理NaN和异常
    
    Args:
        a: 第一个数组
        b: 第二个数组
        default: 发生错误时返回的默认值
    
    Returns:
        相关系数，如果无法计算则返回default
    
    Examples:
        >>> a = np.array([1, 2, 3])
        >>> b = np.array([2, 4, 6])
        >>> corr = safe_corrcoef(a, b)  # 应该接近1.0
    """
    try:
        if len(a) < 2 or len(b) < 2:
            return default
        corr = np.corrcoef(a, b)[0, 1]
        return default if np.isnan(corr) or np.isinf(corr) else float(corr)
    except Exception:
        return default


def safe_ratio(x: float, y: float, default: float = 0.0, clip_range: tuple = (-100, 100)) -> float:
    """
    安全计算比率，自动处理除零和裁剪
    
    Args:
        x: 分子
        y: 分母
        default: y为0时返回的默认值
        clip_range: 结果裁剪范围 (min, max)
    
    Returns:
        x/y，裁剪到指定范围
    
    Examples:
        >>> safe_ratio(10, 2)  # 5.0
        >>> safe_ratio(10, 0)  # 0.0 (default)
        >>> safe_ratio(1000, 1, clip_range=(-100, 100))  # 100.0 (clipped)
    """
    if abs(y) < 1e-12:
        return default
    return float(np.clip(x / y, clip_range[0], clip_range[1]))


def safe_log(x: float, default: float = 0.0, clip_range: tuple = (-10, 10)) -> float:
    """
    安全计算对数，自动处理非正数和裁剪
    
    Args:
        x: 输入值
        default: x <= 0时返回的默认值
        clip_range: 结果裁剪范围 (min, max)
    
    Returns:
        log10(|x|)，裁剪到指定范围
    
    Examples:
        >>> safe_log(100)  # 2.0
        >>> safe_log(0)    # 0.0 (default)
        >>> safe_log(-5)   # 0.0 (default)
    """
    if x <= 0:
        return default
    return float(np.clip(np.log10(abs(x) + 1e-12), clip_range[0], clip_range[1]))


def ensure_numpy_array(data, dtype=float, default_if_empty=None):
    """
    确保数据是numpy数组，处理空数据和类型转换
    
    Args:
        data: 输入数据（列表、数组等）
        dtype: 目标数据类型
        default_if_empty: 如果数据为空，返回的默认值
    
    Returns:
        numpy数组或default_if_empty
    
    Examples:
        >>> ensure_numpy_array([1, 2, 3])
        array([1., 2., 3.])
        >>> ensure_numpy_array([], default_if_empty=np.array([0]))
        array([0])
    """
    if data is None or len(data) == 0:
        return default_if_empty
    try:
        return np.array(data, dtype=dtype)
    except Exception:
        return default_if_empty


def validate_sequence_pair(A, B, n_in: int, n_chk: int) -> tuple:
    """
    验证序列对的长度是否满足要求
    
    Args:
        A: 输入序列
        B: 输出序列
        n_in: 输入长度
        n_chk: 检查长度
    
    Returns:
        (is_valid: bool, error_message: str)
    
    Examples:
        >>> validate_sequence_pair([1,2,3,4,5], [2,4,6,8,10], 3, 2)
        (True, "")
        >>> validate_sequence_pair([1,2], [2,4], 3, 2)
        (False, "序列太短: 需要5项，A有2项，B有2项")
    """
    min_len = min(len(A), len(B))
    required = n_in + n_chk
    
    if required > min_len:
        return False, f"序列太短: 需要{required}项，A有{len(A)}项，B有{len(B)}项"
    
    return True, ""


def format_tokens(tokens: list) -> str:
    """
    格式化token列表为可读字符串
    
    Args:
        tokens: token列表
    
    Returns:
        格式化的字符串
    
    Examples:
        >>> format_tokens(["SCALE", "2", "A"])
        "SCALE 2 A"
        >>> format_tokens([])
        "(empty)"
    """
    if not tokens:
        return "(empty)"
    return " ".join(str(t) for t in tokens)


class ExceptionSuppressor:
    """
    上下文管理器：静默捕获异常，用于非关键操作
    
    Examples:
        >>> with ExceptionSuppressor():
        ...     risky_operation()
        
        >>> with ExceptionSuppressor(default_return=0) as result:
        ...     result.value = compute_something()
        >>> print(result.value)  # 如果异常则为0
    """
    
    def __init__(self, default_return=None, log_errors: bool = False):
        self.default_return = default_return
        self.log_errors = log_errors
        self.value = default_return
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None and self.log_errors:
            import logging
            logging.warning(f"Suppressed exception: {exc_type.__name__}: {exc_val}")
        return True  # Suppress exception


def chunk_list(lst: list, chunk_size: int):
    """
    将列表分割为指定大小的块
    
    Args:
        lst: 输入列表
        chunk_size: 每块的大小
    
    Yields:
        列表块
    
    Examples:
        >>> list(chunk_list([1,2,3,4,5], 2))
        [[1, 2], [3, 4], [5]]
    """
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size]


def safe_int_conversion(value, default: int = 0) -> int:
    """
    安全将值转换为整数
    
    Args:
        value: 输入值
        default: 转换失败时的默认值
    
    Returns:
        整数值
    
    Examples:
        >>> safe_int_conversion("123")
        123
        >>> safe_int_conversion("abc")
        0
        >>> safe_int_conversion(3.7)
        3
    """
    try:
        return int(value)
    except (ValueError, TypeError):
        return default


def get_git_revision() -> Optional[str]:
    """
    获取当前git版本号（如果在git仓库中）
    
    Returns:
        git commit hash或None
    """
    try:
        import subprocess
        result = subprocess.run(
            ['git', 'rev-parse', '--short', 'HEAD'],
            capture_output=True,
            text=True,
            timeout=1
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


def detect_element_repetition(seq: list, max_repeat: int = 10) -> Optional[int]:
    """
    检测序列中每个元素是否重复了k次（REPEAT操作的逆）
    
    例如: [1,1,2,2,3,3] 每个元素重复2次
    
    Args:
        seq: 输入序列
        max_repeat: 最大重复次数
    
    Returns:
        重复次数k（如果存在），否则返回None
    
    Examples:
        >>> detect_element_repetition([1,1,2,2,3,3])
        2
        >>> detect_element_repetition([1,1,1,2,2,2,3,3,3])
        3
        >>> detect_element_repetition([1,2,3,4,5])
        None
    """
    if len(seq) < 2:
        return None
    
    # 尝试不同的重复次数
    for k in range(2, min(max_repeat + 1, len(seq) + 1)):
        if len(seq) % k != 0:
            continue  # 长度必须是k的倍数
        
        # 检查每k个元素是否相同
        is_element_repeated = True
        expected_len = len(seq) // k
        
        for i in range(expected_len):
            # 检查第i个"组"的k个元素是否都相同
            base_val = seq[i * k]
            for j in range(1, k):
                if i * k + j >= len(seq) or seq[i * k + j] != base_val:
                    is_element_repeated = False
                    break
            if not is_element_repeated:
                break
        
        if is_element_repeated:
            return k
    
    return None


def subsample_repeated_sequence(seq: list, repeat_factor: int) -> list:
    """
    从元素重复序列中提取原始序列（SUBSAMPLE操作）
    
    Args:
        seq: 输入序列（每个元素重复了repeat_factor次）
        repeat_factor: 重复次数
    
    Returns:
        简化后的序列（每repeat_factor个取一个）
    
    Examples:
        >>> subsample_repeated_sequence([1,1,2,2,3,3], 2)
        [1, 2, 3]
        >>> subsample_repeated_sequence([1,1,1,2,2,2], 3)
        [1, 2]
    """
    return [seq[i * repeat_factor] for i in range(len(seq) // repeat_factor)]


def preprocess_sequences(A: list, B: list, max_repeat: int = 10) -> tuple:
    """
    预处理序列：检测并简化元素重复模式
    
    如果 A 或 B 的每个元素都重复了k次（REPEAT操作的结果），则使用SUBSAMPLE简化
    
    Args:
        A: 输入序列
        B: 输出序列
        max_repeat: 最大重复次数
    
    Returns:
        (processed_A, processed_B, was_simplified: bool)
    
    Examples:
        >>> preprocess_sequences([1,1,2,2,3,3], [2,2,4,4,6,6])
        ([1, 2, 3], [2, 4, 6], True)
        >>> preprocess_sequences([1,2,3,4], [2,4,6,8])
        ([1, 2, 3, 4], [2, 4, 6, 8], False)
    """
    # 检测 A 的元素重复次数
    repeat_A = detect_element_repetition(A, max_repeat)
    
    # 检测 B 的元素重复次数
    repeat_B = detect_element_repetition(B, max_repeat)
    
    # 如果两者都有元素重复且重复次数相同，则简化
    if repeat_A is not None and repeat_B is not None and repeat_A == repeat_B:
        A_simplified = subsample_repeated_sequence(A, repeat_A)
        B_simplified = subsample_repeated_sequence(B, repeat_B)
        return A_simplified, B_simplified, True
    
    # 如果只有 A 有元素重复，检查 B 是否也符合该重复次数
    if repeat_A is not None:
        # 验证 B 是否也按相同次数重复
        if len(B) % repeat_A == 0:
            expected_len = len(B) // repeat_A
            is_B_repeated = True
            
            for i in range(expected_len):
                base_val = B[i * repeat_A]
                for j in range(1, repeat_A):
                    if i * repeat_A + j >= len(B) or B[i * repeat_A + j] != base_val:
                        is_B_repeated = False
                        break
                if not is_B_repeated:
                    break
            
            if is_B_repeated:
                A_simplified = subsample_repeated_sequence(A, repeat_A)
                B_simplified = subsample_repeated_sequence(B, repeat_A)
                return A_simplified, B_simplified, True
    
    # 如果只有 B 有元素重复，检查 A 是否也符合该重复次数
    if repeat_B is not None:
        # 验证 A 是否也按相同次数重复
        if len(A) % repeat_B == 0:
            expected_len = len(A) // repeat_B
            is_A_repeated = True
            
            for i in range(expected_len):
                base_val = A[i * repeat_B]
                for j in range(1, repeat_B):
                    if i * repeat_B + j >= len(A) or A[i * repeat_B + j] != base_val:
                        is_A_repeated = False
                        break
                if not is_A_repeated:
                    break
            
            if is_A_repeated:
                A_simplified = subsample_repeated_sequence(A, repeat_B)
                B_simplified = subsample_repeated_sequence(B, repeat_B)
                return A_simplified, B_simplified, True
    
    # 没有元素重复或无法简化
    return A, B, False