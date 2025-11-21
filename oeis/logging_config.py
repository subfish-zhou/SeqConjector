"""
统一的日志配置模块
"""

import logging
import sys
from typing import Optional


def setup_logger(
    name: str = "seqconjector",
    level: str = "INFO",
    log_file: Optional[str] = None,
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    设置并返回配置好的logger
    
    Args:
        name: logger名称
        level: 日志级别 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: 可选的日志文件路径
        format_string: 自定义格式字符串
    
    Returns:
        配置好的Logger实例
    
    Examples:
        >>> logger = setup_logger("myapp", "DEBUG")
        >>> logger.info("Application started")
        >>> logger.debug("Debug message")
    """
    logger = logging.getLogger(name)
    
    # 避免重复添加handler
    if logger.handlers:
        return logger
    
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    
    # 默认格式
    if format_string is None:
        format_string = '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
    
    formatter = logging.Formatter(format_string, datefmt='%Y-%m-%d %H:%M:%S')
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str = "seqconjector") -> logging.Logger:
    """
    获取logger实例，如果不存在则创建
    
    Args:
        name: logger名称
    
    Returns:
        Logger实例
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        # 如果没有配置，使用默认配置
        return setup_logger(name)
    return logger


# 模块级别的便捷函数
_default_logger = None


def log_info(message: str):
    """记录INFO级别消息"""
    global _default_logger
    if _default_logger is None:
        _default_logger = get_logger()
    _default_logger.info(message)


def log_debug(message: str):
    """记录DEBUG级别消息"""
    global _default_logger
    if _default_logger is None:
        _default_logger = get_logger()
    _default_logger.debug(message)


def log_warning(message: str):
    """记录WARNING级别消息"""
    global _default_logger
    if _default_logger is None:
        _default_logger = get_logger()
    _default_logger.warning(message)


def log_error(message: str):
    """记录ERROR级别消息"""
    global _default_logger
    if _default_logger is None:
        _default_logger = get_logger()
    _default_logger.error(message)


class LoggerAdapter:
    """
    适配器类：可以逐步将print替换为logging
    在过渡期间，可以选择性地同时输出到print和logging
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None, also_print: bool = False):
        self.logger = logger or get_logger()
        self.also_print = also_print
    
    def info(self, message: str):
        self.logger.info(message)
        if self.also_print:
            print(message)
    
    def debug(self, message: str):
        self.logger.debug(message)
        if self.also_print:
            print(f"[DEBUG] {message}")
    
    def warning(self, message: str):
        self.logger.warning(message)
        if self.also_print:
            print(f"[WARNING] {message}")
    
    def error(self, message: str):
        self.logger.error(message)
        if self.also_print:
            print(f"[ERROR] {message}")
    
    def __call__(self, message: str):
        """允许像print一样调用"""
        self.info(message)

