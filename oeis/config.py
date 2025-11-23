"""
集中管理 SeqConjector 项目的所有配置参数
"""

class Config:
    """全局配置类"""
    
    # ==================== Interpreter 配置 ====================
    # 执行预算参数
    BUDGET_T0 = 10           # 基础时间预算
    BUDGET_T_STEP = 3        # 每个元素的时间步长
    
    # 宽松模式预算（用于数据生成，允许更复杂的程序）
    BUDGET_T0_LOOSE = 20
    BUDGET_T_STEP_LOOSE = 5
    
    # ==================== Beam Search 配置 ====================
    # 优化后的默认值（基于性能测试）
    DEFAULT_BEAM_SIZE = 16           # 从256降到16，提速12倍，成功率不变
    DEFAULT_MAX_STEPS = 96           # 保持不变
    DEFAULT_TIME_LIMIT = 5.0         # 搜索时间限制
    
    # ==================== Moonshine 配置 ====================
    # 严格匹配的前k项
    K_STRICT = 3
    
    # 相对误差阈值
    RELERR0 = 2e-3           # 初始阈值
    RELERR_STEP = 1e-3       # 每步增长
    RELERR_HI = 0.10         # 高阈值
    
    # ==================== 特征提取配置 ====================
    FEATURE_DIM = 54         # enhanced_features 输出维度
    
    # ==================== 模型配置 ====================
    # Transformer参数（针对2M数据优化，适配6GB显存）
    D_MODEL = 256            # 从256增加到384，提升表达能力
    NHEAD = 4                # 从8调整为6（需能被d_model整除）
    NLAYERS = 6              # 保持6层
    D_FF = 1024              # 从1024增加到1536（约4倍d_model）
    DROPOUT = 0.1
    CTX_LEN = 64
    
    # ==================== 训练配置 ====================
    DEFAULT_LR = 2e-4        # 从3e-4降低到2e-4，更稳定
    DEFAULT_BATCH_SIZE = 64  # 从128降低到64，适配6GB显存
    DEFAULT_STEPS = 30000    # 从20000增加到30000，充分利用2M数据
    WARMUP_STEPS = 1000      # 学习率warmup步数
    GRADIENT_ACCUMULATION = 2  # 梯度累积步数，有效batch_size = 64 * 2 = 128
    WEIGHT_DECAY = 0.01      # 权重衰减
    MIN_LR = 1e-5            # 最小学习率（用于学习率调度）
    
    # ==================== 数据生成配置 ====================
    # 序列长度
    MIN_SEQ_LEN = 7
    MAX_SEQ_LEN_SYNTHETIC = 30
    
    # 程序生成参数
    MAX_PROG_DEPTH = 3
    MAX_PROG_LENGTH = 8
    
    # Moonshine数据比例
    DEFAULT_MOONSHINE_PROB = 0.1
    
    # ==================== 路径配置 ====================
    OEIS_DATA_DIR = "oeis_seq_labeled/formula_true"
    DEFAULT_DATA_GEN_DIR = "data_gen"
    DEFAULT_CHECKPOINT = "ckpt.pt"
    
    # ==================== 整数常量范围 ====================
    # 词表范围（不变）
    INT_MIN = -16
    INT_MAX = 16
    
    # 训练时采样的INSERT常量范围（更小，避免超出词表）
    INSERT_CONST_MIN = -8
    INSERT_CONST_MAX = 8
    
    # 注意：智能拟合支持 ±10000 范围的系数，不受词表限制
    
    # ==================== 模板匹配配置 ====================
    # 智能拟合相关
    SMART_FIT_R2_THRESHOLD_QUAD = 0.90   # 二次拟合R²阈值
    SMART_FIT_R2_THRESHOLD_LINEAR = 0.95 # 线性拟合R²阈值
    SMART_FIT_R2_THRESHOLD_PURE = 0.98   # 纯缩放/偏移R²阈值
    SMART_FIT_MAX_COEFF = 10000          # 系数最大绝对值（突破±16限制）
    SMART_FIT_INTEGER_TOLERANCE = 0.05   # 浮点→整数容差
    
    # 模板尝试策略
    MAX_FEATURE_TEMPLATES = None  # None=尝试所有模板（成本极低<1ms）
    
    # ==================== 测试配置 ====================
    TEST_TIMEOUT = 1.0       # run_test中的超时（秒）
    TEST_WORKERS = 0         # 0表示使用CPU数量
    
    # ==================== Split配置 ====================
    # compute_split的规则
    SHORT_SEQ_THRESHOLD = 10  # 短序列阈值
    SHORT_SEQ_VALIDATE = 2    # 短序列的验证项数
    LONG_SEQ_VALIDATE_RATIO = 0.3  # 长序列的验证比例
    
    @classmethod
    def get_interpreter_config(cls, strict=True, loose_budget=False):
        """
        获取Interpreter的ExecConfig
        
        Args:
            strict: 是否使用严格模式
            loose_budget: 是否使用宽松预算（用于数据生成）
        
        Returns:
            ExecConfig实例
        """
        from .interpreter import ExecConfig
        
        if loose_budget:
            return ExecConfig(
                strict=strict,
                t0=cls.BUDGET_T0_LOOSE,
                t_step=cls.BUDGET_T_STEP_LOOSE
            )
        else:
            return ExecConfig(
                strict=strict,
                t0=cls.BUDGET_T0,
                t_step=cls.BUDGET_T_STEP
            )
    
    @classmethod
    def get_model_config(cls):
        """获取模型配置字典"""
        return {
            "d_model": cls.D_MODEL,
            "nhead": cls.NHEAD,
            "nlayers": cls.NLAYERS,
            "d_ff": cls.D_FF,
            "dropout": cls.DROPOUT,
            "ctx_len": cls.CTX_LEN,
            "feat_dim": cls.FEATURE_DIM
        }

