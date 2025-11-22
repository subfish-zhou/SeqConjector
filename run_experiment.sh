#!/bin/bash

# =============================================================================
# 序列关系批量实验脚本
# =============================================================================

# =============================================================================
# 配置参数（可修改）
# =============================================================================

# 输入文件路径（相对于项目根目录）
A_FILE="oeis_seq_labeled/formula_false/algebraic_number_theory.jsonl"
B_FILE="oeis_seq_labeled/formula_false/graph_theory.jsonl"

# 选取数量（留空或设为0表示全部加载）
A_COUNT=""  # 留空 = 全部，或设置如 100
B_COUNT=""  # 留空 = 全部，或设置如 100

# 输出目录
OUTPUT_DIR="experiment_results"

# 实验配置
BEAM_WIDTH=16          # Beam搜索宽度
TIME_LIMIT=20.0        # 单个任务超时时间(秒)
MAX_STEPS=96           # 最大搜索步数
PARALLEL_WORKERS=1     # 并行worker数量（CPU）
USE_GPU=0              # 是否使用GPU (0=否, 1=是)

# =============================================================================
# 脚本开始（一般不需要修改下面的内容）
# =============================================================================

echo "================================================================================"
echo "序列关系批量实验"
echo "================================================================================"
echo ""
echo "配置信息:"
echo "  A文件: $A_FILE"
echo "  B文件: $B_FILE"
echo "  A数量: ${A_COUNT:-全部}"
echo "  B数量: ${B_COUNT:-全部}"
echo "  输出目录: $OUTPUT_DIR"
echo "  Beam宽度: $BEAM_WIDTH"
echo "  超时: ${TIME_LIMIT}s"
echo "  并行数: $PARALLEL_WORKERS"
echo "  GPU: $([ $USE_GPU -eq 1 ] && echo '是' || echo '否')"
echo ""

# 检查文件是否存在
if [ ! -f "$A_FILE" ]; then
    echo "错误: A文件不存在: $A_FILE"
    exit 1
fi

if [ ! -f "$B_FILE" ]; then
    echo "错误: B文件不存在: $B_FILE"
    exit 1
fi

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 生成输出文件名（基于输入文件名和时间戳）
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
A_NAME=$(basename "$A_FILE" .jsonl)
B_NAME=$(basename "$B_FILE" .jsonl)
OUTPUT_FILE="$OUTPUT_DIR/${A_NAME}_to_${B_NAME}_${TIMESTAMP}.jsonl"
STATS_FILE="$OUTPUT_DIR/${A_NAME}_to_${B_NAME}_${TIMESTAMP}_stats.txt"

# 构建命令
CMD="python experiment_batch.py"
CMD="$CMD --A-file \"$A_FILE\""
CMD="$CMD --B-file \"$B_FILE\""
CMD="$CMD --output \"$OUTPUT_FILE\""
CMD="$CMD --beam-width $BEAM_WIDTH"
CMD="$CMD --time-limit $TIME_LIMIT"
CMD="$CMD --max-steps $MAX_STEPS"
CMD="$CMD --parallel-workers $PARALLEL_WORKERS"

# 添加可选参数
if [ -n "$A_COUNT" ] && [ "$A_COUNT" -gt 0 ] 2>/dev/null; then
    CMD="$CMD --A-count $A_COUNT"
fi

if [ -n "$B_COUNT" ] && [ "$B_COUNT" -gt 0 ] 2>/dev/null; then
    CMD="$CMD --B-count $B_COUNT"
fi

if [ $USE_GPU -eq 1 ]; then
    CMD="$CMD --device cuda"
else
    CMD="$CMD --device cpu"
fi

# 显示完整命令
echo "================================================================================"
echo "执行命令:"
echo "$CMD"
echo "================================================================================"
echo ""

# 确认执行
read -p "按回车键开始实验，或按Ctrl+C取消... " DUMMY

# 执行实验
eval $CMD

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "================================================================================"
    echo "实验完成！"
    echo "================================================================================"
    echo "  结果文件: $OUTPUT_FILE"
    echo "  统计文件: $STATS_FILE"
    echo ""
    
    # 显示简要统计
    if [ -f "$OUTPUT_FILE" ]; then
        TOTAL=$(wc -l < "$OUTPUT_FILE")
        echo "  成功案例: $TOTAL 个"
    fi
    
    if [ -f "$STATS_FILE" ]; then
        echo ""
        echo "统计摘要:"
        head -20 "$STATS_FILE"
    fi
else
    echo ""
    echo "================================================================================"
    echo "实验失败，退出码: $EXIT_CODE"
    echo "================================================================================"
    exit $EXIT_CODE
fi

