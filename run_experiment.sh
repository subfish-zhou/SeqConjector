#!/bin/bash

# =============================================================================
# Sequence Relation Batch Experiment Script
# =============================================================================

# Auto-detect system resources
echo "=========================================="
echo "Detecting system resources..."
echo "=========================================="

# Detect CPU cores
CPU_CORES=$(nproc 2>/dev/null || echo "4")
echo "CPU cores: $CPU_CORES"

# Detect available memory (in MB)
TOTAL_MEM=$(free -m 2>/dev/null | awk '/^Mem:/{print $2}')
if [ -z "$TOTAL_MEM" ]; then
    TOTAL_MEM=8000  # Default fallback: 8GB
fi
echo "Total memory: ${TOTAL_MEM}MB"

# Calculate optimal workers
# Each worker needs ~500MB (model 123MB + Python 200MB + buffer 177MB)
WORKER_MEM=500
# Leave 20% memory for system
AVAILABLE_MEM=$((TOTAL_MEM * 80 / 100))
MAX_WORKERS_BY_MEM=$((AVAILABLE_MEM / WORKER_MEM))
MAX_WORKERS_BY_CPU=$CPU_CORES

# Take the minimum to avoid over-subscription
if [ $MAX_WORKERS_BY_MEM -lt $MAX_WORKERS_BY_CPU ]; then
    AUTO_WORKERS=$MAX_WORKERS_BY_MEM
    LIMIT_REASON="memory"
else
    AUTO_WORKERS=$MAX_WORKERS_BY_CPU
    LIMIT_REASON="CPU cores"
fi

# Ensure at least 1 worker, cap at 128
AUTO_WORKERS=$((AUTO_WORKERS < 1 ? 1 : AUTO_WORKERS))
AUTO_WORKERS=$((AUTO_WORKERS > 128 ? 128 : AUTO_WORKERS))

echo "Available memory for workers: ${AVAILABLE_MEM}MB"
echo "Max workers by memory: $MAX_WORKERS_BY_MEM"
echo "Max workers by CPU: $MAX_WORKERS_BY_CPU"
echo "==> Selected workers: $AUTO_WORKERS (limited by $LIMIT_REASON)"
echo ""

# Configuration Parameters
A_FILE="oeis_seq_labeled/modular_forms.jsonl"
B_FILE="oeis_seq_labeled/graph_theory.jsonl"
A_COUNT=""  # Empty = all, or set number like 100
B_COUNT=""  # Empty = all, or set number like 100
OUTPUT_DIR="experiment_results"

# Experiment Settings (optimized for trained model capacity)
BEAM_WIDTH=16
TIME_LIMIT=5.0
MAX_STEPS=8
PARALLEL_WORKERS=$AUTO_WORKERS  # Auto-detected
USE_GPU=0  # CPU parallel mode

# Check input files
if [ ! -f "$A_FILE" ]; then
    echo "ERROR: File not found: $A_FILE"
    exit 1
fi

if [ ! -f "$B_FILE" ]; then
    echo "ERROR: File not found: $B_FILE"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Generate output filenames
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
A_NAME=$(basename "$A_FILE" .jsonl)
B_NAME=$(basename "$B_FILE" .jsonl)
OUTPUT_FILE="$OUTPUT_DIR/${A_NAME}_to_${B_NAME}_${TIMESTAMP}.jsonl"
STATS_FILE="$OUTPUT_DIR/${A_NAME}_to_${B_NAME}_${TIMESTAMP}_stats.txt"

# Build command
CMD="python experiment_batch.py"
CMD="$CMD --A-file \"$A_FILE\""
CMD="$CMD --B-file \"$B_FILE\""
CMD="$CMD --output \"$OUTPUT_FILE\""
CMD="$CMD --beam $BEAM_WIDTH"
CMD="$CMD --time-limit $TIME_LIMIT"
CMD="$CMD --max-steps $MAX_STEPS"
CMD="$CMD --workers $PARALLEL_WORKERS"

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

# Run experiment
echo "Running: $CMD"
eval $CMD

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo "Experiment completed successfully"
    echo "Output: $OUTPUT_FILE"
    if [ -f "$OUTPUT_FILE" ]; then
        TOTAL=$(wc -l < "$OUTPUT_FILE")
        echo "Total results: $TOTAL"
    fi
else
    echo "Experiment failed with exit code: $EXIT_CODE"
    exit $EXIT_CODE
fi

