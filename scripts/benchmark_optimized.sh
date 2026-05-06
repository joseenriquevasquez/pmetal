#!/usr/bin/env bash
#
# PMetal Optimized Benchmark vs mlx_lm.lora
#
# Compares PMetal's default training path with targeted --no-* toggles
# against an mlx_lm.lora baseline.
#

set -euo pipefail

# Configuration - match mlx_lm settings
MODEL="${MODEL:-Qwen/Qwen3-0.6B}"
SAMPLES=100
BATCH_SIZE=4
MAX_SEQ_LEN=2048
LEARNING_RATE="2e-4"
LORA_RANK=16
ITERS=25

# Directories
PMETAL="${PMETAL_BIN:-./target/release/pmetal}"
DATA_DIR="./mlx_lm_data"
OUTPUT_BASE="./output_bench"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

log_info()    { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[PASS]${NC} $1"; }
log_section() { echo -e "\n${CYAN}════════════════════════════════════════${NC}"; echo -e "${CYAN}$1${NC}"; echo -e "${CYAN}════════════════════════════════════════${NC}"; }

# Results file
RESULTS_FILE="/tmp/benchmark_results.txt"
echo "" > "$RESULTS_FILE"

log_section "PMetal Optimization Toggle Benchmark"

echo ""
log_info "Model:      $MODEL"
log_info "Samples:    $SAMPLES"
log_info "Batch Size: $BATCH_SIZE"
log_info "Max Seq:    $MAX_SEQ_LEN"

# Ensure data exists
if [ ! -f "$DATA_DIR/train.jsonl" ] || [ ! -f "./output_bench_pmetal/train.jsonl" ]; then
    log_info "Dataset not found, running setup..."
    "$SCRIPT_DIR/benchmark_pmetal_vs_mlx.sh" --samples $SAMPLES 2>&1 | tail -20
fi

# ============================================================================
log_section "Baseline: mlx_lm.lora"
# ============================================================================

MLX_OUTPUT="${OUTPUT_BASE}_mlx_opt"
mkdir -p "$MLX_OUTPUT"

log_info "Running mlx_lm.lora baseline..."

MLX_START=$(python3 -c "import time; print(time.time())")

python3 -m mlx_lm lora \
    --model "$MODEL" \
    --train \
    --data "$DATA_DIR" \
    --batch-size $BATCH_SIZE \
    --iters $ITERS \
    --learning-rate $LEARNING_RATE \
    --max-seq-length $MAX_SEQ_LEN \
    --num-layers -1 \
    --adapter-path "$MLX_OUTPUT/adapters" \
    --steps-per-report 10 \
    --val-batches 0 \
    2>&1 | tee "$MLX_OUTPUT/training.log" | grep -E "(Iter|Train loss|Tokens/sec|completed)" || true

MLX_END=$(python3 -c "import time; print(time.time())")
MLX_TIME=$(python3 -c "print(f'{$MLX_END - $MLX_START:.2f}')")
MLX_TOKENS=$(grep -oE "Tokens/sec [0-9.]+" "$MLX_OUTPUT/training.log" | tail -1 | awk '{print $2}' || echo "2000")
MLX_LOSS=$(grep -oE "Train loss [0-9.]+" "$MLX_OUTPUT/training.log" | tail -1 | awk '{print $3}' || echo "N/A")

echo "mlx_lm|$MLX_TIME|$MLX_TOKENS|$MLX_LOSS" >> "$RESULTS_FILE"
log_success "mlx_lm.lora: ${MLX_TIME}s, ${MLX_TOKENS} tok/s, loss=${MLX_LOSS}"

# ============================================================================
log_section "PMetal: Defaults"
# ============================================================================

PM_BASE_OUTPUT="${OUTPUT_BASE}_pmetal_default"
mkdir -p "$PM_BASE_OUTPUT"

log_info "Running PMetal with default optimizations..."

PM_BASE_START=$(python3 -c "import time; print(time.time())")

$PMETAL train \
    --model "$MODEL" \
    --dataset "./output_bench_pmetal/train.jsonl" \
    --output "$PM_BASE_OUTPUT" \
    --lora-r $LORA_RANK \
    --learning-rate $LEARNING_RATE \
    --batch-size $BATCH_SIZE \
    --epochs 1 \
    --max-seq-len $MAX_SEQ_LEN \
    --log-metrics "$PM_BASE_OUTPUT/metrics.jsonl" \
    2>&1 | tee "$PM_BASE_OUTPUT/training.log" | grep -E "(Step|loss|tokens/s|complete)" || true

PM_BASE_END=$(python3 -c "import time; print(time.time())")
PM_BASE_TIME=$(python3 -c "print(f'{$PM_BASE_END - $PM_BASE_START:.2f}')")
PM_BASE_TOKENS=$(grep -oE "tokens/s=[0-9]+" "$PM_BASE_OUTPUT/training.log" | tail -1 | sed 's/tokens\/s=//' || echo "0")
PM_BASE_LOSS=$(grep "Final Loss" "$PM_BASE_OUTPUT/training.log" | awk '{print $3}' || echo "N/A")

echo "pmetal_default|$PM_BASE_TIME|$PM_BASE_TOKENS|$PM_BASE_LOSS" >> "$RESULTS_FILE"
log_success "PMetal default: ${PM_BASE_TIME}s, ${PM_BASE_TOKENS} tok/s, loss=${PM_BASE_LOSS}"

# ============================================================================
log_section "PMetal: Without Fused Training Step"
# ============================================================================

PM_FUSED_OUTPUT="${OUTPUT_BASE}_pmetal_no_fused"
mkdir -p "$PM_FUSED_OUTPUT"

log_info "Running PMetal with --no-fused..."

PM_FUSED_START=$(python3 -c "import time; print(time.time())")

$PMETAL train \
    --model "$MODEL" \
    --dataset "./output_bench_pmetal/train.jsonl" \
    --output "$PM_FUSED_OUTPUT" \
    --lora-r $LORA_RANK \
    --learning-rate $LEARNING_RATE \
    --batch-size $BATCH_SIZE \
    --epochs 1 \
    --max-seq-len $MAX_SEQ_LEN \
    --no-fused \
    --log-metrics "$PM_FUSED_OUTPUT/metrics.jsonl" \
    2>&1 | tee "$PM_FUSED_OUTPUT/training.log" | grep -E "(Step|loss|tokens/s|complete)" || true

PM_FUSED_END=$(python3 -c "import time; print(time.time())")
PM_FUSED_TIME=$(python3 -c "print(f'{$PM_FUSED_END - $PM_FUSED_START:.2f}')")
PM_FUSED_TOKENS=$(grep -oE "tokens/s=[0-9]+" "$PM_FUSED_OUTPUT/training.log" | tail -1 | sed 's/tokens\/s=//' || echo "0")
PM_FUSED_LOSS=$(grep "Final Loss" "$PM_FUSED_OUTPUT/training.log" | awk '{print $3}' || echo "N/A")

echo "pmetal_no_fused|$PM_FUSED_TIME|$PM_FUSED_TOKENS|$PM_FUSED_LOSS" >> "$RESULTS_FILE"
log_success "PMetal no-fused: ${PM_FUSED_TIME}s, ${PM_FUSED_TOKENS} tok/s, loss=${PM_FUSED_LOSS}"

# ============================================================================
log_section "PMetal: Without Metal Fused Optimizer"
# ============================================================================

PM_METAL_OUTPUT="${OUTPUT_BASE}_pmetal_no_metal_opt"
mkdir -p "$PM_METAL_OUTPUT"

log_info "Running PMetal with --no-metal-fused-optimizer..."

PM_METAL_START=$(python3 -c "import time; print(time.time())")

$PMETAL train \
    --model "$MODEL" \
    --dataset "./output_bench_pmetal/train.jsonl" \
    --output "$PM_METAL_OUTPUT" \
    --lora-r $LORA_RANK \
    --learning-rate $LEARNING_RATE \
    --batch-size $BATCH_SIZE \
    --epochs 1 \
    --max-seq-len $MAX_SEQ_LEN \
    --no-metal-fused-optimizer \
    --log-metrics "$PM_METAL_OUTPUT/metrics.jsonl" \
    2>&1 | tee "$PM_METAL_OUTPUT/training.log" | grep -E "(Step|loss|tokens/s|complete)" || true

PM_METAL_END=$(python3 -c "import time; print(time.time())")
PM_METAL_TIME=$(python3 -c "print(f'{$PM_METAL_END - $PM_METAL_START:.2f}')")
PM_METAL_TOKENS=$(grep -oE "tokens/s=[0-9]+" "$PM_METAL_OUTPUT/training.log" | tail -1 | sed 's/tokens\/s=//' || echo "0")
PM_METAL_LOSS=$(grep "Final Loss" "$PM_METAL_OUTPUT/training.log" | awk '{print $3}' || echo "N/A")

echo "pmetal_no_metal_opt|$PM_METAL_TIME|$PM_METAL_TOKENS|$PM_METAL_LOSS" >> "$RESULTS_FILE"
log_success "PMetal no-metal-opt: ${PM_METAL_TIME}s, ${PM_METAL_TOKENS} tok/s, loss=${PM_METAL_LOSS}"

# ============================================================================
log_section "PMetal: Without Sequence Packing + FlashAttention"
# ============================================================================

PM_PACK_OUTPUT="${OUTPUT_BASE}_pmetal_no_pack_fa"
mkdir -p "$PM_PACK_OUTPUT"

log_info "Running PMetal with --no-sequence-packing --no-flash-attention..."

PM_PACK_START=$(python3 -c "import time; print(time.time())")

$PMETAL train \
    --model "$MODEL" \
    --dataset "./output_bench_pmetal/train.jsonl" \
    --output "$PM_PACK_OUTPUT" \
    --lora-r $LORA_RANK \
    --learning-rate $LEARNING_RATE \
    --batch-size $BATCH_SIZE \
    --epochs 1 \
    --max-seq-len $MAX_SEQ_LEN \
    --no-sequence-packing \
    --no-flash-attention \
    --log-metrics "$PM_PACK_OUTPUT/metrics.jsonl" \
    2>&1 | tee "$PM_PACK_OUTPUT/training.log" | grep -E "(Step|loss|tokens/s|complete|Packing)" || true

PM_PACK_END=$(python3 -c "import time; print(time.time())")
PM_PACK_TIME=$(python3 -c "print(f'{$PM_PACK_END - $PM_PACK_START:.2f}')")
PM_PACK_TOKENS=$(grep -oE "tokens/s=[0-9]+" "$PM_PACK_OUTPUT/training.log" | tail -1 | sed 's/tokens\/s=//' || echo "0")
PM_PACK_LOSS=$(grep "Final Loss" "$PM_PACK_OUTPUT/training.log" | awk '{print $3}' || echo "N/A")

echo "pmetal_no_pack_fa|$PM_PACK_TIME|$PM_PACK_TOKENS|$PM_PACK_LOSS" >> "$RESULTS_FILE"
log_success "PMetal no-pack/FA: ${PM_PACK_TIME}s, ${PM_PACK_TOKENS} tok/s, loss=${PM_PACK_LOSS}"

# ============================================================================
log_section "PMetal: Conservative All Optimizations Disabled"
# ============================================================================

PM_ALL_OUTPUT="${OUTPUT_BASE}_pmetal_conservative"
mkdir -p "$PM_ALL_OUTPUT"

log_info "Running PMetal with all optional training optimizations disabled..."

PM_ALL_START=$(python3 -c "import time; print(time.time())")

$PMETAL train \
    --model "$MODEL" \
    --dataset "./output_bench_pmetal/train.jsonl" \
    --output "$PM_ALL_OUTPUT" \
    --lora-r $LORA_RANK \
    --learning-rate $LEARNING_RATE \
    --batch-size $BATCH_SIZE \
    --epochs 1 \
    --max-seq-len $MAX_SEQ_LEN \
    --no-fused \
    --no-metal-fused-optimizer \
    --no-sequence-packing \
    --no-flash-attention \
    --log-metrics "$PM_ALL_OUTPUT/metrics.jsonl" \
    2>&1 | tee "$PM_ALL_OUTPUT/training.log" | grep -E "(Step|loss|tokens/s|complete|Packing)" || true

PM_ALL_END=$(python3 -c "import time; print(time.time())")
PM_ALL_TIME=$(python3 -c "print(f'{$PM_ALL_END - $PM_ALL_START:.2f}')")
PM_ALL_TOKENS=$(grep -oE "tokens/s=[0-9]+" "$PM_ALL_OUTPUT/training.log" | tail -1 | sed 's/tokens\/s=//' || echo "0")
PM_ALL_LOSS=$(grep "Final Loss" "$PM_ALL_OUTPUT/training.log" | awk '{print $3}' || echo "N/A")

echo "pmetal_conservative|$PM_ALL_TIME|$PM_ALL_TOKENS|$PM_ALL_LOSS" >> "$RESULTS_FILE"
log_success "PMetal conservative: ${PM_ALL_TIME}s, ${PM_ALL_TOKENS} tok/s, loss=${PM_ALL_LOSS}"

# ============================================================================
log_section "RESULTS COMPARISON"
# ============================================================================

echo ""
echo "┌─────────────────────────────────────────────────────────────────────────────┐"
echo "│                         BENCHMARK RESULTS                                   │"
echo "├─────────────────────────────────────────────────────────────────────────────┤"
printf "│  %-30s %12s %14s %12s │\n" "Configuration" "Time (s)" "Tokens/sec" "Loss"
echo "├─────────────────────────────────────────────────────────────────────────────┤"

while IFS='|' read -r config time tokens loss; do
    [ -z "$config" ] && continue
    case "$config" in
        mlx_lm)
            printf "│  %-30s %12s %14s %12s │\n" "mlx_lm.lora (baseline)" "$time" "$tokens" "$loss"
            echo "├─────────────────────────────────────────────────────────────────────────────┤"
            ;;
        pmetal_default)
            printf "│  %-30s %12s %14s %12s │\n" "PMetal: defaults" "$time" "$tokens" "$loss"
            ;;
        pmetal_no_fused)
            printf "│  %-30s %12s %14s %12s │\n" "PMetal: --no-fused" "$time" "$tokens" "$loss"
            ;;
        pmetal_no_metal_opt)
            printf "│  %-30s %12s %14s %12s │\n" "PMetal: --no-metal-opt" "$time" "$tokens" "$loss"
            ;;
        pmetal_no_pack_fa)
            printf "│  %-30s %12s %14s %12s │\n" "PMetal: --no-pack/FA" "$time" "$tokens" "$loss"
            ;;
        pmetal_conservative)
            printf "│  %-30s %12s %14s %12s │\n" "PMetal: conservative" "$time" "$tokens" "$loss"
            ;;
    esac
done < "$RESULTS_FILE"

echo "└─────────────────────────────────────────────────────────────────────────────┘"

# Calculate speedups
echo ""
log_info "Speedup comparison vs mlx_lm.lora baseline ($MLX_TIME s):"
echo ""

while IFS='|' read -r config time tokens loss; do
    [ -z "$config" ] && continue
    [ "$config" = "mlx_lm" ] && continue

    if [ -n "$time" ] && [ "$time" != "0" ]; then
        SPEEDUP=$(python3 -c "print(f'{$MLX_TIME / $time:.2f}')")
        RELATIVE=$(python3 -c "t=$MLX_TIME/$time; print('FASTER' if t > 1 else 'SLOWER')")
        printf "  %-30s %6sx  (%s)\n" "$config" "$SPEEDUP" "$RELATIVE"
    fi
done < "$RESULTS_FILE"

echo ""
log_section "Summary"
echo ""

# Find best PMetal config
BEST_TIME=999999
BEST_CONFIG=""
while IFS='|' read -r config time tokens loss; do
    [ -z "$config" ] && continue
    [ "$config" = "mlx_lm" ] && continue

    if [ -n "$time" ]; then
        IS_BETTER=$(python3 -c "print(1 if $time < $BEST_TIME else 0)")
        if [ "$IS_BETTER" = "1" ]; then
            BEST_TIME=$time
            BEST_CONFIG=$config
        fi
    fi
done < "$RESULTS_FILE"

echo "Best PMetal configuration: $BEST_CONFIG (${BEST_TIME}s)"
FINAL_SPEEDUP=$(python3 -c "print(f'{$MLX_TIME / $BEST_TIME:.2f}')")

IS_FASTER=$(python3 -c "print(1 if $BEST_TIME < $MLX_TIME else 0)")
if [ "$IS_FASTER" = "1" ]; then
    echo -e "${GREEN}PMetal ($BEST_CONFIG) is ${FINAL_SPEEDUP}x FASTER than mlx_lm.lora!${NC}"
else
    SLOWER=$(python3 -c "print(f'{$BEST_TIME / $MLX_TIME:.2f}')")
    echo -e "${YELLOW}mlx_lm.lora is ${SLOWER}x faster than best PMetal config${NC}"
fi
echo ""
