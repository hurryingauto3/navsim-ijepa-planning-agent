#!/bin/bash
# Monitor NAVSIM Evaluation Progress
# Usage: ./monitor_eval.sh

echo "=================================="
echo "NAVSIM Evaluation Monitor"
echo "=================================="
echo ""

# Find latest evaluation directories
NAVMINI_DIR=$(ls -td /scratch/ah7072/experiments/eval_baseline_epoch39_navmini_* 2>/dev/null | head -1)
NAVTEST_DIR=$(ls -td /scratch/ah7072/experiments/eval_baseline_epoch39_2* 2>/dev/null | head -1)

# Check job status
echo "=== Job Status ==="
squeue -u $USER -o "%.10i %.12P %.20j %.8T %.10M %.6D %R"
echo ""

# navmini progress
if [ -n "$NAVMINI_DIR" ] && [ -f "$NAVMINI_DIR/run_pdm_score.log" ]; then
    NAVMINI_COUNT=$(grep -c "Processing stage one" "$NAVMINI_DIR/run_pdm_score.log" 2>/dev/null || echo "0")
    NAVMINI_PCT=$((NAVMINI_COUNT * 100 / 388))
    echo "=== navmini Progress ==="
    echo "  Scenarios: $NAVMINI_COUNT / 388 ($NAVMINI_PCT%)"
    echo "  Log: $NAVMINI_DIR/run_pdm_score.log"
    tail -3 "$NAVMINI_DIR/run_pdm_score.log" 2>/dev/null | grep "Processing"
    echo ""
fi

# navtest progress
if [ -n "$NAVTEST_DIR" ] && [ -f "$NAVTEST_DIR/run_pdm_score.log" ]; then
    NAVTEST_COUNT=$(grep -c "Processing stage one" "$NAVTEST_DIR/run_pdm_score.log" 2>/dev/null || echo "0")
    # navtest has ~1500 scenarios
    NAVTEST_PCT=$((NAVTEST_COUNT * 100 / 1500))
    echo "=== navtest Progress ==="
    echo "  Scenarios: $NAVTEST_COUNT / ~1500 ($NAVTEST_PCT%)"
    echo "  Log: $NAVTEST_DIR/run_pdm_score.log"
    tail -3 "$NAVTEST_DIR/run_pdm_score.log" 2>/dev/null | grep "Processing"
    echo ""
fi

# Check for errors
echo "=== Recent Errors (if any) ==="
tail -5 /scratch/ah7072/navsim_workspace/exp/logs/eval_*.err 2>/dev/null | grep -i "error\|exception\|traceback" | tail -3 || echo "  No errors"
echo ""

# Check for completed results
echo "=== Results ==="
if [ -n "$NAVMINI_DIR" ]; then
    if ls "$NAVMINI_DIR"/*.parquet >/dev/null 2>&1 || ls "$NAVMINI_DIR"/*metrics*.json >/dev/null 2>&1; then
        echo "✅ navmini COMPLETED - results found in $NAVMINI_DIR"
    else
        echo "⏳ navmini still running..."
    fi
fi
if [ -n "$NAVTEST_DIR" ]; then
    if ls "$NAVTEST_DIR"/*.parquet >/dev/null 2>&1 || ls "$NAVTEST_DIR"/*metrics*.json >/dev/null 2>&1; then
        echo "✅ navtest COMPLETED - results found in $NAVTEST_DIR"
    else
        echo "⏳ navtest still running..."
    fi
fi

echo ""
echo "Run: watch -n 30 ./monitor_eval.sh   (auto-refresh every 30s)"
