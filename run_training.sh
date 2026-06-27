#!/bin/bash
# run_training.sh
# ===============
# Runs the full 200,000-step IQL training in chunks that resume automatically.
# Each chunk takes ~15-20 min on a laptop CPU. The script saves progress
# after every chunk so you can stop and restart safely at any time.
#
# Usage:
#   chmod +x run_training.sh
#   ./run_training.sh
#
# To resume after stopping:
#   ./run_training.sh
# (It detects the existing checkpoint and continues from where it left off.)

set -e
cd "$(dirname "$0")"

CONFIG="configs/config.yaml"
FYLLO="./fyllo.xlsx"
LOG="./data/soil_log.csv"
FIELD1="./data/TST1234_001.csv"
FIELD2="./data/TST1234_002.csv"
CKPT="./artifacts/ckpts/iql_final.pt"
BUFFER="./artifacts/buffer/offline.npz"
STEP_FILE="./artifacts/ckpts/train_step.txt"
CHUNK=25000      # steps per chunk (~12-15 min each on CPU)
TOTAL=200000     # full training target

mkdir -p artifacts/ckpts artifacts/buffer

# --- Read current step from file ---
if [ -f "$STEP_FILE" ]; then
    DONE=$(cat "$STEP_FILE")
else
    DONE=0
fi

echo "============================================"
echo "  Smart Irrigation IQL — Full Training"
echo "  Target: $TOTAL steps | Done so far: $DONE"
echo "============================================"

if [ "$DONE" -ge "$TOTAL" ]; then
    echo "Training already complete at $DONE steps."
    echo "Run: python benchmark.py --config $CONFIG --ckpt $CKPT --episodes 30"
    exit 0
fi

# --- First chunk: generate and cache the buffer ---
if [ ! -f "$BUFFER" ]; then
    echo ""
    echo "Step 1/2: Generating offline buffer (one-time, ~20 min)..."
    python -m src.train \
        --config "$CONFIG" \
        --fyllo "$FYLLO" \
        --log "$LOG" \
        --field "$FIELD1" "$FIELD2" \
        --steps 1 \
        --save-buffer "$BUFFER"
    echo "Buffer saved to $BUFFER"
fi

# --- Training chunks ---
while [ "$DONE" -lt "$TOTAL" ]; do
    REMAINING=$((TOTAL - DONE))
    THIS_CHUNK=$CHUNK
    if [ "$REMAINING" -lt "$CHUNK" ]; then
        THIS_CHUNK=$REMAINING
    fi
    NEXT=$((DONE + THIS_CHUNK))

    echo ""
    echo "Training steps $((DONE+1)) → $NEXT  (of $TOTAL)..."

    RESUME_ARGS=""
    if [ "$DONE" -gt 0 ] && [ -f "$CKPT" ]; then
        RESUME_ARGS="--resume $CKPT --resume-step $DONE"
    fi

    python -m src.train \
        --config "$CONFIG" \
        --fyllo "$FYLLO" \
        --log "$LOG" \
        --field "$FIELD1" "$FIELD2" \
        --load-buffer "$BUFFER" \
        --steps "$THIS_CHUNK" \
        $RESUME_ARGS

    DONE=$NEXT
    echo "$DONE" > "$STEP_FILE"
    echo "  ✓ Progress: $DONE / $TOTAL steps"
done

echo ""
echo "============================================"
echo "  Training complete! ($TOTAL steps)"
echo "  Checkpoint: $CKPT"
echo "============================================"
echo ""
echo "Next step — run benchmarks:"
echo "  python benchmark.py --config $CONFIG --ckpt $CKPT --episodes 30"
echo ""
echo "Then start the web app:"
echo "  cd webapp && uvicorn app:app --reload --port 8000"
