#!/usr/bin/env bash
# ============================================================
#  Desert Segmentation — Full Pipeline Runner
#  Run this from the desert_seg/ directory
#  Usage: bash run_all.sh
# ============================================================

set -e   # exit on error

echo ""
echo "============================================================"
echo "  🏜  Desert Segmentation Pipeline"
echo "============================================================"
echo ""

# ── Step 0: Install deps ─────────────────────────────────────
echo "[0/5] Installing dependencies..."
pip install -r requirements.txt -q
echo "  ✅ Dependencies installed"
echo ""

# ── Step 1: Generate synthetic data ──────────────────────────
echo "[1/5] Generating synthetic desert dataset..."
python src/generate_data.py \
    --train 600 \
    --val   150 \
    --test  100 \
    --out   data
echo "  ✅ Dataset ready"
echo ""

# ── Step 2: Train ─────────────────────────────────────────────
echo "[2/5] Training SegFormer-B2..."
python src/train.py \
    --data     data \
    --out      outputs \
    --epochs   50 \
    --batch    4 \
    --accum    4 \
    --lr       6e-5 \
    --warmup   3 \
    --variant  b2 \
    --workers  2 \
    --patience 10
echo "  ✅ Training complete"
echo ""

# ── Step 3: Evaluate ─────────────────────────────────────────
echo "[3/5] Evaluating on validation set..."
python src/evaluate.py \
    --ckpt    outputs/checkpoints/best_model.pth \
    --data    data \
    --split   val \
    --out     outputs \
    --variant b2 \
    --batch   4 \
    --n_vis   12 \
    --n_fail  6
echo "  ✅ Evaluation complete → outputs/evaluation/"
echo ""

# ── Step 4: ONNX export & benchmark ──────────────────────────
echo "[4/5] Exporting to ONNX and benchmarking..."
python src/export.py \
    --ckpt    outputs/checkpoints/best_model.pth \
    --out     outputs \
    --variant b2 \
    --n_runs  50
echo "  ✅ ONNX export done → outputs/desert_seg.onnx"
echo ""

# ── Step 5: Launch API server ─────────────────────────────────
echo "[5/5] Launching inference API server..."
echo "  Server → http://localhost:8000"
echo "  Docs   → http://localhost:8000/docs"
echo "  Press Ctrl+C to stop"
echo ""
python api/server.py \
    --ckpt    outputs/checkpoints/best_model.pth \
    --variant b2 \
    --host    0.0.0.0 \
    --port    8000
