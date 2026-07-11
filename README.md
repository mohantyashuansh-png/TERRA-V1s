# 🏜 Desert Vision — Semantic Segmentation

**PS-01DEVNOVATESAI | Computer Vision Track**

SegFormer-B2 fine-tuned for 6-class desert scene segmentation on synthetic data.

---

## 📁 Project Structure

```
desert_seg/
├── src/
│   ├── generate_data.py   # Synthetic dataset generator
│   ├── dataset.py         # PyTorch Dataset + Albumentations pipeline
│   ├── model.py           # SegFormer-B2, CombinedLoss, IoUMetric
│   ├── train.py           # Full training loop (fp16, grad accum, early stop)
│   ├── evaluate.py        # mIoU, confusion matrix, failure cases, visuals
│   └── export.py          # ONNX export + inference benchmark
├── api/
│   └── server.py          # FastAPI inference server
├── frontend/
│   └── index.html         # Demo dashboard (for frontend team)
├── data/                  # Generated dataset (auto-created)
├── outputs/               # Checkpoints, plots, evaluation (auto-created)
├── requirements.txt
└── run_all.sh             # Master pipeline script
```

---

## 🏷 Classes

| ID | Class | Color |
|----|-------|-------|
| 0 | Sky | Light Blue |
| 1 | Sand | Sandy Tan |
| 2 | Rock | Brownish Gray |
| 3 | Vegetation | Muted Green |
| 4 | Shadow | Dark Sand |
| 5 | Distant Terrain | Hazy Tan |

---

## ⚡ Quick Start (run everything)

```bash
# From desert_seg/ directory:
bash run_all.sh
```

---

## 🔧 Step-by-Step

### 0. Install dependencies
```bash
pip install -r requirements.txt
```

### 1. Generate synthetic dataset
```bash
python src/generate_data.py --train 600 --val 150 --test 100 --out data
```

### 2. Train SegFormer-B2
```bash
# RTX 2050 4GB config (fp16 + gradient accumulation)
python src/train.py \
    --data     data \
    --out      outputs \
    --epochs   50 \
    --batch    4 \
    --accum    4 \
    --lr       6e-5 \
    --warmup   3 \
    --variant  b2 \
    --patience 10
```

**Key training config for your GPU:**
- `--batch 4` — safe for 4GB VRAM with fp16
- `--accum 4` — effective batch size = 16
- Pretrained SegFormer-B2 from HuggingFace (downloads ~90MB once)

### 3. Evaluate & generate report visuals
```bash
python src/evaluate.py \
    --ckpt  outputs/checkpoints/best_model.pth \
    --data  data \
    --split val \
    --out   outputs
```
**Outputs:**
- `outputs/evaluation/metrics.json` — mIoU, per-class IoU, pixel accuracy
- `outputs/evaluation/iou_per_class.png` — IoU bar chart
- `outputs/evaluation/confusion_matrix.png` — normalised confusion matrix
- `outputs/evaluation/prediction_grid.png` — 12 overlay comparisons
- `outputs/evaluation/failure_cases.png` — 6 worst predictions + error maps
- `outputs/evaluation/miou_distribution.png` — distribution of per-image IoU
- `outputs/plots/training_curves.png` — loss + mIoU + LR curves

### 4. Export to ONNX + benchmark inference speed
```bash
python src/export.py \
    --ckpt   outputs/checkpoints/best_model.pth \
    --out    outputs \
    --n_runs 50
```

### 5. Start inference API (for frontend team)
```bash
python api/server.py \
    --ckpt outputs/checkpoints/best_model.pth \
    --port 8000
```
- API docs: http://localhost:8000/docs
- POST /predict — single image → overlay + class % + inference time
- GET /health — check model loaded
- GET /classes — class names + colors

---

## 📡 API Usage (for frontend)

```javascript
// Send image, get back segmentation
const form = new FormData();
form.append('file', imageFile);

const res  = await fetch('http://localhost:8000/predict', {
  method: 'POST', body: form
});
const data = await res.json();

// data.overlay_b64 → base64 PNG of result overlay
// data.mask_b64    → base64 PNG of color mask only
// data.class_pct   → { Sky: 28.4, Sand: 38.1, ... }
// data.inference_ms → 47.2
```

---

## 🎯 Expected Performance

| Metric | Target |
|--------|--------|
| Val mIoU | > 0.65 |
| Pixel Accuracy | > 0.85 |
| Inference (PyTorch fp16) | < 80ms |
| ONNX Speedup | 1.5-2× |

---

## 🔁 Resume Training

```bash
python src/train.py \
    --resume outputs/checkpoints/best_model.pth \
    [other args...]
```

---

## ⚡ VRAM Troubleshooting

| Issue | Fix |
|-------|-----|
| OOM during training | Reduce `--batch` to 2, keep `--accum 8` |
| OOM during eval | Reduce `--batch` to 1 |
| Slow download | HuggingFace downloads model once, cached at ~/.cache/huggingface |

---

## 📊 Report Checklist

- [x] Training loss curve (outputs/plots/training_curves.png)
- [x] Val mIoU curve (same file)
- [x] Per-class IoU bar chart
- [x] Confusion matrix
- [x] Prediction overlays (12 samples)
- [x] Failure case analysis (6 worst cases with error maps)
- [x] IoU distribution plot
- [x] Inference speed benchmark (PyTorch vs ONNX)
- [x] Live demo via frontend dashboard

---

*Built for hackathon PS-01DEVNOVATESAI | SegFormer-B2 | PyTorch 2.0 | FastAPI*
