"""
Desert Segmentation — Evaluation & Visualization
Generates all report-ready visuals:
  - Per-class IoU bar chart
  - Confusion matrix
  - Prediction overlays (grid)
  - Failure case analysis
  - Per-image IoU distribution
"""

import sys
import json
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import seaborn as sns
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from dataset import DesertSegDataset, get_val_transforms, CLASS_NAMES, NUM_CLASSES
from model   import DesertSegFormer, IoUMetric

# ─── Config ───────────────────────────────────────────────────────────────────
CLASS_COLORS = [
    (135, 206, 235),  # Sky
    (210, 180, 140),  # Sand
    (105,  90,  75),  # Rock
    ( 85, 120,  60),  # Vegetation
    ( 90,  80,  65),  # Shadow
    (160, 140, 115),  # Distant Terrain
]
MEAN = np.array([0.485, 0.456, 0.406])
STD  = np.array([0.229, 0.224, 0.225])


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt',    type=str,  required=True)
    p.add_argument('--data',    type=str,  default='data')
    p.add_argument('--split',   type=str,  default='val', choices=['val', 'test'])
    p.add_argument('--out',     type=str,  default='outputs')
    p.add_argument('--variant', type=str,  default='b2')
    p.add_argument('--batch',   type=int,  default=4)
    p.add_argument('--n_vis',   type=int,  default=12, help='samples to visualise')
    p.add_argument('--n_fail',  type=int,  default=6,  help='failure cases to show')
    return p.parse_args()

# ─── Helpers ──────────────────────────────────────────────────────────────────

def denormalize(tensor):
    """C×H×W tensor → H×W×3 uint8."""
    arr = tensor.cpu().numpy().transpose(1, 2, 0)
    arr = np.clip(arr * STD + MEAN, 0, 1)
    return (arr * 255).astype(np.uint8)


def mask_to_color(mask: np.ndarray) -> np.ndarray:
    color = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for cls, col in enumerate(CLASS_COLORS):
        color[mask == cls] = col
    return color


def overlay(img: np.ndarray, mask: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Blend image with colour mask."""
    col = mask_to_color(mask)
    return (img * (1 - alpha) + col * alpha).astype(np.uint8)

# ─── Inference Loop ────────────────────────────────────────────────────────────

def run_inference(model, loader, device):
    model.eval()
    global_metric = IoUMetric()
    records = []   # list of dicts {img, gt, pred, miou_img, path}

    with torch.no_grad():
        for batch in tqdm(loader, desc='Inference', ncols=90):
            imgs  = batch['image'].to(device)
            masks = batch['mask']
            paths = batch['path']

            logits = model(imgs)
            preds  = F.interpolate(logits, size=masks.shape[-2:],
                                   mode='bilinear', align_corners=False).argmax(1)

            global_metric.update(preds.cpu(), masks)

            for i in range(len(paths)):
                # Per-image IoU
                m = IoUMetric()
                m.update(preds[i:i+1].cpu(), masks[i:i+1])
                img_results = m.compute()
                records.append({
                    'img':      imgs[i].cpu(),
                    'gt':       masks[i].cpu().numpy(),
                    'pred':     preds[i].cpu().numpy(),
                    'miou':     img_results['miou'],
                    'iou_cls':  img_results['iou_per_class'],
                    'path':     paths[i],
                })

    global_results = global_metric.compute()
    return global_results, records

# ─── Plot: IoU Bar Chart ───────────────────────────────────────────────────────

def plot_iou_barchart(results, out_dir):
    iou = results['iou_per_class']
    miou = results['miou']
    colors = [tuple(c/255 for c in col) for col in CLASS_COLORS]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(CLASS_NAMES, iou, color=colors, edgecolor='black', linewidth=0.7)

    for bar, v in zip(bars, iou):
        ax.text(min(v + 0.01, 0.97), bar.get_y() + bar.get_height()/2,
                f'{v:.3f}', va='center', fontsize=10)

    ax.axvline(miou, color='red', linestyle='--', linewidth=1.5,
               label=f'mIoU = {miou:.4f}')
    ax.set_xlim(0, 1.0)
    ax.set_xlabel('IoU Score', fontsize=12)
    ax.set_title('Per-Class IoU — Desert Segmentation', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    path = out_dir / 'iou_per_class.png'
    plt.savefig(path, dpi=150)
    plt.close()
    return path

# ─── Plot: Confusion Matrix ────────────────────────────────────────────────────

def plot_confusion_matrix(results, global_metric, out_dir):
    conf = global_metric.confusion.numpy().astype(float)
    conf_norm = conf / (conf.sum(axis=1, keepdims=True) + 1e-8)

    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(conf_norm, annot=True, fmt='.2f',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                cmap='Blues', ax=ax, vmin=0, vmax=1,
                linewidths=0.5, linecolor='gray')
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('Ground Truth', fontsize=12)
    ax.set_title('Normalised Confusion Matrix', fontsize=14, fontweight='bold')
    plt.xticks(rotation=30, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    path = out_dir / 'confusion_matrix.png'
    plt.savefig(path, dpi=150)
    plt.close()
    return path

# ─── Plot: Prediction Overlays ────────────────────────────────────────────────

def plot_prediction_grid(records, out_dir, n=12):
    records_sorted = sorted(records, key=lambda x: -x['miou'])[:n]
    n_rows = (n + 2) // 3
    fig, axes = plt.subplots(n_rows, 3 * 3, figsize=(24, n_rows * 4.5))

    legend = [mpatches.Patch(color=[c/255 for c in col], label=name)
              for name, col in zip(CLASS_NAMES, CLASS_COLORS)]

    for ax_row in axes:
        for ax in ax_row:
            ax.axis('off')

    for idx, rec in enumerate(records_sorted):
        row = idx // 3
        col = (idx % 3) * 3
        img_np = denormalize(rec['img'])

        axes[row][col + 0].imshow(img_np)
        axes[row][col + 0].set_title(f'Image\nmIoU={rec["miou"]:.3f}', fontsize=8)

        axes[row][col + 1].imshow(mask_to_color(rec['gt']))
        axes[row][col + 1].set_title('Ground Truth', fontsize=8)

        axes[row][col + 2].imshow(overlay(img_np, rec['pred']))
        axes[row][col + 2].set_title('Prediction', fontsize=8)

    fig.legend(handles=legend, loc='lower center', ncol=NUM_CLASSES,
               fontsize=9, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle('Desert Segmentation — Prediction Overlays (Best Samples)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = out_dir / 'prediction_grid.png'
    plt.savefig(path, dpi=120, bbox_inches='tight')
    plt.close()
    return path

# ─── Plot: Failure Cases ──────────────────────────────────────────────────────

def plot_failure_cases(records, out_dir, n=6):
    """Show worst-performing samples with error map."""
    failures = sorted(records, key=lambda x: x['miou'])[:n]
    fig, axes = plt.subplots(n, 4, figsize=(20, n * 4))
    if n == 1:
        axes = [axes]

    legend = [mpatches.Patch(color=[c/255 for c in col], label=name)
              for name, col in zip(CLASS_NAMES, CLASS_COLORS)]

    for idx, rec in enumerate(failures):
        img_np = denormalize(rec['img'])
        gt, pred = rec['gt'], rec['pred']
        error_map = (gt != pred).astype(np.uint8) * 255

        axes[idx][0].imshow(img_np)
        axes[idx][0].set_title(f'Input Image\n(mIoU={rec["miou"]:.3f})', fontsize=9)

        axes[idx][1].imshow(mask_to_color(gt))
        axes[idx][1].set_title('Ground Truth', fontsize=9)

        axes[idx][2].imshow(overlay(img_np, pred))
        axes[idx][2].set_title('Prediction', fontsize=9)

        axes[idx][3].imshow(error_map, cmap='Reds')
        axes[idx][3].set_title('Error Map\n(white=wrong)', fontsize=9)

        for ax in axes[idx]:
            ax.axis('off')

        # Per-class analysis for this failure
        print(f'\n  Failure {idx+1}: {Path(rec["path"]).name}')
        print(f'    mIoU: {rec["miou"]:.4f}')
        for cls_name, iou_val in zip(CLASS_NAMES, rec['iou_cls']):
            print(f'    {cls_name:<20}: {iou_val:.4f}')

    fig.legend(handles=legend, loc='lower center', ncol=NUM_CLASSES,
               fontsize=9, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle('Failure Case Analysis — Lowest mIoU Samples',
                 fontsize=14, fontweight='bold', color='darkred')
    plt.tight_layout()
    path = out_dir / 'failure_cases.png'
    plt.savefig(path, dpi=120, bbox_inches='tight')
    plt.close()
    return path

# ─── Plot: IoU Distribution ──────────────────────────────────────────────────

def plot_iou_distribution(records, out_dir):
    mious = [r['miou'] for r in records]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(mious, bins=30, color='steelblue', edgecolor='white', linewidth=0.5)
    ax.axvline(np.mean(mious), color='red', linestyle='--',
               label=f'Mean={np.mean(mious):.3f}')
    ax.axvline(np.median(mious), color='orange', linestyle='--',
               label=f'Median={np.median(mious):.3f}')
    ax.set_xlabel('Per-Image mIoU', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Distribution of Per-Image mIoU Scores', fontsize=13, fontweight='bold')
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    path = out_dir / 'miou_distribution.png'
    plt.savefig(path, dpi=150)
    plt.close()
    return path

# ─── Results Table ────────────────────────────────────────────────────────────

def print_results_table(results):
    print('\n' + '─'*50)
    print('  EVALUATION RESULTS')
    print('─'*50)
    print(f'  mIoU       : {results["miou"]:.4f}')
    print(f'  Pixel Acc  : {results["pixel_acc"]:.4f}')
    print('─'*50)
    print(f'  {"Class":<22} {"IoU":>8}')
    print('─'*50)
    for name, iou in zip(CLASS_NAMES, results['iou_per_class']):
        bar = '█' * int(iou * 25)
        print(f'  {name:<22} {iou:.4f}  {bar}')
    print('─'*50)

# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    out_dir = Path(args.out) / 'evaluation'
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f'\n🏜  Desert Seg — Evaluation on [{args.split}] split')
    print(f'   Checkpoint: {args.ckpt}')
    print(f'   Device    : {device}\n')

    # Load model
    model = DesertSegFormer(variant=args.variant, pretrained=False).to(device)
    ckpt  = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt['model'])
    print(f'  Loaded checkpoint (epoch {ckpt.get("epoch","?")},'
          f' best_miou={ckpt.get("best_miou",0):.4f})\n')

    # Dataset
    from torch.utils.data import DataLoader
    ds     = DesertSegDataset(args.data, args.split,
                               transform=get_val_transforms())
    loader = DataLoader(ds, batch_size=args.batch, shuffle=False,
                        num_workers=2, pin_memory=True)

    # Run
    global_metric = IoUMetric()
    global_results, records = run_inference(model, loader, device)

    # Rebuild global metric for confusion matrix
    for rec in records:
        m = IoUMetric()
        m.update(torch.from_numpy(rec['pred']).unsqueeze(0),
                 torch.from_numpy(rec['gt']).unsqueeze(0))
    # Re-run globally for confusion matrix (clean accumulation)
    global_metric2 = IoUMetric()
    for rec in records:
        global_metric2.update(torch.from_numpy(rec['pred']).unsqueeze(0),
                               torch.from_numpy(rec['gt']).unsqueeze(0))

    print_results_table(global_results)

    # Save metrics JSON
    metrics_path = out_dir / 'metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump({
            'split': args.split,
            'miou':  global_results['miou'],
            'pixel_acc': global_results['pixel_acc'],
            'iou_per_class': dict(zip(CLASS_NAMES, global_results['iou_per_class'])),
        }, f, indent=2)
    print(f'\n  Metrics → {metrics_path}')

    # Plots
    print('\n  Generating visualizations...')
    p1 = plot_iou_barchart(global_results, out_dir)
    p2 = plot_confusion_matrix(global_results, global_metric2, out_dir)
    p3 = plot_prediction_grid(records, out_dir, n=min(args.n_vis, len(records)))
    p4 = plot_failure_cases(records, out_dir, n=min(args.n_fail, len(records)))
    p5 = plot_iou_distribution(records, out_dir)

    print(f'  ✅ IoU bar chart     → {p1}')
    print(f'  ✅ Confusion matrix  → {p2}')
    print(f'  ✅ Prediction grid   → {p3}')
    print(f'  ✅ Failure cases     → {p4}')
    print(f'  ✅ IoU distribution  → {p5}')
    print('\n  🎉 Evaluation complete!\n')


if __name__ == '__main__':
    main()
