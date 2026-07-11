"""
Desert Segmentation — ONNX Export & Inference Benchmark
Exports trained model to ONNX for optimized deployment.
Benchmarks PyTorch vs ONNX runtime latency.
"""

import sys
import time
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from model import DesertSegFormer

try:
    import onnx
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print('[WARN] onnx / onnxruntime not installed. Skipping ONNX export.')


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt',      type=str, required=True)
    p.add_argument('--out',       type=str, default='outputs')
    p.add_argument('--variant',   type=str, default='b2')
    p.add_argument('--img_size',  type=int, default=512)
    p.add_argument('--n_runs',    type=int, default=50, help='warmup+bench iterations')
    p.add_argument('--batch',     type=int, default=1)
    return p.parse_args()


def benchmark_pytorch(model, dummy_input, device, n_runs=50, warmup=10):
    model.eval()
    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy_input)
    torch.cuda.synchronize() if device.type == 'cuda' else None

    # Benchmark
    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            t0 = time.perf_counter()
            logits = model(dummy_input)
            _ = F.interpolate(logits, size=(dummy_input.shape[-2], dummy_input.shape[-1]),
                              mode='bilinear', align_corners=False).argmax(1)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)

    return np.array(times)


def benchmark_onnx(session, dummy_np, n_runs=50, warmup=10):
    inp_name = session.get_inputs()[0].name
    # Warmup
    for _ in range(warmup):
        _ = session.run(None, {inp_name: dummy_np})

    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        _ = session.run(None, {inp_name: dummy_np})
        times.append((time.perf_counter() - t0) * 1000)
    return np.array(times)


def export_onnx(model, dummy_input, out_path, img_size):
    print(f'\n  Exporting to ONNX → {out_path}')
    torch.onnx.export(
        model,
        dummy_input,
        str(out_path),
        opset_version=13,
        input_names=['pixel_values'],
        output_names=['logits'],
        dynamic_axes={
            'pixel_values': {0: 'batch'},
            'logits':       {0: 'batch'},
        },
        do_constant_folding=True,
        verbose=False,
    )
    # Verify
    onnx_model = onnx.load(str(out_path))
    onnx.checker.check_model(onnx_model)
    size_mb = out_path.stat().st_size / 1e6
    print(f'  ✅ ONNX export OK  |  Size: {size_mb:.1f} MB')
    return out_path


def print_benchmark_table(pt_times, onnx_times=None):
    print('\n' + '─'*55)
    print('  INFERENCE BENCHMARK  (ms per image, batch=1)')
    print('─'*55)
    print(f'  {"Metric":<20} {"PyTorch":>12}', end='')
    if onnx_times is not None:
        print(f'  {"ONNX":>12}', end='')
    print()
    print('─'*55)

    for label, fn in [('Mean (ms)', np.mean), ('Median (ms)', np.median),
                       ('Std (ms)', np.std), ('Min (ms)', np.min),
                       ('Max (ms)', np.max)]:
        pt_val = fn(pt_times)
        row = f'  {label:<20} {pt_val:>11.2f}'
        if onnx_times is not None:
            row += f'  {fn(onnx_times):>11.2f}'
        print(row)

    print('─'*55)
    if onnx_times is not None:
        speedup = np.mean(pt_times) / (np.mean(onnx_times) + 1e-9)
        print(f'  ONNX Speedup: {speedup:.2f}×')
    throughput = 1000 / np.mean(pt_times)
    print(f'  PyTorch Throughput: {throughput:.1f} FPS')
    print('─'*55 + '\n')


def main():
    args   = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f'\n🏜  Desert Seg — ONNX Export & Benchmark')
    print(f'   Checkpoint : {args.ckpt}')
    print(f'   Device     : {device}')
    print(f'   Image size : {args.img_size}×{args.img_size}')

    # Load model
    model = DesertSegFormer(variant=args.variant, pretrained=False).to(device)
    ckpt  = torch.load(args.ckpt, map_location=device)
    if 'model' in ckpt: model.load_state_dict(ckpt['model'])
    else: model.load_state_dict(ckpt)
    model.eval()
    print(f'  Loaded epoch {ckpt.get("epoch","?")} | mIoU={ckpt.get("best_miou",0):.4f}\n')

    dummy_input = torch.randn(args.batch, 3, args.img_size, args.img_size).to(device)

    # ── PyTorch benchmark ─────────────────────────────────────────────────────
    print('  Benchmarking PyTorch...')
    pt_times = benchmark_pytorch(model, dummy_input, device, n_runs=args.n_runs)
    print(f'  PyTorch mean latency: {np.mean(pt_times):.2f} ms')

    # ── ONNX export + benchmark ───────────────────────────────────────────────
    onnx_times = None
    if ONNX_AVAILABLE:
        onnx_path = out_dir / 'desert_seg.onnx'
        model_cpu = model.cpu()
        dummy_cpu = dummy_input.cpu()
        export_onnx(model_cpu, dummy_cpu, onnx_path, args.img_size)

        # ONNX Runtime session
        providers = (['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
                     if device.type == 'cuda' else ['CPUExecutionProvider'])
        
        provider_options = [{'trt_engine_cache_enable': True, 'trt_engine_cache_path': str(out_dir)}] if device.type == 'cuda' else None

        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        if provider_options:
            session = ort.InferenceSession(str(onnx_path), sess_opts, providers=providers, provider_options=provider_options)
        else:
            session = ort.InferenceSession(str(onnx_path), sess_opts, providers=providers)

        dummy_np = dummy_cpu.numpy()
        print('  Benchmarking ONNX Runtime...')
        onnx_times = benchmark_onnx(session, dummy_np, n_runs=args.n_runs)
        print(f'  ONNX mean latency: {np.mean(onnx_times):.2f} ms')

    # ── Summary table ─────────────────────────────────────────────────────────
    print_benchmark_table(pt_times, onnx_times)

    # Save results
    import json
    results = {
        'pytorch_mean_ms': float(np.mean(pt_times)),
        'pytorch_fps':     float(1000 / np.mean(pt_times)),
        'img_size':        args.img_size,
        'device':          str(device),
    }
    if onnx_times is not None:
        results['onnx_mean_ms']  = float(np.mean(onnx_times))
        results['onnx_fps']      = float(1000 / np.mean(onnx_times))
        results['onnx_speedup']  = float(np.mean(pt_times) / np.mean(onnx_times))
        results['onnx_path']     = str(onnx_path)

    with open(out_dir / 'benchmark_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f'  Results → {out_dir}/benchmark_results.json')


if __name__ == '__main__':
    main()
