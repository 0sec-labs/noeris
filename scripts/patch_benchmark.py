#!/usr/bin/env python3
"""Benchmark noeris.patch() latency on a HuggingFace model.

Usage:
    python scripts/patch_benchmark.py
    python scripts/patch_benchmark.py --model meta-llama/Llama-3.2-1B
    python scripts/patch_benchmark.py --model google/gemma-4-2b --seq-len 512 --batch 2

Measures median forward-pass latency before and after ``noeris.patch()`` and
reports the speedup ratio.
"""

from __future__ import annotations

import argparse
import time


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark noeris.patch() on a HuggingFace model")
    parser.add_argument("--model", type=str, default="google/gemma-4-2b", help="Model name or local path")
    parser.add_argument("--batch", type=int, default=1, help="Batch size")
    parser.add_argument("--seq-len", type=int, default=256, help="Sequence length")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup iterations")
    parser.add_argument("--repeats", type=int, default=20, help="Timed iterations")
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float16", "bfloat16", "float32"],
        help="Model dtype",
    )
    parser.add_argument("--device", type=str, default="cuda", help="torch device (for example: cuda, cuda:0, cpu)")
    parser.add_argument(
        "--kernels",
        type=str,
        default="rmsnorm,geglu",
        help="Comma-separated noeris patch kernels (supported: rmsnorm,geglu)",
    )
    return parser.parse_args()


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    middle = len(ordered) // 2
    return ordered[middle]


def benchmark_forward(model, input_ids, warmup: int, repeats: int, device: str) -> float:
    """Return median forward-pass time in milliseconds."""
    import torch

    use_cuda_sync = device.startswith("cuda") and torch.cuda.is_available()

    with torch.no_grad():
        for _ in range(warmup):
            _ = model(input_ids)
    if use_cuda_sync:
        torch.cuda.synchronize()

    times_ms: list[float] = []
    with torch.no_grad():
        for _ in range(repeats):
            if use_cuda_sync:
                torch.cuda.synchronize()
            start = time.perf_counter()
            _ = model(input_ids)
            if use_cuda_sync:
                torch.cuda.synchronize()
            end = time.perf_counter()
            times_ms.append((end - start) * 1000.0)

    return _median(times_ms)


def _parse_kernels(raw: str) -> list[str]:
    items = [item.strip() for item in raw.split(",") if item.strip()]
    return items or ["rmsnorm", "geglu"]


def main() -> None:
    args = parse_args()

    import torch
    from transformers import AutoModelForCausalLM

    import noeris

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    dtype = dtype_map[args.dtype]
    kernels = _parse_kernels(args.kernels)

    print(f"Model: {args.model}")
    print(f"Batch: {args.batch}, Seq: {args.seq_len}, Dtype: {args.dtype}")
    print(f"Device: {args.device}")
    print(f"Patch kernels: {kernels}")
    print()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but torch.cuda.is_available() is False")

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype)
    model = model.to(args.device)
    model.eval()

    vocab_size = int(model.config.vocab_size)
    input_ids = torch.randint(0, vocab_size, (args.batch, args.seq_len), device=args.device)

    print("Benchmarking baseline (unpatched)...")
    baseline_ms = benchmark_forward(model, input_ids, args.warmup, args.repeats, args.device)
    print(f"  Baseline: {baseline_ms:.3f} ms")

    print("\nApplying noeris.patch()...")
    noeris.patch(model, kernels=kernels, verbose=True)

    print("\nBenchmarking patched model...")
    patched_ms = benchmark_forward(model, input_ids, args.warmup, args.repeats, args.device)
    print(f"  Patched: {patched_ms:.3f} ms")

    speedup = baseline_ms / patched_ms if patched_ms > 0 else float("inf")
    delta_pct = ((baseline_ms - patched_ms) / baseline_ms) * 100 if baseline_ms > 0 else 0.0

    print(f"\n{'=' * 56}")
    print(f"  Baseline median latency: {baseline_ms:.3f} ms")
    print(f"  Patched  median latency: {patched_ms:.3f} ms")
    print(f"  Speedup               : {speedup:.3f}x")
    print(f"  Delta                 : {delta_pct:+.2f}%")
    print(f"{'=' * 56}")

    if speedup < 1.0:
        print("\nWARNING: patched model is slower in this setup.")
        print("- Try larger shapes or batch size to amortize overhead")
        print("- Prefer GPU runs for Triton-backed kernels")
        print("- Verify dtype and model architecture are patch-supported")


if __name__ == "__main__":
    main()
