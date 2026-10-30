"""Isolated validation/measurement of the prototype; no model or training imports."""
import argparse
from contextlib import contextmanager
from functools import lru_cache
import json
from pathlib import Path
import statistics

import torch
import triton
from triton.runtime.jit import JITFunction
from .a3_sm75_inline_mma import mma_m16n8k8_kernel


@lru_cache(maxsize=1)
def require_target():
    # Check the compiler first, before any CUDA query or allocation.
    if triton.__version__ != "3.6.0":
        raise RuntimeError(f"Requires Triton 3.6.0; found {triton.__version__}")
    if not torch.cuda.is_available():
        raise RuntimeError("Requires Tesla T4, SM75, device 0")
    name = torch.cuda.get_device_name(0)
    capability = torch.cuda.get_device_capability(0)
    if "T4" not in name or capability != (7, 5) or torch.cuda.current_device() != 0:
        raise RuntimeError(f"Requires Tesla T4 SM75 on device 0; found {name}, {capability}")
    return {"device": name, "capability": list(capability), "triton": triton.__version__,
            "torch": torch.__version__, "cuda": torch.version.cuda}


@contextmanager
def fp32_reference_mode():
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        with torch.autocast("cuda", enabled=False):
            yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous


def _launch(a, b, out, transposed):
    require_target()
    m, k = a.shape
    n = out.shape[1]
    mma_m16n8k8_kernel[(triton.cdiv(m, 16), triton.cdiv(n, 8))](
        a, b, out, m, n, k, transposed, num_warps=1, num_stages=1)


def mma_matmul(a, b, transposed, out_dtype=torch.float16):
    require_target()
    if a.ndim != 2 or b.ndim != 2 or min(*a.shape, *b.shape) < 1:
        raise ValueError("Expected nonempty matrices")
    if any(t.device != torch.device("cuda:0") or t.dtype != torch.float16
           or not t.is_contiguous() for t in (a, b)):
        raise ValueError("Expected contiguous FP16 tensors on cuda:0")
    k = b.shape[1] if transposed else b.shape[0]
    if a.shape[1] != k or out_dtype not in (torch.float16, torch.float32):
        raise ValueError("Incompatible dimensions or output dtype")
    n = b.shape[0] if transposed else b.shape[1]
    out = torch.empty((a.shape[0], n), device=a.device, dtype=out_dtype)
    _launch(a, b, out, transposed)
    return out


class _Projection(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight):
        ctx.save_for_backward(weight)
        return mma_matmul(x, weight, True)

    @staticmethod
    def backward(ctx, dy):
        (weight,) = ctx.saved_tensors
        return mma_matmul(dy.contiguous(), weight, False), None


def experimental_projection(x, weight):
    """Frozen head probe with differentiable casts outside the custom Function."""
    require_target()
    if weight.requires_grad or x.ndim < 2 or weight.ndim != 2:
        raise ValueError("Expected a frozen matrix and hidden states")
    if x.shape[-1] != weight.shape[1]:
        raise ValueError("Hidden width differs from weight")
    amp = torch.is_autocast_enabled("cuda")
    dtype = torch.get_autocast_dtype("cuda") if amp else x.dtype
    if dtype != torch.float16:
        raise ValueError("This probe supports FP16 compute only")
    if not amp and weight.dtype != dtype:
        raise ValueError("Mismatched storage dtype without autocast")
    flat = x.to(dtype).reshape(-1, x.shape[-1]).contiguous()
    y = _Projection.apply(flat, weight.to(dtype).contiguous())
    return y.reshape(*x.shape[:-1], weight.shape[0])


@contextmanager
def watch_mma_launches():
    """Observe a correctness pass only; never wrap measured iterations."""
    records = []
    original = JITFunction.run
    def watched(jit, *args, **kwargs):
        compiled = original(jit, *args, **kwargs)
        if jit is mma_m16n8k8_kernel and not kwargs.get("warmup", False):
            ptx = compiled.asm["ptx"]
            assert ".target sm_75" in ptx
            assert "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32" in ptx
            records.append({"transposed_storage": bool(args[6]),
                            "source": jit.fn.__code__.co_filename,
                            "mma_sync_count": ptx.count("mma.sync.aligned.m16n8k8")})
        return compiled
    JITFunction.run = watched
    try:
        yield records
    finally:
        JITFunction.run = original


def assert_fp16_close(actual, reference):
    assert torch.isfinite(actual).all(), "Nonfinite FP16 result"
    # Includes one final FP16 rounding; references use already-quantized inputs.
    torch.testing.assert_close(actual.float(), reference, rtol=3e-3, atol=5e-4)


def measure_pair(x, w, dy, implementation, iterations):
    if implementation == "Triton_MMA":
        forward = lambda: mma_matmul(x, w, True)
        backward = lambda: mma_matmul(dy, w, False)
    else:
        forward = lambda: torch.mm(x, w.t())
        backward = lambda: torch.mm(dy, w)
    y, dx = forward(), backward()  # Compilation and one warmup outside measurement.
    torch.cuda.synchronize()
    del y, dx
    torch.cuda.empty_cache()
    base_alloc = torch.cuda.memory_allocated()
    base_reserved = torch.cuda.memory_reserved()
    torch.cuda.reset_peak_memory_stats()
    fwd_ms, bwd_ms, pair_ms = [], [], []
    for _ in range(iterations):
        start, middle, end = (torch.cuda.Event(enable_timing=True) for _ in range(3))
        start.record()
        y = forward()
        middle.record()
        dx = backward()
        end.record()
        end.synchronize()  # Surface errors before reading times or releasing outputs.
        fwd_ms.append(start.elapsed_time(middle))
        bwd_ms.append(middle.elapsed_time(end))
        pair_ms.append(start.elapsed_time(end))
        del y, dx
    return {"forward_ms": fwd_ms, "dhidden_ms": bwd_ms, "pair_ms": pair_ms,
            "median_pair_ms": statistics.median(pair_ms),
            "baseline_allocated_MiB": base_alloc / 2**20,
            "baseline_reserved_MiB": base_reserved / 2**20,
            "peak_allocated_MiB": torch.cuda.max_memory_allocated() / 2**20,
            "extra_peak_allocated_MiB": (torch.cuda.max_memory_allocated()-base_alloc) / 2**20,
            "peak_reserved_MiB": torch.cuda.max_memory_reserved() / 2**20}


def run_large(case, rows, iterations):
    environment = require_target()
    features, vocab = (1536, 8192) if case == "medium" else (1536, 151936)
    torch.manual_seed(2026)
    x = torch.randn(rows, features, device="cuda:0", dtype=torch.float16) * 0.125
    w = torch.randn(vocab, features, device="cuda:0", dtype=torch.float16) * 0.125
    dy = torch.randn(rows, vocab, device="cuda:0", dtype=torch.float16) * 0.01
    errors = {}
    with torch.no_grad(), fp32_reference_mode(), watch_mma_launches() as launches:
        for direction, a, b, transposed in (("forward", x, w, True), ("dhidden", dy, w, False)):
            reference = a.float() @ (b.float().t() if transposed else b.float())
            result = mma_matmul(a, b, transposed)
            assert_fp16_close(result, reference)
            errors[direction] = {"max_abs_error": (result.float()-reference).abs().max().item(),
                                 "rms_error": (result.float()-reference).square().mean().sqrt().item()}
            del result, reference
    assert [r["transposed_storage"] for r in launches] == [True, False]
    torch.cuda.synchronize()
    results = {}
    with torch.no_grad(), torch.autocast("cuda", enabled=False):
        for implementation in ("PyTorch_FP16", "Triton_MMA"):
            results[implementation] = measure_pair(x, w, dy, implementation, iterations)
    return {"environment": environment, "shape_M_H_V": [rows, features, vocab],
            "seed": 2026, "iterations": iterations, "correctness": errors,
            "validation_launches": launches, "measurements": results,
            "scope": "Resident FP16 head forward + dHidden GEMMs only; output allocation included. "
                     "No AMP storage casts, decoder, CE, optimizer, training or whole-task memory. "
                     "FP32 reference tensors released before measurement. Allocated and reserved memory differ. "
                     "This minimal one-warp tile is not a production performance design."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=["medium", "qwen"], default="medium")
    parser.add_argument("--rows", type=int, default=32)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.rows <= 512 or not 1 <= args.iterations <= 10:
        parser.error("rows must be 1..512 and iterations 1..10")
    if args.out.exists():
        raise FileExistsError(f"Refusing to overwrite {args.out}")
    result = run_large(args.case, args.rows, args.iterations)
    rendered = json.dumps(result, indent=2)
    args.out.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
