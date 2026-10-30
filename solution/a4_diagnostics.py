"""A4 diagnostics: isolated operations or target-checked full competition run.

Microbenchmarks never establish contest kernel share. --train requires T4,
PyTorch 2.11 and Triton 3.6; delegates unchanged harness configuration.
"""
import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path

import torch
import triton

OWN = {"ce_forward_kernel", "ce_backward_kernel", "nf4_dq_cast_kernel", "head_cast_kernel"}


def environment():
    return {"torch": torch.__version__, "triton": triton.__version__,
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "capability": list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None}


def require_target():
    info = environment()
    if (info["capability"] != [7, 5] or "T4" not in (info["device"] or "")
            or not info["torch"].startswith("2.11.") or not info["triton"].startswith("3.6.")):
        raise RuntimeError(f"Full A4 measurement requires T4, torch 2.11, Triton 3.6; found {info}")
    return info


def metrics(path):
    data = json.loads(Path(path).read_text())
    loop = data["timed_loop_s"]
    own = {name: data.get("own_kernel_ms", {}).get(name, 0) for name in sorted(OWN)
           if data.get("triton_launches", {}).get(name, 0) > 0
           and data.get("triton_files", {}).get(name, "").replace("\\", "/").startswith("solution/")}
    total = sum(own.values())
    return {"source": str(path), "own_kernel_ms": own, "own_total_ms": total,
            "timed_loop_s": loop, "own_share_percent": total / (loop * 10) if loop else None,
            "complete_64_steps": data.get("steps") == 64,
            "missing_kernels": sorted(OWN - own.keys()),
            "note": "Example SwiGLU and A3 GEMMs excluded. Use unchanged rule checker for admission."}


def timing(fn):
    for _ in range(3):
        out = fn()
        del out
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    samples = []
    for _ in range(7):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        out = fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
        del out
    return {"median_ms": statistics.median(samples), "samples_ms": samples,
            "extra_peak_allocated_MiB": (torch.cuda.max_memory_allocated() - baseline) / 2**20}


def microbench():
    import bitsandbytes.functional as F
    from .head_cast import project_head
    from .nf4_backward import dequantize_nf4
    torch.manual_seed(2026)
    result = {"scope": "isolated operations, NOT training or contest share", "environment": environment()}
    for rows, cols in [(1536, 1536), (8960, 1536)]:
        source = torch.randn(rows, cols, device="cuda", dtype=torch.float16) * 0.1
        packed, state = F.quantize_4bit(source, quant_type="nf4", compress_statistics=True)
        del source
        result[f"NF4_{rows}x{cols}"] = {
            "native": timing(lambda: F.dequantize_4bit(packed, state)),
            "A4": timing(lambda: dequantize_nf4(packed, state, torch.float16))}
    del packed, state
    head = torch.nn.Linear(1536, 151936, device="cuda", bias=False).requires_grad_(False)
    x = torch.randn(4, 512, 1536, device="cuda")[:, :128].detach().requires_grad_(True)
    def step(project):
        x.grad = None
        with torch.autocast("cuda", dtype=torch.float16):
            y = project(head, x)
        y.sum().backward()
        return x.grad
    result["head_forward_backward"] = {
        "A2_native": timing(lambda: step(lambda h, x: h(x))),
        "A4": timing(lambda: step(project_head))}
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                           torch.profiler.ProfilerActivity.CUDA],
                                record_shapes=True) as prof:
        step(lambda h, x: h(x))
        torch.cuda.synchronize()
    result["isolated_head_A2_operators"] = sorted([
        {"name": event.key, "shapes": event.input_shapes, "count": event.count,
         "self_gpu_us": event.self_device_time_total}
        for event in prof.key_averages(group_by_input_shape=True)
        if event.self_device_time_total > 0], key=lambda row: -row["self_gpu_us"])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--microbench", action="store_true")
    group.add_argument("--train", action="store_true")
    group.add_argument("--metrics", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    if args.train:
        require_target()
        subprocess.run([sys.executable, "harness/run_task.py", "--only", "solution",
                        "--profile", "--out", str(args.out)], check=True)
        return
    value = microbench() if args.microbench else metrics(args.metrics)
    args.out.write_text(json.dumps(value, indent=2) + "\n")
    print(json.dumps(value, indent=2))


if __name__ == "__main__":
    main()
