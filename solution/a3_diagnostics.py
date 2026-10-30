"""Bounded head-only diagnostics; never loads/trains the model or changes its task.

python -m solution.a3_diagnostics --benchmark --out /kaggle/working/a3_head.json
python -m solution.a3_diagnostics --metrics runs_try/solution_1/metrics.json
"""

import argparse
import json
import statistics
from pathlib import Path


OWN_NAMES = {
    "ce_forward_kernel", "ce_backward_kernel", "lm_head_forward_kernel",
    "lm_head_dhidden_partial_kernel", "lm_head_dhidden_reduce_kernel",
}


def summarize_training_metrics(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    files = data.get("triton_files", {})
    launches = data.get("triton_launches", {})
    timings = data.get("own_kernel_ms", {})
    accepted = {
        name: float(timings.get(name, 0.0)) for name in OWN_NAMES
        if launches.get(name, 0) > 0
        and str(files.get(name, "")).replace("\\", "/").startswith("solution/")
    }
    loop = float(data.get("timed_loop_s") or 0.0)
    total = sum(accepted.values())
    return {
        "source": str(path), "own_kernel_ms_excluding_example": accepted,
        "own_ms": total, "timed_loop_s": loop,
        "own_share_percent": 100 * total / (1000 * loop) if loop > 0 else None,
        "missing_own_launches": sorted(OWN_NAMES - accepted.keys()),
        "note": "For final admission use harness/check_rules.py and the lowest share of all runs.",
    }


def benchmark_head(rows=512, features=1536, vocab=151936, iterations=5, profile=False):
    import torch
    from triton.runtime.jit import JITFunction
    from .frozen_lm_head import project_frozen_head, projection_support

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")
    torch.manual_seed(2026)
    head = torch.nn.Linear(features, vocab, bias=False, device="cuda").requires_grad_(False)
    hidden = (torch.randn(rows, features, device="cuda") * 0.1).requires_grad_(True)
    upstream = torch.randn(rows, vocab, device="cuda", dtype=torch.float16) * 0.001
    with torch.autocast("cuda", dtype=torch.float16):
        ready, reason = projection_support(head, hidden)
    if not ready:
        raise RuntimeError(f"Benchmark unexpectedly falls back: {reason}")

    def step(project):
        hidden.grad = None
        with torch.autocast("cuda", dtype=torch.float16):
            logits = project(head, hidden)
        logits.backward(upstream)
        return logits

    results = {
        "kind": "isolated frozen-head microbenchmark, NOT timed training loop",
        "device": torch.cuda.get_device_name(0), "torch": torch.__version__,
        "shape": {"rows": rows, "hidden": features, "vocab": vocab},
        "iterations": iterations,
    }
    reference_logits, reference_grad = None, None
    for name, project in (("A2_PyTorch", lambda module, x: module(x)), ("A3_Triton", project_frozen_head)):
        for _ in range(2):
            step(project)
        torch.cuda.synchronize()
        hidden.grad = None
        torch.cuda.reset_peak_memory_stats()
        times = []
        for _ in range(iterations):
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            begin.record()
            output = step(project)
            end.record()
            end.synchronize()
            times.append(begin.elapsed_time(end))
            del output
        peak = torch.cuda.max_memory_allocated() / 2**20
        output = step(project)
        if name == "A2_PyTorch":
            reference_logits = output.detach().clone()
            reference_grad = hidden.grad.detach().clone()
        else:
            # Full contest vocabulary and hidden width, beyond the small unit fixtures.
            torch.testing.assert_close(output, reference_logits, rtol=0.003, atol=5e-4)
            torch.testing.assert_close(hidden.grad, reference_grad, rtol=0.01, atol=1e-4)
        del output
        results[name] = {"median_ms": statistics.median(times), "samples_ms": times, "peak_allocated_MiB": peak}
    # The reference copies were allocated after the A2 peak; subtract their
    # bytes from A3's absolute peak for an equal live-input comparison.
    comparison_bytes = reference_logits.nbytes + reference_grad.nbytes
    results["A3_Triton"]["peak_allocated_MiB"] -= comparison_bytes / 2**20

    counts, files, events = {}, {}, []
    original_run = JITFunction.run

    def watched(jit, *args, **kwargs):
        fn = jit.fn
        name = fn.__name__
        counts[name] = counts.get(name, 0) + 1
        files[name] = str(Path(fn.__code__.co_filename).resolve())
        if name not in OWN_NAMES or kwargs.get("warmup"):
            return original_run(jit, *args, **kwargs)
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        value = original_run(jit, *args, **kwargs)
        end.record()
        events.append((name, start, end))
        return value

    JITFunction.run = watched
    try:
        step(project_frozen_head)
        torch.cuda.synchronize()
    finally:
        JITFunction.run = original_run
    results["head_only_triton_launches"] = counts
    results["head_only_triton_files"] = files
    results["head_only_kernel_ms"] = {name: start.elapsed_time(end) for name, start, end in events}
    if profile:
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as prof:
            step(lambda module, x: module(x))
            torch.cuda.synchronize()
        results["isolated_A2_aten_profile"] = sorted(
            [{"name": event.key, "self_device_time_us": event.self_device_time_total, "count": event.count}
             for event in prof.key_averages() if event.self_device_time_total > 0],
            key=lambda row: -row["self_device_time_us"],
        )[:15]
    results["note"] = "No inference of whole-task speed, memory, loss or own-kernel share is valid from this microbenchmark."
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--metrics", type=Path, nargs="+")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--iterations", type=int, default=5)
    args = parser.parse_args()
    if not args.benchmark and not args.metrics:
        parser.error("pass --benchmark or --metrics PATH [PATH ...]")
    if not 1 <= args.iterations <= 20:
        parser.error("iterations must be between 1 and 20")
    output = {}
    if args.metrics:
        output["training_metrics"] = [summarize_training_metrics(path) for path in args.metrics]
    if args.benchmark:
        output["head_benchmark"] = benchmark_head(iterations=args.iterations, profile=args.profile)
    rendered = json.dumps(output, indent=2, ensure_ascii=False)
    print(rendered)
    if args.out:
        args.out.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
