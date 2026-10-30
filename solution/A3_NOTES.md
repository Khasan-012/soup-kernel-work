# A3: useful Triton work in the frozen vocabulary head

## Control and scope

A2 control: `main` at `a231371d30bd0d2dd2d9073d9ea8d50991a7e671`.
Development branch: `a3-triton-work`, based on that commit. No A3 commit or push
has been made. Only `solution/` changed. No full model training was run locally.

User-supplied A2 Kaggle T4 results: 80.0 s whole command, 8525 MiB peak,
eval loss 2.04984, 44 solution tests and 79 main tests passed. Own CE time:
129.308 + 235.712 = 365.020 ms, timed loop 43.615 s, own share 0.8369%.
The example SwiGLU kernels do not count, even if copied or renamed.

## Audit and selection

No full `profile.json` or `metrics.json` was available in the repository. Thus
the broad GEMM/copy/softmax/NF4/attention categories cannot honestly be assigned
layer-specific percentages. Source inspection identifies the frozen LM-head
operations unambiguously: `hidden @ weight.T` and `dlogits @ weight` per chunk.

The actual Qwen head has hidden width 1536 and vocabulary 151936. At the maximum
2048-token batch each of those two GEMMs is about 0.956 TFLOP. These are ordinary
floating-point GEMMs, separate from quantized decoder/LoRA projections.

An isolated A2 head profile on RTX 5070 attributed ~6.93 ms to its two `aten::mm`
calls and ~2.30 ms to three copies (two fp16 casts and one small direct copy).
This identifies these copies only; it does not explain every training copy.
PEFT kbit preparation upcasts the head storage to fp32, while Soup's T4 AMP
selects fp16 compute, so preserving the cast and gradient-rounding boundaries
is essential. A3 casts each chunk's head weight once and reuses it in backward.

Other candidates were deferred:

| Work | Decision for this increment |
|---|---|
| Large frozen-head GEMMs | Replace forward and input gradient; substantial known useful work |
| General tensor copies | Do not replace without knowing producer, dtype and lifetime |
| CE and its reductions | Keep the tested A2 kernels; 0.84% alone is insufficient |
| Attention/softmax | Keep the existing causal/GQA/mask/backward behavior; no safe full attribution available |
| NF4 dequantization | Keep bitsandbytes format/quantization semantics; no new packed-weight decoder |
| RMSNorm/RoPE | Smaller targets; do not combine unrelated changes before measuring this candidate |
| Example SwiGLU | Unchanged and excluded from all own-share calculations |

## Architecture

`frozen_lm_head.py` implements:

1. `lm_head_forward_kernel`: tiled tensor-core vocabulary projection, fp32
   accumulation, output in the current AMP/native fp16 or bf16 compute dtype.
2. `lm_head_dhidden_partial_kernel`: the existing input-gradient GEMM split
   into eight disjoint vocabulary ranges for the task head (one for small test
   vocabularies), producing fp32 partial hidden gradients.
3. `lm_head_dhidden_reduce_kernel`: fixed-order sum of the disjoint partials
   and one cast to the compute dtype. No atomics, repeated GEMMs or added dummy work.

The conversion from original hidden dtype to compute dtype is outside the
custom autograd Function. Thus its input gradient is rounded to fp16/bf16 before
PyTorch converts it back to the original dtype, matching the original AMP Linear.
The head remains frozen; no weight gradient is computed or discarded. Tensor
layout tails and strided chunks are handled without changing batch/data order.

Row counts are runtime scalars, avoiding a new full compilation for every
last-chunk sequence length. A local check across 512, 496, 384 and 128 rows
reused one compiled variant per kernel (Triton may still specialize alignment).

Only `_chunk_terms`' head call changes. A2 CE, `ignore_index`, supervised-token
denominator, external loss scaling, entropy, accuracy, token losses and the
`_soup_chunked_step` Rewind contract are unchanged.

The custom path accepts a plain frozen, bias-free CUDA Linear with supported
storage/compute dtypes and no local module hooks or forward override. CPU,
fp32 without AMP, hooks, subclasses and other unsupported heads retain PyTorch.
Unexpected half/AMP fallbacks emit `A3 frozen head fallback: ...` once per
reason, so a task fallback is visible in `train.log`. Compilation/launch errors
on supported inputs are never swallowed. Explicitly setting
`solution.frozen_lm_head.ENABLED = False` before the forward selects A2 for a
same-process diagnostic comparison; default execution always enables A3.

## Evidence and limits

Local CUDA hardware: RTX 5070, PyTorch 2.10.0+cu128, Windows Triton 3.9.0.post30;
Soup 0.77.0, Transformers 5.16.1, TRL 0.29.1, PEFT 0.20.0.

Final local suites: **71 solution tests passed**, no skips; **79 main tests
passed** with `SOUP_APPLY_SOLUTION=1`. Warnings were Triton 3.9's default FP
fusion notice and the intentionally exercised unsupported-head fallback paths.
On T4 the two native-bf16 projection cases are expected to skip; the real task
uses fp16 AMP and must execute all three head kernels.

- Small forward/backward fixtures cover native fp16, bf16 where supported,
  fp32 storage under fp16 AMP, nonmultiple dimensions, strided inputs, masked
  and all-ignored rows, external gradient scaling, denominator, real Qwen/LoRA,
  real A2 CE and the installed Soup Rewind row-loss reader.
- Tests require actual launches of all three A3 kernels on the supported path.
- The source rule scanner recognizes all five CE/head kernels as own, reports
  no banned imports or parse failures, and separately excludes example SwiGLU.
- Offline code generation for `sm75` succeeds for all three new kernels;
  forward/partial backward use 16 KiB shared memory each. This does not prove
  runtime behavior or speed with Kaggle's installed Triton.
- Full-size head-only comparison `[512,1536] @ [151936,1536].T`, with backward:
  logits and hidden gradients match PyTorch within explicit fp16 tolerances.
  Five-iteration median: A2 9.304 ms, A3 9.632 ms (+3.5%) on RTX 5070.
  These numbers include input/weight conversion. They are not T4 training times.
- Peak allocated memory in that isolated comparison increases by 24 MiB,
  exactly the eight fp32 partial hidden-gradient buffers at maximum chunk size.
  Full-task nvidia-smi peak/reserved memory remains unmeasured.

The own-share target is plausible because the head GEMMs now count, but it is
not verified. At the old 43.615 s loop, 5% means 2180.75 ms total; A3 needs at
least 1815.73 ms of useful head-kernel time in addition to the old CE total.
A purely illustrative calculation using ~7.21 ms of local own-head kernels,
256 full-sized chunks and the old CE/loop totals gives about 5.1%. Actual
sequence lengths, T4 throughput and the new loop duration invalidate treating
that calculation as a forecast or admission result. Shorter batches may miss
5%. No artificial work should be added to compensate.

Main risks: T4 may prefer cuBLAS more strongly; compilation/AMP behavior can
differ across Triton versions; split reduction changes floating-point ordering;
the whole-task allocator peak can differ from the isolated +24 MiB. The
required <=88 s command time, ~8.5 GB peak, eval-quality limit and >=5% own share
must all be checked on Kaggle. A3 is a candidate, not a validated replacement
for the measured A2 control.

## Kaggle readiness and next commands

Inside the A3 repository, using the competition's pinned environment:

```bash
python -m pytest solution -q
SOUP_APPLY_SOLUTION=1 python -m pytest tests -q
python -m solution.a3_diagnostics --benchmark --out /kaggle/working/a3_head.json
```

The third command is bounded: two warmups and five measured head-only forward/
backward iterations per implementation. It loads no dataset/model and trains
no adapter. Stop if correctness fails or expected head kernels do not launch.
`--profile` optionally profiles one additional isolated A2 head pass.

After this readiness report, obtain authorization before full training:

```bash
python harness/run_task.py --only solution --profile --out runs_a3_try
python -m solution.a3_diagnostics --metrics runs_a3_try/solution_1/metrics.json
```

Check `train.log` for fallback warnings and verify that the recorded Triton
files are under `solution/`. The share reader excludes example kernels and
requires both launch and file evidence. A solution-only run lacks a measured
baseline and cannot establish all final rule checks. When ready for final
baseline/solution repeats:

```bash
python harness/run_task.py --repeats 2 --out runs_a3_final
python harness/check_rules.py --result runs_a3_final/result.json
```

Use the lowest own-kernel share among solution runs. Compare time, nvidia-smi
peak, curve/rewind metrics and held-out eval loss, and retain A2 if any required
check or the user's stricter 88-second budget fails.
