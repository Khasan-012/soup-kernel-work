# A4 implementation and verification plan

Scope: only solution/ changes. Original A2 commit a231371 remains intact.
Work in a copied repository on a4-triton-optimization; preserve every existing
tracked/untracked A3 edit in both source and snapshot. Existing __init__.py
and chunked_lm_head.py backed up outside solution in a4_backups/.

- [x] Inspect full chunked loss, frozen head, SwiGLU, A2/A3 notes and diagnostics.
- [x] Trace installed bitsandbytes 0.50.2 backward and nested NF4 representation.
- [x] Write failing correctness tests before adding each kernel.
- [x] Implement independent fused nested NF4 decode/cast; retain native GEMMs.
- [x] Implement strided hidden packing/cast and frozen head cast with AMP backward.
- [x] Wire through apply; keep A2 CE and example SwiGLU (example excluded).
- [x] Check exact native outputs/gradients including actual tiny Qwen NF4+LoRA graph.
- [x] Run all 79 unchanged Soup regression tests with apply enabled.
- [x] Profile isolated head copy origins and compare isolated operation timings.
- [ ] Validate exact compiler/runtime and all kernels on Tesla T4, torch 2.11, Triton 3.6.
- [ ] Run unchanged full harness and compare time/memory/eval/share against A2.
- [ ] Establish >=5% measured own-kernel share without extra work.

Design choice: LoRA is on q/v only, rank 16, and its native dropout must retain
its RNG stream. Rewriting its GEMMs risks repeating A3's SM75 tensor-core
problem. RMSNorm/residual/activation transforms have smaller traffic than the
full vocabulary weight. NF4 decoding is required for each frozen base layer's
input gradient. It is a bandwidth operation with no tensor-core dependency;
Triton can fuse nested scale decode, offset addition, nibble decode, native
rounding and final compute cast. No weight cache is kept, so no stale parameters
or long-lived dequantized matrices alter memory usage.

head_cast_kernel processes the existing fp32 frozen head weight conversion
and strided chunk packing. Its autograd inverse converts the already fp16-rounded
input gradient back to fp32. cuBLAS performs the original two head GEMMs.
It is a real dtype/layout operation; do not launch it for an identity conversion.

A3 experiment files are preserved. apply routes head projection to A4 rather
than A3's tl.dot kernels. A3 launch time is excluded from A4's accepted kernel set.

Local evidence (2026-10-30): RTX 5070 sm120, torch 2.10.0+cu128,
triton-windows 3.9.0, bitsandbytes 0.50.2. These are NOT competition conditions.

74 tests for head casts, NF4, A2 CE/chunking and example pass. One additional
actual tiny Qwen training-graph test passes: nonzero LoRA B, native dropout
0.05 with matched RNG, exact loss and all adapter gradients, real CE/NF4/cast
launches, no A3 head launches. 79/79 original regression tests pass with apply.

First strict NF4 tests exposed rare 1-ulp fp16 differences on Blackwell when
Triton selected packed f32x2 arithmetic. Scalar inline PTX mul.rn.f32 and
add.rn.f32 preserve the native materialized fp32 scale-rounding boundaries.
All 24 dequant cases now compare bit-for-bit, including odd element counts,
nested/non-nested states and fp16/fp32/bf16 storage. This still needs T4 validation.

Isolated local medians (7 samples after 3 warmups), not timed-loop evidence:

| Operation | Native/A2 ms | A4 ms |
| --- | ---: | ---: |
| NF4 decode 1536x1536 | 0.08141 | 0.05494 |
| NF4 decode 8960x1536 | 0.09978 | 0.07341 |
| Head forward+backward 512x1536x151936 | 10.04387 | 10.09155 |

Native head profiling resolves aten::copy_ [151936,1536] to the large
float16_copy kernel (~2.565 ms in this isolated profiled pass). Chunk casts
[4,128,1536] are much smaller. The profiler also sees direct_copy of an expanded
[512,151936] gradient: in this microbenchmark it comes from y.sum().backward(),
which creates an expanded upstream gradient. That particular copy is a diagnostic
artifact, NOT evidence of a matching copy in the real CE training loop. The native
NF4 backward's explicit .to(grad_output.dtype) can also copy when dtypes differ.
No claim assigns all copies in the user's earlier whole-loop profile to these sites;
that raw trace was not supplied. Do not globally replace Tensor.to/copy_ or remove
copies required by cuBLAS layouts.

| Competition metric | A2 (user supplied) | A4 T4 |
| --- | ---: | --- |
| Training time | 70.04 s | unmeasured |
| Peak memory | 8528 MiB | unmeasured |
| Eval loss | 2.04981 | unmeasured |
| Own kernel time | not supplied | unmeasured |
| Own timed-loop share | 0.83% | unmeasured |

A2 acceptance limits requested: <=77.044 s, <=9380.8 MiB, <=2.06005905 eval loss.
The 5% share remains unproven; local microbenchmarks cannot prove or disprove
that any safe optimization can meet it. No artificial loops or dummy launches.

On the target, after installing the competition dependencies, run:

    python -m pytest solution/test_nf4_backward.py solution/test_head_cast.py solution/test_a4_training.py solution/test_chunked_lm_head.py solution/test_triton_ce.py solution/test_kernels.py -q
    SOUP_APPLY_SOLUTION=1 python -m pytest tests -q
    python -m solution.a4_diagnostics --train --out runs_a4
    python -m solution.a4_diagnostics --metrics runs_a4/solution_1/metrics.json --out a4_share.json

The train command rejects non-T4 or mismatched versions before loading any model;
it delegates to the unmodified harness. --only solution is an A4 measurement,
not final contest admission. For full baseline/solution repeats and rule checks:

    python harness/run_task.py --repeats 2 --out runs_a4_admission
    python harness/check_rules.py --result runs_a4_admission/result.json

A2-vs-A4 comparison additionally requires running preserved A2 in the same target
session, with identical data/cache policy and a separate output directory.

Final verification: 78 selected A4/A2/example tests and 79 original regression tests pass locally. Offline SM75 compilation tests pass (Triton 3.9, not target 3.6); both new kernels use zero shared memory. The unchanged harness source scanner recognizes both kernels, with no banned imports or unparseable files. Full training command exits before model load because the exact target environment is unavailable.
