# A2: Triton cross-entropy for the A1 chunked LM head

## Scope

This continues the existing chunked head and its Qwen2/TRL patches. Model,
dataset, batching, number/order of steps, LoRA, optimizer and schedule are
unchanged. The example SwiGLU remains in place. No stage B or C work is included.

## Kernels and math

`ce_forward_kernel` streams each row in vocabulary tiles. Its online state is
the row maximum `m`, `s = sum(exp(x - m))` and
`w = sum(exp(x - m) * (x - m))`. When the maximum changes, both sums are
rescaled into the new coordinate system. It returns:

- token loss `(m - x[target]) + log(s)` for supervised tokens, otherwise zero;
- entropy `log(s) - w / s` for every position, including ignored labels;
- correct predictions using the lowest vocabulary index to break ties.

`ce_backward_kernel` runs independent row/vocabulary tiles and writes
`(exp((x - m) - log(s)) - one_hot(target)) * incoming_scale`, with exactly zero
gradients for ignored labels. Separate `m` and `log(s)` prevent cancellation
under large common logit shifts. Reduction statistics are fp32; gradients are
stored in the original logits dtype, as with the fp32 PyTorch CE upcast.

The autograd function returns the summed CE. A1 still divides by the trainer's
`num_items_in_batch`, or the supervised-token count if none was supplied, before
taking hidden-state gradients. The external upstream loss gradient is preserved.
All-ignored batches keep A1's zero-loss convention when no denominator is passed.

The existing `ChunkedStep` and `_soup_chunked_step` contracts remain intact.
Soup 0.77.0 Rewind consumes its token losses; TRL 0.29.1 metrics consume entropy,
correct predictions and labelled-token counts.

## Fallback

The PyTorch A1 path remains available for CPU, noncontiguous or unsupported
inputs, and when `solution.chunked_lm_head._TRITON_ENABLED = False` is set before
the forward. Dispatch rejects empty rows and targets on a different device.
Compilation and launch failures are reported, never silently hidden by fallback.

## Verification on Kaggle T4

Local correctness run (2026-10-30, RTX 5070, Windows; PyTorch 2.10.0+cu128,
triton-windows 3.9.0.post30; Soup/HF versions pinned as in the competition):

- `python -m pytest solution -q`: **44 passed**; the CUDA cases ran rather than skipped.
- `SOUP_APPLY_SOLUTION=1 python -m pytest tests -q`: **79 passed**.
- The kernel source scanner recognizes both CE kernels as own kernels and reports
  no forbidden imports or unparseable files. This is source inspection, not a
  full competition rule-check result.
- Triton 3.9 emits its default FP fusion warning; no test failed. Compilation
  uses the existing default, with no new environment requirement for the solution.

Run inside the repository with the competition's pinned environment:

```bash
python -m pytest solution -q
SOUP_APPLY_SOLUTION=1 python -m pytest tests -q
```

Optional mathematical check without GPU compilation:

```bash
TRITON_INTERPRET=1 python -m pytest solution/test_triton_ce.py -q
```

Only after reviewing correctness, measure the complete task:

```bash
python harness/run_task.py --only solution --profile --out runs_a2_profile
python harness/run_task.py --repeats 2 --out runs_final
python harness/check_rules.py --result runs_final/result.json
```

The profile-only solution run lacks baseline runs; it cannot pass the full
rule checker by itself. Final admission requires both base and solution runs.
Review `own_kernel_ms` and `timed_loop_s` in each solution run's `metrics.json`:
only the two CE kernels count here. The copied example SwiGLU kernels explicitly
do not count toward the required 5%, so the earlier 5.16% example timing is not
evidence that this condition passes. Do not add artificial work to increase it.

No A2 training time, VRAM reduction, eval loss or kernel-share result is claimed
until measured on Kaggle T4. Local RTX results establish correctness only.
