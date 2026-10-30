# Isolated SM75 MMA validation on Kaggle

These files do not integrate the prototype into training. Use the existing
PyTorch, CUDA and Triton environment; no package installation is required.
The required target is GPU 0: Tesla T4, capability (7,5), Triton exactly 3.6.0.
The guard raises an error before allocations if the target does not match.

Transfer these new files into the Kaggle clone under solution/:
- a3_sm75_validation.py
- test_a3_sm75_inline_mma.py
The existing a3_sm75_inline_mma.py is also required. New files are currently
local changes; cloning the published branch alone does not transfer them.

Run cells individually. Each shell cell must use `set -euo pipefail` so a
failure stops subsequent commands. A failed comparison is a blocking result;
do not relax tolerances or continue to timing after a failure.

```bash
%%bash
set -euo pipefail
cd /kaggle/working/soup-kernel-a3
python -c "from solution.a3_sm75_validation import require_target; print(require_target())"
python -m solution.a3_sm75_inline_mma --codegen
python -m pytest solution/test_a3_sm75_inline_mma.py -q -W error -p no:cacheprovider
```

The expected collection is 60 cases: 40 matrix/tail/storage/output-dtype
comparisons, 8 exact accumulator checks, 8 forward/gradient/AMP cases and
4 output-canary checks for writes outside tail buffers.
FP32 references use the actual FP16-quantized operands promoted to FP32, with
CUDA autocast disabled and TF32 disallowed. FP16 output tolerance is
rtol=0.003, atol=0.0005 (one final FP16 rounding included). Raw FP32 output
uses rtol=0.0002, atol=0.00002. Accumulator sentinels require exact equality.
The latter test retained unit contributions, products exceeding FP16 range,
cancellation and zero inputs. FP32 accumulation does not extend the range of
the final FP16 output; true outputs above 65504 can still overflow when stored
as FP16. Overflow sentinels therefore inspect raw FP32 outputs. Gradient tests include both native FP16 and
FP32 storage under FP16 AMP. Every gradient case checks two real launches,
their forward/backward storage modes, SM75 target and MMA instruction in PTX.
These are prototype head checks; LoRA/Rewind/-100 integration is still pending.

After the full small suite passes, run a bounded medium-size comparison:

```bash
%%bash
set -euo pipefail
cd /kaggle/working/soup-kernel-a3
python -m solution.a3_sm75_validation --case medium --rows 32 --iterations 3 --out /kaggle/working/sm75_medium.json
```

Then test full Qwen head width/vocabulary (H=1536, V=151936), initially with
32 rows. A real chunk may have up to 512 flattened rows. Use a separate cell
for that larger workload after inspecting the 32-row result:

```bash
%%bash
set -euo pipefail
cd /kaggle/working/soup-kernel-a3
python -m solution.a3_sm75_validation --case qwen --rows 32 --iterations 3 --out /kaggle/working/sm75_qwen32.json
```

```bash
%%bash
set -euo pipefail
cd /kaggle/working/soup-kernel-a3
python -m solution.a3_sm75_validation --case qwen --rows 512 --iterations 3 --out /kaggle/working/sm75_qwen512.json
```

The script checks FP32-reference forward/dHidden accuracy before timing.
An error prevents timing and output publication. It records actual validation
launches and their PTX; instrumentation is removed before measured passes.
It warms each implementation once, excludes JIT compilation, synchronizes
CUDA events, and reports every forward, dHidden and combined timing sample.

Both implementations receive identical resident FP16 x/weight/dLogits tensors
and allocate FP16 outputs. FP32 reference tensors are released before memory
measurement. The report includes environment, shape, seed, absolute and RMS
errors, baseline allocation/reservation, peak allocation/reservation and extra
allocated peak. Unused allocator cache is released between implementations;
reserved memory is allocator-dependent. Run in a fresh session without other
live CUDA models/tensors for interpretable memory results.

This compares isolated mathematical GEMMs, not complete A2/A3 training: AMP
storage casts, CE, decoder and optimizer are excluded. Do not compare these
values directly with the earlier end-to-end head diagnostic or infer contest
own-kernel share, loss or full-task memory. The prototype has one warp per
16x8 tile and direct loads; production data reuse is not implemented.
No full training is launched by any command above.

Before integration: require all comparisons to pass on T4/3.6.0, inspect
large-shape errors/time/memory, then design production tiles and rerun the
same checks. Integrating AMP/LoRA/Rewind/masking requires separate model-level
tests and all main tests before any separately approved training run.
