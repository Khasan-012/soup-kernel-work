"""Isolated SM75 inline-PTX MMA experiment; not wired into A3 training."""

from __future__ import annotations

import argparse

import triton
import triton.language as tl


@triton.jit
def _pack_f16x2(lo, hi):
    lo_bits = lo.to(tl.uint16, bitcast=True).to(tl.uint32)
    hi_bits = hi.to(tl.uint16, bitcast=True).to(tl.uint32)
    return lo_bits | (hi_bits << 16)


@triton.jit
def _mma_m16n8k8(a0, a1, b0, c0, c1, c2, c3):
    return tl.inline_asm_elementwise(
        asm="mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
            "{$0,$1,$2,$3},{$4,$5},{$6},{$7,$8,$9,$10};",
        constraints="=f,=f,=f,=f,r,r,r,f,f,f,f",
        args=[a0, a1, b0, c0, c1, c2, c3],
        dtype=(tl.float32, tl.float32, tl.float32, tl.float32),
        is_pure=True,
        pack=1,
    )


@triton.jit
def mma_m16n8k8_kernel(a_ptr, b_ptr, out_ptr, M, N, K, B_TRANSPOSED_STORAGE: tl.constexpr):
    """One warp computes one 16x8 output tile with a Turing m16n8k8 MMA.

    A is row-major [M,K]. B's logical [K,N] fragment always uses the PTX
    column layout. B_TRANSPOSED_STORAGE selects whether the source is [N,K]
    row-major (forward) or [K,N] row-major (backward).
    Output is row-major [M,N]. This small correctness probe prioritizes a
    direct PTX fragment mapping over production tiling and data reuse.
    """
    lane = tl.arange(0, 32)
    group = lane >> 2
    lane_in_group = lane & 3
    m0 = tl.program_id(0) * 16
    n0 = tl.program_id(1) * 8
    acc0 = tl.full((32,), 0.0, tl.float32)
    acc1 = tl.full((32,), 0.0, tl.float32)
    acc2 = tl.full((32,), 0.0, tl.float32)
    acc3 = tl.full((32,), 0.0, tl.float32)

    for k0 in range(tl.cdiv(K, 8)):
        kk = k0 * 8
        ka = lane_in_group * 2
        a00 = tl.load(a_ptr + (m0 + group) * K + kk + ka,
                      (m0 + group < M) & (kk + ka < K), other=0.0)
        a01 = tl.load(a_ptr + (m0 + group) * K + kk + ka + 1,
                      (m0 + group < M) & (kk + ka + 1 < K), other=0.0)
        a10 = tl.load(a_ptr + (m0 + group + 8) * K + kk + ka,
                      (m0 + group + 8 < M) & (kk + ka < K), other=0.0)
        a11 = tl.load(a_ptr + (m0 + group + 8) * K + kk + ka + 1,
                      (m0 + group + 8 < M) & (kk + ka + 1 < K), other=0.0)
        af0, af1 = _pack_f16x2(a00, a01), _pack_f16x2(a10, a11)

        kb = lane_in_group * 2
        if B_TRANSPOSED_STORAGE:
            # Logical B[K,N], source is physically B^T[N,K].
            b00 = tl.load(b_ptr + (n0 + group) * K + kk + kb,
                          (n0 + group < N) & (kk + kb < K), other=0.0)
            b01 = tl.load(b_ptr + (n0 + group) * K + kk + kb + 1,
                          (n0 + group < N) & (kk + kb + 1 < K), other=0.0)
        else:
            # Logical B[K,N], source is physically row-major [K,N].
            b00 = tl.load(b_ptr + (kk + kb) * N + n0 + group,
                          (kk + kb < K) & (n0 + group < N), other=0.0)
            b01 = tl.load(b_ptr + (kk + kb + 1) * N + n0 + group,
                          (kk + kb + 1 < K) & (n0 + group < N), other=0.0)
        bf0 = _pack_f16x2(b00, b01)
        acc0, acc1, acc2, acc3 = _mma_m16n8k8(
            af0, af1, bf0, acc0, acc1, acc2, acc3
        )

    col0 = n0 + lane_in_group * 2
    col1 = col0 + 1
    tl.store(out_ptr + (m0 + group) * N + col0, acc0,
             (m0 + group < M) & (col0 < N))
    tl.store(out_ptr + (m0 + group) * N + col1, acc1,
             (m0 + group < M) & (col1 < N))
    tl.store(out_ptr + (m0 + group + 8) * N + col0, acc2,
             (m0 + group + 8 < M) & (col0 < N))
    tl.store(out_ptr + (m0 + group + 8) * N + col1, acc3,
             (m0 + group + 8 < M) & (col1 < N))


def compile_sm75_ptx():
    """Compile both B layouts for sm75 without opening a CUDA device."""
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource

    reports = []
    for b_transposed in (True, False):
        constants = {"B_TRANSPOSED_STORAGE": b_transposed}
        signature = {
            "a_ptr": "*fp16", "b_ptr": "*fp16", "out_ptr": "*fp16",
            "M": "i32", "N": "i32", "K": "i32", "B_TRANSPOSED_STORAGE": "constexpr",
        }
        compiled = triton.compile(
            ASTSource(mma_m16n8k8_kernel, signature=signature, constexprs=constants),
            target=GPUTarget("cuda", 75, 32),
            options={"num_warps": 1, "num_stages": 1},
        )
        ptx = compiled.asm["ptx"]
        count = ptx.count("mma.sync.aligned.m16n8k8")
        if ".target sm_75" not in ptx or count < 1:
            raise AssertionError(f"SM75 MMA codegen missing for transposed_storage={b_transposed}")
        reports.append({"B_TRANSPOSED_STORAGE": b_transposed, "sm75": True, "mma_sync_count": count,
                        "ptx_bytes": len(ptx), "shared_bytes": compiled.metadata.shared})
    return reports


def run_correctness_on_t4():
    """Exercise both operand layouts and custom backward only on Kaggle T4."""
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Correctness run requires Kaggle Tesla T4 (sm75)")
    name = torch.cuda.get_device_name(0)
    capability = torch.cuda.get_device_capability(0)
    if "T4" not in name or capability != (7, 5):
        raise RuntimeError(f"Only Tesla T4 (sm75) is allowed; found {name}, {capability}")

    class Projection(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, weight):
            m, k = x.shape
            n = weight.shape[0]
            y = torch.empty((m, n), device=x.device, dtype=x.dtype)
            mma_m16n8k8_kernel[(triton.cdiv(m, 16), triton.cdiv(n, 8))](
                x, weight, y, m, n, k, True, num_warps=1, num_stages=1
            )
            ctx.save_for_backward(weight)
            ctx.m = m
            return y

        @staticmethod
        def backward(ctx, grad_y):
            (weight,) = ctx.saved_tensors
            m, n = ctx.m, weight.shape[0]
            k = weight.shape[1]
            grad_x = torch.empty((m, k), device=grad_y.device, dtype=grad_y.dtype)
            mma_m16n8k8_kernel[(triton.cdiv(m, 16), triton.cdiv(k, 8))](
                grad_y.contiguous(), weight, grad_x, m, k, n, False,
                num_warps=1, num_stages=1
            )
            return grad_x, None

    torch.manual_seed(2026)
    m, k, n = 19, 32, 13
    x = torch.randn((m, k), device="cuda", dtype=torch.float16, requires_grad=True)
    w = torch.randn((n, k), device="cuda", dtype=torch.float16).requires_grad_(False)
    upstream = torch.randn((m, n), device="cuda", dtype=torch.float16)
    x_ref = x.detach().clone().requires_grad_(True)
    y_ref = x_ref @ w.t()
    (y_ref * upstream).sum().backward()
    y = Projection.apply(x, w)
    (y * upstream).sum().backward()
    torch.cuda.synchronize()
    torch.testing.assert_close(y, y_ref, rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(x.grad, x_ref.grad, rtol=3e-3, atol=3e-3)
    return {"device": name, "capability": list(capability), "shape": [m, k, n],
            "forward_max_abs_error": (y - y_ref).abs().max().item(),
            "backward_max_abs_error": (x.grad - x_ref.grad).abs().max().item()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--codegen", action="store_true")
    parser.add_argument("--correctness-on-t4", action="store_true")
    args = parser.parse_args()
    if not args.codegen and not args.correctness_on_t4:
        parser.error("select --codegen and/or --correctness-on-t4")
    if args.codegen:
        print(compile_sm75_ptx())
    if args.correctness_on_t4:
        print(run_correctness_on_t4())


if __name__ == "__main__":
    main()
