"""A4: fused double-quantized NF4 decode for frozen-weight input gradients.

Derivation: the documented bitsandbytes NF4 format and its 0.50.2
MatMul4Bit backward contract. This is an independent elementwise decoder,
not a copied kernel. Forward, cuBLAS matmul, LoRA and dropout stay native.
"""
import functools
import math

import torch
import triton
import triton.language as tl


@triton.jit
def nf4_dq_cast_kernel(Q, S, S2, CODE2, OFFSET, CODE4, OUT, N,
                      GROUP: tl.constexpr, GROUP2: tl.constexpr,
                      NESTED: tl.constexpr, STORAGE: tl.constexpr,
                      BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = i < N
    packed = tl.load(Q + i // 2, valid, other=0).to(tl.int32)
    index = tl.where(i % 2 == 0, packed >> 4, packed & 15)
    value = tl.load(CODE4 + index)
    group = i // GROUP
    if NESTED:
        scode = tl.load(S + group, valid, other=0).to(tl.int32)
        base = tl.load(CODE2 + scode)
        scale2 = tl.load(S2 + group // GROUP2, valid, other=0)
        # Native nested scales materialize fp32 multiply, then fp32 add.
        scale = tl.inline_asm_elementwise("mul.rn.f32 $0, $1, $2;", "=f,f,f", [base, scale2], dtype=tl.float32, is_pure=True, pack=1)
        scale = tl.inline_asm_elementwise("add.rn.f32 $0, $1, $2;", "=f,f,f", [scale, tl.load(OFFSET)], dtype=tl.float32, is_pure=True, pack=1)
    else:
        scale = tl.load(S + group, valid, other=0)
    decoded = tl.inline_asm_elementwise("mul.rn.f32 $0, $1, $2;", "=f,f,f", [value, scale], dtype=tl.float32, is_pure=True, pack=1)
    # Preserve native dequantize(state.dtype).to(compute_dtype) double rounding.
    decoded = decoded.to(STORAGE).to(tl.float32)
    tl.store(OUT + i, decoded, valid)


def supports(packed, state, dtype):
    allowed = (torch.float16, torch.float32, torch.bfloat16)
    if (not isinstance(packed, torch.Tensor) or not packed.is_cuda
            or packed.dtype != torch.uint8 or not packed.is_contiguous()
            or state is None or state.quant_type != "nf4"
            or state.dtype not in allowed or dtype not in allowed
            or len(state.shape) != 2 or math.prod(state.shape) <= 0
            or math.prod(state.shape) >= 2**31
            or packed.numel() != (math.prod(state.shape) + 1) // 2
            or state.blocksize not in (32, 64, 128, 256, 512, 1024, 2048, 4096)
            or getattr(state, "packing_format_for_cpu", False)):
        return False
    tensors = [state.absmax, state.code]
    if state.code is None or state.code.numel() != 16 or state.code.dtype != torch.float32:
        return False
    if state.nested:
        second = state.state2
        if (second is None or second.nested or second.dtype != torch.float32
                or state.absmax.dtype != torch.uint8 or second.blocksize <= 0):
            return False
        tensors.extend([second.absmax, second.code, state.offset])
    elif state.absmax.dtype != torch.float32:
        return False
    return all(isinstance(t, torch.Tensor) and t.device == packed.device
               and t.is_contiguous() for t in tensors)


def dequantize_nf4(packed, state, dtype):
    """Return canonical [N,K], matching native decode then dtype conversion."""
    if not supports(packed, state, dtype):
        raise ValueError("Unsupported NF4 state for A4 decoder")
    out = torch.empty(state.shape, device=packed.device, dtype=dtype)
    second = state.state2 if state.nested else None
    storage = {torch.float16: tl.float16, torch.float32: tl.float32,
               torch.bfloat16: tl.bfloat16}[state.dtype]
    with torch.cuda.device(packed.device):
        nf4_dq_cast_kernel[(triton.cdiv(out.numel(), 1024),)](
            packed, state.absmax, second.absmax if second else state.absmax,
            second.code if second else state.code,
            state.offset if second else state.absmax, state.code, out, out.numel(),
            GROUP=state.blocksize, GROUP2=second.blocksize if second else 1,
            NESTED=state.nested, STORAGE=storage, BLOCK=1024,
            num_warps=4, enable_fp_fusion=False,
        )
    return out


def patch_nf4_backward():
    """Patch only bitsandbytes' frozen 4-bit input-gradient path, idempotently."""
    try:
        import bitsandbytes
        from bitsandbytes.autograd._functions import MatMul4Bit
    except ImportError:
        return
    # This implementation uses the canonical packed layout introduced by 0.50.2.
    # Other versions retain their original backward rather than risk transposition.
    if bitsandbytes.__version__ != "0.50.2":
        return
    original = MatMul4Bit.backward
    if getattr(original, "_soup_a4_nf4", False):
        return

    @functools.wraps(original)
    def backward(ctx, grad_output):
        if ctx.is_empty or not ctx.needs_input_grad[0] or ctx.needs_input_grad[1]:
            return original(ctx, grad_output)
        _, packed = ctx.tensors
        if not supports(packed, ctx.state, grad_output.dtype):
            return original(ctx, grad_output)
        grad_bias = None
        if ctx.needs_input_grad[3]:
            grad_bias = grad_output.sum(0, dtype=ctx.dtype_bias)
        weight = dequantize_nf4(packed, ctx.state, grad_output.dtype)
        grad_input = torch.matmul(grad_output, weight)
        return grad_input, None, None, grad_bias, None

    backward._soup_a4_nf4 = True
    MatMul4Bit.backward = staticmethod(backward)
