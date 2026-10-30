"""A4 frozen LM head: fuse strided chunk packing and AMP dtype conversion.

Uses native cuBLAS GEMMs. Only actual casts/copies are Triton operations;
no redundant conversion is launched when the input already matches.
"""
import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def head_cast_kernel(X, Y, N, WIDTH: tl.constexpr, SEQ: tl.constexpr,
                     S0: tl.constexpr, S1: tl.constexpr, S2: tl.constexpr,
                     BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    source = (i // (SEQ * WIDTH)) * S0 + (i // WIDTH % SEQ) * S1 + (i % WIDTH) * S2
    x = tl.load(X + source, i < N, other=0)
    tl.store(Y + i, x, i < N)


def _cast(x, dtype):
    if x.is_contiguous() and x.dtype == dtype:
        return x
    # Only 2D weights/gradients and 3D chunks enter this helper.
    if x.ndim == 2:
        seq, width = x.shape
        s0, s1, s2 = x.numel(), *x.stride()
    else:
        seq, width = x.shape[-2:]
        s0, s1, s2 = x.stride()
    out = torch.empty(x.shape, dtype=dtype, device=x.device)
    with torch.cuda.device(x.device):
        head_cast_kernel[(triton.cdiv(x.numel(), 1024),)](
            x, out, x.numel(), WIDTH=width, SEQ=seq, S0=s0, S1=s1, S2=s2,
            BLOCK=1024, num_warps=4,
        )
    return out


class _Cast(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, dtype):
        ctx.dtype = x.dtype
        return _cast(x, dtype)

    @staticmethod
    def backward(ctx, grad):
        # AMP propagates a half-rounded gradient back into the fp32 decoder.
        return _cast(grad, ctx.dtype), None


def project_head(head, hidden):
    from .frozen_lm_head import projection_support
    supported, _ = projection_support(head, hidden)
    if not supported or hidden.ndim not in (2, 3):
        return head(hidden)
    dtype = torch.get_autocast_dtype("cuda") if torch.is_autocast_enabled("cuda") else hidden.dtype
    x = _Cast.apply(hidden, dtype)
    weight = _cast(head.weight, dtype)
    with torch.autocast("cuda", enabled=False):
        return F.linear(x, weight)


def patch_head_cast():
    from . import chunked_lm_head
    chunked_lm_head.project_frozen_head = project_head
