"""A3: the frozen vocabulary projection and its hidden-state gradient.

Only the two existing LM-head GEMMs are replaced. Autocast conversion happens
before the custom Function, preserving PyTorch's fp16 gradient rounding boundary.
The fp16/bf16 weight is reused for both GEMMs, then released with the chunk.
"""

import warnings

import torch
import triton
import triton.language as tl


ENABLED = True
_WARNED_FALLBACKS = set()


@triton.jit
def lm_head_forward_kernel(
    hidden_ptr, weight_ptr, logits_ptr,
    M, H: tl.constexpr, V: tl.constexpr,
    BM: tl.constexpr, BV: tl.constexpr, BK: tl.constexpr,
):
    # Adjacent row tiles reuse one vocabulary weight tile in L2.
    row_tiles = tl.cdiv(M, BM)
    tile = tl.program_id(0)
    rows = (tile % row_tiles) * BM + tl.arange(0, BM)
    vocab = (tile // row_tiles) * BV + tl.arange(0, BV)
    k = tl.arange(0, BK)
    acc = tl.zeros((BM, BV), tl.float32)
    for start in range(tl.cdiv(H, BK)):
        features = start * BK + k
        x = tl.load(
            hidden_ptr + rows[:, None] * H + features[None, :],
            (rows[:, None] < M) & (features[None, :] < H), other=0.0,
        )
        w = tl.load(
            weight_ptr + vocab[None, :] * H + features[:, None],
            (vocab[None, :] < V) & (features[:, None] < H), other=0.0,
        )
        acc += tl.dot(x, w)
    tl.store(
        logits_ptr + rows[:, None] * V + vocab[None, :], acc,
        (rows[:, None] < M) & (vocab[None, :] < V),
    )


@triton.jit
def lm_head_dhidden_partial_kernel(
    dlogits_ptr, weight_ptr, partial_ptr,
    M, H: tl.constexpr, V: tl.constexpr,
    PART_V: tl.constexpr, BM: tl.constexpr, BH: tl.constexpr, BK: tl.constexpr,
):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    features = tl.program_id(1) * BH + tl.arange(0, BH)
    split = tl.program_id(2)
    k = tl.arange(0, BK)
    acc = tl.zeros((BM, BH), tl.float32)
    for start in range(tl.cdiv(PART_V, BK)):
        vocab = split * PART_V + start * BK + k
        grad = tl.load(
            dlogits_ptr + rows[:, None] * V + vocab[None, :],
            (rows[:, None] < M) & (vocab[None, :] < V), other=0.0,
        )
        w = tl.load(
            weight_ptr + vocab[:, None] * H + features[None, :],
            (vocab[:, None] < V) & (features[None, :] < H), other=0.0,
        )
        acc += tl.dot(grad, w)
    tl.store(
        partial_ptr + split * M * H + rows[:, None] * H + features[None, :], acc,
        (rows[:, None] < M) & (features[None, :] < H),
    )


@triton.jit
def lm_head_dhidden_reduce_kernel(
    partial_ptr, dhidden_ptr, N, SPLITS: tl.constexpr, BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    parts = tl.arange(0, SPLITS)
    values = tl.load(
        partial_ptr + parts[:, None] * N + offsets[None, :],
        offsets[None, :] < N, other=0.0,
    )
    # No atomics: fixed reduction order, one write/cast per hidden element.
    tl.store(dhidden_ptr + offsets, tl.sum(values, axis=0), offsets < N)


class _FrozenProjection(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, weight):
        rows, features = hidden.shape
        vocab = weight.shape[0]
        logits = torch.empty((rows, vocab), device=hidden.device, dtype=hidden.dtype)
        lm_head_forward_kernel[(triton.cdiv(rows, 64) * triton.cdiv(vocab, 128),)](
            hidden, weight, logits, rows, features, vocab,
            BM=64, BV=128, BK=32, num_warps=4, num_stages=2,
        )
        ctx.save_for_backward(weight)
        ctx.rows = rows
        return logits

    @staticmethod
    def backward(ctx, grad_logits):
        (weight,) = ctx.saved_tensors
        rows, features, vocab = ctx.rows, weight.shape[1], weight.shape[0]
        grad_logits = grad_logits.contiguous()
        splits = 8 if vocab >= 4096 else 1
        part_v = triton.cdiv(triton.cdiv(vocab, splits), 64) * 64
        partial = torch.empty((splits, rows, features), device=weight.device, dtype=torch.float32)
        grad_hidden = torch.empty((rows, features), device=weight.device, dtype=grad_logits.dtype)
        lm_head_dhidden_partial_kernel[(triton.cdiv(rows, 32), triton.cdiv(features, 64), splits)](
            grad_logits, weight, partial, rows, features, vocab,
            PART_V=part_v, BM=32, BH=64, BK=64, num_warps=4, num_stages=2,
        )
        lm_head_dhidden_reduce_kernel[(triton.cdiv(rows * features, 256),)](
            partial, grad_hidden, rows * features, SPLITS=splits, BLOCK=256, num_warps=4,
        )
        return grad_hidden, None


def projection_support(head, hidden):
    """Explain dispatch; unsupported modules keep their original semantics.

    A normal task head is a frozen, bias-free nn.Linear. Hooks or an instance
    forward override could change its result, so they must not be bypassed.
    """
    if not ENABLED:
        return False, "A3 disabled"
    if type(head) is not torch.nn.Linear:
        return False, "head is not a plain nn.Linear"
    if head.bias is not None or head.weight.requires_grad:
        return False, "head has bias or trainable weights"
    if ("forward" in head.__dict__ or head._forward_hooks or head._forward_pre_hooks
            or head._backward_hooks or head._backward_pre_hooks):
        return False, "head has a forward override or module hooks"
    if not hidden.is_cuda or head.weight.device != hidden.device:
        return False, "head and hidden must share a CUDA device"
    allowed = (torch.float16, torch.bfloat16, torch.float32)
    if hidden.dtype not in allowed or head.weight.dtype not in allowed:
        return False, "unsupported storage dtype"
    dtype = torch.get_autocast_dtype("cuda") if torch.is_autocast_enabled("cuda") else hidden.dtype
    if dtype not in (torch.float16, torch.bfloat16):
        return False, "fp32 without autocast stays with PyTorch"
    if not torch.is_autocast_enabled("cuda") and head.weight.dtype != dtype:
        return False, "storage dtypes differ without autocast"
    if hidden.dim() < 2 or hidden.shape[-1] != head.in_features or hidden.numel() == 0:
        return False, "unsupported hidden shape"
    if not head.weight.is_contiguous():
        return False, "noncontiguous weight"
    return True, "Triton frozen head"


def project_frozen_head(head, hidden):
    """Project one A2 chunk, preserving autocast and the PyTorch fallback.

    Conversion/reshape are ordinary differentiable PyTorch operations, so a
    strided fp32 input receives exactly the fp16-rounded, upcast gradient that
    AMP's original Linear would propagate. No base weight or training setting
    is changed. Failures for supported inputs are never caught or hidden.
    """
    supported, reason = projection_support(head, hidden)
    if not supported:
        # A CPU/fp32 reference or explicit A2 opt-out is intentional. Other
        # CUDA half/AMP fallbacks must be visible in a real task's train.log.
        half_compute = hidden.dtype in (torch.float16, torch.bfloat16) or torch.is_autocast_enabled("cuda")
        if ENABLED and hidden.is_cuda and half_compute and reason not in _WARNED_FALLBACKS:
            _WARNED_FALLBACKS.add(reason)
            warnings.warn(f"A3 frozen head fallback: {reason}", RuntimeWarning, stacklevel=2)
        return head(hidden)
    dtype = torch.get_autocast_dtype("cuda") if torch.is_autocast_enabled("cuda") else hidden.dtype
    x = hidden.to(dtype).reshape(-1, hidden.shape[-1]).contiguous()
    weight = head.weight.to(dtype)
    logits = _FrozenProjection.apply(x, weight)
    return logits.view(*hidden.shape[:-1], head.out_features)
