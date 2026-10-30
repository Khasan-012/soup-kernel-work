"""Chunked LM head and causal-LM loss (stage A1 + A2).

The unchunked step builds ``[batch, seq, vocab]`` logits, upcasts them to fp32
and keeps another buffer of that shape for the cross-entropy.  With the task's
vocabulary (151 936) that is the largest single allocation of the run.  Here the
head and the loss run ``chunk`` positions at a time: each chunk runs the head,
takes the cross-entropy sum and the gradient on its own slice of the hidden
states, and drops its logits.  No graph over the logits outlives its chunk.

The maths is the one of ``transformers``' ``ForCausalLMLoss``:

* ``labels`` are shifted one position left; the last position predicts nothing;
* the denominator is ``num_items_in_batch`` when the trainer passes one (HF
  counts ``labels[..., 1:] != -100``), else the number of labelled positions;
* the head is frozen, so only the gradient on the hidden states is taken.

Stage A2 replaces the per-chunk cross-entropy with two Triton kernels
(``ce_forward_kernel`` and ``ce_backward_kernel``): one online pass over the
vocabulary computes the log-sum-exp, the entropy, the target log-probability and
the argmax at once, and the backward pass rebuilds the softmax from the saved
row maximum and shifted log-normalizer. The PyTorch path of A1 stays as the
fallback for a CPU, an unsupported dtype or a non-contiguous tensor.

What differs from the unchunked step is the order of the final sum and the shape
of each matmul, so the loss and the gradients agree with it to rounding, not bit
for bit.  The per-token terms that TRL and Soup's rewind recorder read are
produced here as well, so no full logits are needed downstream either.
"""

from __future__ import annotations

import functools
import inspect
from dataclasses import dataclass
from typing import Any, Callable, Optional

import triton
import triton.language as tl

IGNORE_INDEX = -100

# Positions per head slice.  batch * CHUNK is the M of the head GEMM; 4 * 128
# keeps it at 512 rows, still compute-bound on a T4, while the logits term
# drops from ~1.2 GB (fp32) to ~0.3 GB.
CHUNK = 128

# Vocabulary elements per Triton block.  4 * 128 rows x 151 936 columns is
# streamed once per pass; 4096 keeps the block reductions cheap without
# spilling.
BLOCK_V = 4096
CE_NUM_WARPS = 8

# Explicit opt-out for comparing with A1. Launch failures are never swallowed:
# a supported input that fails to compile or launch is a bug to investigate.
_TRITON_ENABLED = True


# The trainer leaves the last chunked step under this name for Soup's rewind
# recorder, which reads it exactly as it reads the chunked step of Soup's own
# ``stream_loss_chunk`` (#1693).
STEP_ATTR = "_soup_chunked_step"

_ENABLED = False
_PATCHED = False
_PENDING: list = []
_WARNED = False
# Test-only switch: lets the CPU tests exercise the chunked path without a GPU.
# `apply()` never sets it.
_FORCE = False


@dataclass(frozen=True)
class ChunkedStep:
    """What a chunked forward leaves for the trainer's metrics.

    ``token_entropy`` is ``[batch, seq]`` over every position (TRL weights it
    by the attention mask), ``token_loss`` is ``[batch, seq]`` float32 with the
    cross-entropy of ``labels[:, t + 1]`` at position ``t`` and 0.0 where that
    label is ignored, ``correct`` is the number of labelled positions whose
    argmax is the label and ``labelled`` is the number of labelled positions.
    """

    token_entropy: Any
    token_loss: Any
    correct: Any
    labelled: Any


def shift_for_next_token(labels: Any, ignore_index: int = IGNORE_INDEX) -> Any:
    """``labels`` moved one position left, exactly as ``ForCausalLMLoss`` does."""
    import torch.nn.functional as functional

    return functional.pad(labels, (0, 1), value=ignore_index)[..., 1:]


@triton.jit
def ce_forward_kernel(
    logits_ptr,
    target_ptr,
    max_ptr,
    log_sum_ptr,
    entropy_ptr,
    loss_ptr,
    correct_ptr,
    cols,
    ignore_index,
    BLOCK_V: tl.constexpr,
):
    """One row of logits: log-sum-exp, entropy, target loss and argmax in one pass.

    The row is streamed block by block with the online softmax update, so the
    maximum, exponential sum and shifted first moment are built together.
    Keeping the maximum separate avoids cancellation under large common
    logit shifts, including in the backward. No vocabulary-sized stats remain.
    """
    row = tl.program_id(0)
    row_start = row * cols
    target = tl.load(target_ptr + row)
    labelled = target != ignore_index
    safe_target = tl.where(labelled, target, 0)

    offs = tl.arange(0, BLOCK_V)
    running_max = -float("inf")
    running_sum = 0.0
    running_weight = 0.0
    running_best = -float("inf")
    running_argmax = 0
    for start in range(0, cols, BLOCK_V):
        idx = start + offs
        mask = idx < cols
        x = tl.load(logits_ptr + row_start + idx, mask=mask, other=-float("inf"))
        x = x.to(tl.float32)
        block_max = tl.max(x, axis=0)
        block_argmax = tl.argmax(x, axis=0)
        new_max = tl.maximum(running_max, block_max)
        # The first block rescales by exp(-inf) = 0, so the sums start clean.
        alpha = tl.exp(running_max - new_max)
        shift = tl.where(running_sum > 0.0, running_max - new_max, 0.0)
        centered = x - new_max
        weight = tl.where(mask, tl.exp(centered), 0.0)
        running_weight = alpha * (running_weight + shift * running_sum) + tl.sum(
            weight * tl.where(mask, centered, 0.0), axis=0
        )
        running_sum = running_sum * alpha + tl.sum(weight, axis=0)
        # Strictly greater keeps the first maximum, as ``argmax`` does.
        take = block_max > running_best
        running_argmax = tl.where(take, start + block_argmax, running_argmax)
        running_best = tl.maximum(running_best, block_max)
        running_max = new_max

    target_logit = tl.load(
        logits_ptr + row_start + safe_target, mask=labelled, other=0.0
    ).to(tl.float32)
    log_sum = tl.log(running_sum)
    entropy = log_sum - running_weight / running_sum
    loss = tl.where(labelled, (running_max - target_logit) + log_sum, 0.0)
    hit = labelled & (running_argmax == target)
    tl.store(max_ptr + row, running_max)
    tl.store(log_sum_ptr + row, log_sum)
    tl.store(entropy_ptr + row, entropy)
    tl.store(loss_ptr + row, loss)
    tl.store(correct_ptr + row, hit.to(tl.int32))


@triton.jit
def ce_backward_kernel(
    logits_ptr,
    target_ptr,
    max_ptr,
    log_sum_ptr,
    dlogits_ptr,
    scale_ptr,
    cols,
    ignore_index,
    BLOCK_V: tl.constexpr,
):
    """The logits gradient: ``(softmax - onehot) * scale`` in independent tiles.

    ``scale`` is the incoming gradient on the loss sum, i.e. ``1 / denominator``
    times whatever the trainer's backward asks for; it is read on the device, so
    no host synchronisation is needed to pass it.
    """
    row = tl.program_id(0)
    row_start = row * cols
    scale = tl.load(scale_ptr).to(tl.float32)
    row_max = tl.load(max_ptr + row)
    log_sum = tl.load(log_sum_ptr + row)
    target = tl.load(target_ptr + row)
    labelled = target != ignore_index
    safe_target = tl.where(labelled, target, 0)

    idx = tl.program_id(1) * BLOCK_V + tl.arange(0, BLOCK_V)
    mask = idx < cols
    x = tl.load(logits_ptr + row_start + idx, mask=mask, other=-float("inf")).to(tl.float32)
    prob = tl.exp((x - row_max) - log_sum)
    grad = (prob - tl.where(idx == safe_target, 1.0, 0.0)) * scale
    # An ignored position has no loss term, so it has no gradient either.
    grad = tl.where(labelled, grad, 0.0)
    tl.store(
        dlogits_ptr + row_start + idx,
        grad.to(dlogits_ptr.dtype.element_ty),
        mask=mask,
    )


def _triton_ce_ready(logits: Any, target: Any) -> bool:
    """Whether the Triton cross-entropy can take these two tensors."""
    import torch

    if not _TRITON_ENABLED:
        return False
    if not isinstance(logits, torch.Tensor) or not logits.is_cuda:
        return False
    if not isinstance(target, torch.Tensor) or target.device != logits.device:
        return False
    if logits.dim() != 2 or target.dim() != 1:
        return False
    if logits.shape[0] != target.shape[0] or logits.shape[0] < 1 or logits.shape[1] < 1:
        return False
    if not logits.is_contiguous() or not target.is_contiguous():
        return False
    if logits.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        return False
    if target.dtype not in (torch.int32, torch.int64):
        return False
    # The kernels address rows with int32 offsets, so the whole matrix has to
    # fit in 2**31 elements.  A chunk of this task is ~8e7, far below that.
    return logits.numel() < 2**31


@functools.lru_cache(maxsize=None)
def _triton_ce_function() -> type:
    """The autograd function around the two kernels, built once on first use."""
    import torch

    class _TritonCrossEntropy(torch.autograd.Function):
        """Cross-entropy sum plus the per-row terms, in two launches per chunk."""

        @staticmethod
        def forward(ctx, logits, target, ignore_index=IGNORE_INDEX):
            rows, cols = logits.shape
            device = logits.device
            row_max = torch.empty(rows, dtype=torch.float32, device=device)
            log_sum = torch.empty(rows, dtype=torch.float32, device=device)
            entropy = torch.empty(rows, dtype=torch.float32, device=device)
            loss = torch.empty(rows, dtype=torch.float32, device=device)
            correct = torch.empty(rows, dtype=torch.int32, device=device)
            ce_forward_kernel[(rows,)](
                logits,
                target,
                row_max,
                log_sum,
                entropy,
                loss,
                correct,
                cols,
                ignore_index,
                BLOCK_V=BLOCK_V,
                num_warps=CE_NUM_WARPS,
            )
            ctx.save_for_backward(logits, target, row_max, log_sum)
            ctx.ignore_index = ignore_index
            # Same contract as the PyTorch path: only the loss sum is
            # differentiable, the metrics are read, not differentiated.
            ctx.mark_non_differentiable(entropy, loss, correct)
            return loss.sum(), entropy, loss, correct

        @staticmethod
        def backward(ctx, grad_total, _grad_entropy, _grad_loss, _grad_correct):
            logits, target, row_max, log_sum = ctx.saved_tensors
            rows, cols = logits.shape
            dlogits = torch.empty_like(logits)
            scale = grad_total.to(torch.float32)
            ce_backward_kernel[(rows, triton.cdiv(cols, BLOCK_V))](
                logits,
                target,
                row_max,
                log_sum,
                dlogits,
                scale,
                cols,
                ctx.ignore_index,
                BLOCK_V=BLOCK_V,
                num_warps=CE_NUM_WARPS,
            )
            return (dlogits, None, None)[:len(ctx.needs_input_grad)]

    return _TritonCrossEntropy


def _chunk_terms(head: Callable[[Any], Any], piece: Any, target: Any, ignore_index: int):
    """One chunk: the cross-entropy sum and the metric terms.

    On a GPU with a supported dtype this is the Triton pair: one pass over the
    vocabulary gives the sum, the entropy, the per-token loss and the argmax,
    and the backward pass rebuilds the softmax from the saved row statistics. The
    PyTorch path is kept as the fallback for a CPU, another dtype or a
    non-contiguous tensor; both return the same four terms.
    """
    import torch
    import torch.nn.functional as functional

    logits = head(piece)
    if _TRITON_ENABLED and logits.is_cuda:
        flat = logits.reshape(-1, logits.shape[-1])
        # The chunk is a column slice of ``labels``, so this is a small copy;
        # the Triton path needs a flat, contiguous target.
        flat_target = target.reshape(-1).contiguous()
        if _triton_ce_ready(flat, flat_target):
            total, entropy, token_loss, correct = _triton_ce_function().apply(
                flat, flat_target, ignore_index
            )
            return (
                total,
                entropy.view(target.shape),
                token_loss.view(target.shape),
                correct.to(torch.int64).sum(),
            )
    with torch.no_grad():
        log_probs = functional.log_softmax(logits, dim=-1)
        entropy = -(log_probs.exp() * log_probs).sum(-1)
        labelled = target != ignore_index
        correct = ((logits.argmax(dim=-1) == target) & labelled).sum()
        at_label = log_probs.gather(-1, target.clamp(min=0).unsqueeze(-1)).squeeze(-1)
        token_loss = (-at_label).float() * labelled
        # Gone before the fp32 copy below is made, so the two do not add up.
        del log_probs
    # `ForCausalLMLoss` upcasts before the loss; so does each chunk.  The
    # view is only for the flattened CE and is not kept.
    wide = logits.float()
    total = functional.cross_entropy(
        wide.reshape(-1, wide.shape[-1]),
        target.reshape(-1),
        ignore_index=ignore_index,
        reduction="sum",
    )
    return total, entropy, token_loss, correct


@functools.lru_cache(maxsize=None)
def _chunked_function() -> type:
    """The autograd function, built once on first use (torch is imported lazily)."""
    import torch

    class _ChunkedHeadLoss(torch.autograd.Function):
        """The loss, with the gradient on the hidden states taken in ``forward``."""

        @staticmethod
        def forward(ctx, hidden, head, target, chunk, denominator, ignore_index, with_grad):
            grad = torch.zeros_like(hidden) if with_grad else None
            total = torch.zeros((), dtype=torch.float32, device=hidden.device)
            entropies, losses = [], []
            correct = torch.zeros((), dtype=torch.long, device=hidden.device)
            for start in range(0, hidden.shape[1], chunk):
                stop = min(start + chunk, hidden.shape[1])
                piece = hidden[:, start:stop].detach()
                labels = target[:, start:stop]
                if with_grad:
                    piece.requires_grad_(True)
                    # `forward` of an autograd.Function runs with grad disabled.
                    with torch.enable_grad():
                        part, entropy, token_loss, hits = _chunk_terms(
                            head, piece, labels, ignore_index
                        )
                        # Scaled BEFORE the gradient is taken, so each token's
                        # gradient is rounded at the scale the unchunked step
                        # rounds it at.
                        (piece_grad,) = torch.autograd.grad(
                            part / denominator, piece, allow_unused=True
                        )
                    if piece_grad is not None:
                        grad[:, start:stop] = piece_grad
                else:
                    with torch.no_grad():
                        part, entropy, token_loss, hits = _chunk_terms(
                            head, piece, labels, ignore_index
                        )
                total = total + part.detach()
                entropies.append(entropy)
                losses.append(token_loss)
                correct = correct + hits
            token_entropy = torch.cat(entropies, dim=1)
            token_loss = torch.cat(losses, dim=1)
            ctx.with_grad = with_grad
            if with_grad:
                ctx.save_for_backward(grad)
            ctx.mark_non_differentiable(token_entropy, token_loss, correct)
            return total / denominator, token_entropy, token_loss, correct

        @staticmethod
        def backward(ctx, grad_loss, _grad_entropy, _grad_token_loss, _grad_correct):
            if not ctx.with_grad:  # nothing asked for a gradient
                return (None,) * 7
            (grad,) = ctx.saved_tensors
            return (grad * grad_loss, None, None, None, None, None, None)

    return _ChunkedHeadLoss


def chunked_causal_lm_loss(
    hidden: Any,
    head: Callable[[Any], Any],
    labels: Any,
    chunk: int = CHUNK,
    *,
    num_items_in_batch: Any = None,
    shift_labels: Any = None,
    ignore_index: int = IGNORE_INDEX,
) -> tuple[Any, ChunkedStep]:
    """``ForCausalLMLoss(head(hidden), labels)`` without the full logits.

    ``head`` maps ``[batch, n, hidden]`` to ``[batch, n, vocab]`` and must be
    frozen: the gradient is taken on the hidden states only.  ``labels``,
    ``shift_labels`` and ``num_items_in_batch`` mean what they mean to
    ``ForCausalLMLoss``: the mean over the labelled positions, or the sum over
    ``num_items_in_batch`` when the trainer passes one.
    """
    import torch

    if isinstance(chunk, bool) or not isinstance(chunk, int) or chunk < 1:
        raise ValueError(f"chunk must be a positive integer, got {chunk!r}")
    target = shift_labels if shift_labels is not None else shift_for_next_token(
        labels, ignore_index
    )
    target = target.to(hidden.device)
    if target.shape != hidden.shape[:2]:
        raise ValueError(
            f"labels of shape {tuple(target.shape)} do not match hidden states of "
            f"shape {tuple(hidden.shape)}"
        )
    labelled = (target != ignore_index).sum()
    if num_items_in_batch is None:
        # `ForCausalLMLoss` with reduction="mean"; the clamp only keeps a fully
        # ignored batch from producing 0/0, it does not touch this task.
        denominator = labelled.clamp(min=1)
    elif torch.is_tensor(num_items_in_batch):
        denominator = num_items_in_batch.to(hidden.device)
    else:
        denominator = num_items_in_batch
    with_grad = torch.is_grad_enabled() and hidden.requires_grad
    loss, token_entropy, token_loss, correct = _chunked_function().apply(
        hidden, head, target, chunk, denominator, ignore_index, with_grad
    )
    step = ChunkedStep(
        token_entropy=token_entropy, token_loss=token_loss, correct=correct, labelled=labelled
    )
    return loss, step


# --------------------------------------------------------------------------- #
# Wiring: the causal LM forward and TRL's compute_loss
# --------------------------------------------------------------------------- #


def _unwrap(model: Any) -> Any:
    """The causal LM under a PEFT/DDP wrapper, or ``model`` itself."""
    module = getattr(model, "module", None)
    if module is not None and hasattr(module, "config"):
        model = module
    get_base_model = getattr(model, "get_base_model", None)
    if callable(get_base_model):
        try:
            return get_base_model()
        except Exception:  # pragma: no cover - a half-built wrapper
            return model
    return model


def _can_chunk_model(causal_lm: Any) -> bool:
    """Cheap, side-effect free check that ``causal_lm`` can take the chunked path."""
    import torch

    config = getattr(causal_lm, "config", None)
    if config is None:
        return False
    if getattr(config, "final_logit_softcapping", None) is not None:
        return False
    get_head = getattr(causal_lm, "get_output_embeddings", None)
    if not callable(get_head):
        return False
    head = get_head()
    if head is None:
        return False
    if getattr(head, "in_features", None) != getattr(config, "hidden_size", None):
        return False
    loss_fn = getattr(causal_lm, "loss_function", None)
    if loss_fn is not None and getattr(loss_fn, "__name__", "") != "ForCausalLMLoss":
        return False
    # The gradient is taken on the hidden states only, so the head must be
    # frozen; a trainable head would silently lose its gradient.
    try:
        if any(parameter.requires_grad for parameter in head.parameters()):
            return False
    except Exception:  # pragma: no cover - exotic head
        return False
    return isinstance(getattr(head, "weight", None), torch.Tensor)


def _take_step() -> Optional[ChunkedStep]:
    """The step of the last chunked forward, once."""
    step = _PENDING.pop() if _PENDING else None
    _PENDING.clear()
    return step


def _publish(step: ChunkedStep) -> None:
    _PENDING.clear()
    _PENDING.append(step)


def _chunked_forward(original: Callable, signature: inspect.Signature) -> Callable:
    @functools.wraps(original)
    def forward(self, *args, **kwargs):
        if not _ENABLED:
            return original(self, *args, **kwargs)
        try:
            bound = signature.bind(self, *args, **kwargs)
        except TypeError:  # pragma: no cover - a caller the real forward rejects
            return original(self, *args, **kwargs)
        bound.apply_defaults()
        labels = bound.arguments.get("labels")
        if labels is None or not _can_chunk_model(self):
            return original(self, *args, **kwargs)
        logits_to_keep = bound.arguments.get("logits_to_keep", 0)
        if logits_to_keep not in (0, None):
            return original(self, *args, **kwargs)
        if not getattr(labels, "is_cuda", False) and not _FORCE:
            return original(self, *args, **kwargs)
        if labels.dim() != 2:
            return original(self, *args, **kwargs)
        input_ids = bound.arguments.get("input_ids")
        inputs_embeds = bound.arguments.get("inputs_embeds")
        if input_ids is not None:
            seq = input_ids.shape[1]
        elif inputs_embeds is not None:
            seq = inputs_embeds.shape[1]
        else:  # pragma: no cover - the real forward rejects this too
            return original(self, *args, **kwargs)
        if labels.shape[1] != seq:
            return original(self, *args, **kwargs)

        # The decoder runs exactly as before; only the head is held back, so
        # its hidden states are what the loss needs and no logits are built.
        call_kwargs = {
            key: value
            for key, value in bound.arguments.items()
            if key not in ("self", "kwargs", "labels")
        }
        call_kwargs.update(bound.arguments.get("kwargs") or {})
        call_kwargs["labels"] = None
        # `can_return_tuple` would turn the output into a tuple otherwise; the
        # trainer always reads a ModelOutput.
        call_kwargs["return_dict"] = True
        num_items_in_batch = call_kwargs.pop("num_items_in_batch", None)

        head = self.get_output_embeddings()
        captured: dict = {}
        previous = head.__dict__.get("forward")

        def capture(hidden_states, *head_args, **head_kwargs):
            captured["hidden"] = hidden_states
            # Stands for the logits: no vocabulary axis, so it costs nothing.
            return hidden_states.new_empty((*hidden_states.shape[:-1], 0))

        head.forward = capture
        try:
            outputs = original(self, **call_kwargs)
        finally:
            if previous is None:
                head.__dict__.pop("forward", None)
            else:
                head.forward = previous

        hidden = captured.get("hidden")
        if hidden is None:  # pragma: no cover - incompatible head wiring
            # The decoder already ran; run it again the plain way rather than
            # returning a wrong loss.
            return original(self, *args, **kwargs)
        loss, step = chunked_causal_lm_loss(
            hidden, head, labels, CHUNK, num_items_in_batch=num_items_in_batch
        )
        _publish(step)
        rest = {
            key: value for key, value in outputs.items() if key not in ("loss", "logits")
        }
        return type(outputs)(loss=loss, logits=None, **rest)

    forward._soup_chunked_head = True
    return forward


def _log_trl_metrics(trainer: Any, mode: str, inputs: Any, step: ChunkedStep) -> None:
    """``entropy``, ``num_tokens`` and ``mean_token_accuracy`` as TRL logs them."""
    import torch

    metrics = getattr(trainer, "_metrics", None)
    gather = getattr(getattr(trainer, "accelerator", None), "gather_for_metrics", None)
    if metrics is None or gather is None:  # pragma: no cover - other TRL versions
        return
    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        entropy = (step.token_entropy * attention_mask).sum() / attention_mask.sum()
    elif "position_ids" in inputs:
        entropy = step.token_entropy.mean()
    else:  # pragma: no cover - TRL raises here too
        return
    metrics[mode]["entropy"].append(gather(entropy).mean().item())
    if mode == "train":
        if attention_mask is not None:
            num_tokens = gather(attention_mask.sum()).sum().item()
        else:
            local = torch.tensor(
                inputs["position_ids"].size(1), device=inputs["position_ids"].device
            )
            num_tokens = gather(local).sum().item()
        if hasattr(trainer, "_total_train_tokens"):
            trainer._total_train_tokens += num_tokens
    if hasattr(trainer, "_total_train_tokens"):
        metrics[mode]["num_tokens"] = [trainer._total_train_tokens]
    total = gather(step.labelled).sum()
    accuracy = (gather(step.correct).sum() / total).item() if total > 0 else 0.0
    metrics[mode]["mean_token_accuracy"].append(accuracy)


def _trainer_can_chunk(trainer: Any, model: Any, inputs: Any) -> bool:
    """Mirror of the model's own guard, decided before the forward runs."""
    if not _ENABLED:
        return False
    if "labels" not in inputs or "shift_labels" in inputs:
        return False
    if getattr(trainer, "label_smoother", None) is not None:
        return False
    if getattr(trainer, "compute_loss_func", None) is not None:
        return False
    args = getattr(trainer, "args", None)
    if getattr(args, "use_liger_kernel", False):
        return False
    if getattr(trainer, "num_virtual_tokens", 0):
        return False
    labels = inputs.get("labels")
    import torch

    if not isinstance(labels, torch.Tensor):
        return False
    if not labels.is_cuda and not _FORCE:
        return False
    unwrap_model = getattr(getattr(trainer, "accelerator", None), "unwrap_model", None)
    if callable(unwrap_model):
        try:
            model = unwrap_model(model)
        except Exception:  # pragma: no cover
            pass
    return _can_chunk_model(_unwrap(model))


def _chunked_compute_loss(original: Callable) -> Callable:
    @functools.wraps(original)
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if not _trainer_can_chunk(self, model, inputs):
            return original(self, model, inputs, return_outputs, num_items_in_batch)
        from transformers import Trainer

        mode = "train" if self.model.training else "eval"
        # As TRL's `compute_loss` does before it delegates.
        inputs.pop("_prediction_loss_only", None)
        inputs["use_cache"] = False
        loss, outputs = Trainer.compute_loss(
            self, model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch
        )
        step = _take_step()
        if step is None:  # pragma: no cover - the model should have chunked
            global _WARNED
            if not _WARNED:
                _WARNED = True
                import warnings

                warnings.warn(
                    "solution: the model did not take the chunked LM-head path; "
                    "TRL's entropy/accuracy are missing for this step",
                    stacklevel=2,
                )
        else:
            setattr(self, STEP_ATTR, step)
            _log_trl_metrics(self, mode, inputs, step)
        return (loss, outputs) if return_outputs else loss

    compute_loss._soup_chunked = True
    return compute_loss


def patch_chunked_lm_head() -> None:
    """Send the Qwen2 causal-LM head and TRL's loss through the chunked path.

    Class patches work before or after Soup builds the model and the trainer:
    Python looks the methods up on every call.  Everything degrades to the
    original code when the guard does not hold (no CUDA, no labels, a trainable
    or soft-capped head, a label smoother, ...).
    """
    global _ENABLED, _PATCHED

    if _PATCHED:
        _ENABLED = True
        return
    try:
        from transformers.models.qwen2 import modeling_qwen2
    except Exception:  # pragma: no cover - transformers without qwen2
        return
    try:
        from trl import SFTTrainer
    except Exception:  # pragma: no cover - trl missing
        SFTTrainer = None

    cls = modeling_qwen2.Qwen2ForCausalLM
    original = cls.forward
    if not getattr(original, "_soup_chunked_head", False):
        try:
            signature = inspect.signature(original)
        except (TypeError, ValueError):  # pragma: no cover
            signature = None
        if signature is not None:
            cls.forward = _chunked_forward(original, signature)

    if SFTTrainer is not None:
        original_loss = SFTTrainer.compute_loss
        if not getattr(original_loss, "_soup_chunked", False):
            SFTTrainer.compute_loss = _chunked_compute_loss(original_loss)

    _PATCHED = True
    _ENABLED = True
