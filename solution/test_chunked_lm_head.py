"""Correctness of the chunked LM head against the unchunked loss, on CPU.

These tests need no GPU and no Triton kernel: they check the maths of
``chunked_causal_lm_loss`` and that the class patch falls back to the plain
forward when the chunked path cannot run.
"""

import pytest
import torch
import torch.nn.functional as F

from solution.chunked_lm_head import (
    chunked_causal_lm_loss,
    patch_chunked_lm_head,
    shift_for_next_token,
)

TOL = 1e-5


def _reference(hidden, head, labels, num_items_in_batch=None):
    """The unchunked step: full logits, ``ForCausalLMLoss`` and TRL's metrics."""
    ref_hidden = hidden.detach().clone().requires_grad_(True)
    logits = head(ref_hidden).float()
    shift = shift_for_next_token(labels)
    total = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        shift.reshape(-1),
        ignore_index=-100,
        reduction="sum",
    )
    if num_items_in_batch is None:
        denominator = (shift != -100).sum().clamp(min=1)
    elif torch.is_tensor(num_items_in_batch):
        denominator = num_items_in_batch
    else:
        denominator = num_items_in_batch
    loss = total / denominator
    loss.backward()
    with torch.no_grad():
        log_probs = F.log_softmax(logits, dim=-1)
        entropy = -(log_probs.exp() * log_probs).sum(-1)
        labelled = shift != -100
        token_loss = (
            -log_probs.gather(-1, shift.clamp(min=0).unsqueeze(-1)).squeeze(-1)
        ).float() * labelled
        correct = ((logits.argmax(dim=-1) == shift) & labelled).sum()
    return loss.detach(), ref_hidden.grad, entropy, token_loss, correct, labelled.sum()


def _case(shape, vocab, seed=0, device="cpu"):
    torch.manual_seed(seed)
    batch, seq, hidden = shape
    head = torch.nn.Linear(hidden, vocab, bias=False).to(device)
    for parameter in head.parameters():
        parameter.requires_grad_(False)
    states = torch.randn(batch, seq, hidden, requires_grad=True, device=device)
    labels = torch.randint(0, vocab, (batch, seq), device=device)
    labels[:, :2] = -100
    labels[0, 2] = -100
    labels[1, -1] = -100
    return head, states, labels


@pytest.mark.parametrize("chunk", [1, 3, 7, 64])
@pytest.mark.parametrize("shape,vocab", [((2, 9, 8), 71), ((3, 5, 16), 31)])
def test_loss_grad_and_metrics_match_full_logits(chunk, shape, vocab):
    head, hidden, labels = _case(shape, vocab)

    ref_loss, ref_grad, ref_entropy, ref_tl, ref_correct, ref_labelled = _reference(
        hidden, head, labels
    )

    loss, step = chunked_causal_lm_loss(hidden, head, labels, chunk)
    loss.backward()

    torch.testing.assert_close(loss.detach(), ref_loss, rtol=TOL, atol=TOL)
    torch.testing.assert_close(hidden.grad, ref_grad, rtol=1e-4, atol=TOL)
    torch.testing.assert_close(step.token_entropy, ref_entropy, rtol=1e-4, atol=TOL)
    torch.testing.assert_close(step.token_loss, ref_tl, rtol=1e-4, atol=TOL)
    assert step.correct.item() == ref_correct.item()
    assert step.labelled.item() == ref_labelled.item()
    # The head is frozen: the gradient is taken on the hidden states only.
    assert head.weight.grad is None


def test_num_items_in_batch_is_the_denominator():
    head, hidden, labels = _case((2, 9, 8), 71)
    labelled = int((shift_for_next_token(labels) != -100).sum())
    denominator = torch.tensor(float(2 * labelled))

    ref_loss, ref_grad, *_ = _reference(
        hidden, head, labels, num_items_in_batch=denominator
    )
    loss, _ = chunked_causal_lm_loss(
        hidden, head, labels, 4, num_items_in_batch=denominator
    )
    loss.backward()

    torch.testing.assert_close(loss.detach(), ref_loss, rtol=TOL, atol=TOL)
    torch.testing.assert_close(hidden.grad, ref_grad, rtol=1e-4, atol=TOL)


def test_no_grad_path_returns_detached_terms():
    head, hidden, labels = _case((2, 9, 8), 71)
    with torch.no_grad():
        loss, step = chunked_causal_lm_loss(hidden, head, labels, 4)
    assert not loss.requires_grad
    assert step.token_entropy.shape == labels.shape
    assert step.token_loss.shape == labels.shape
    assert step.token_loss.dtype == torch.float32
    # Ignored positions carry exactly 0.0, as Soup's rewind recorder expects.
    shift = shift_for_next_token(labels)
    assert bool((step.token_loss[shift == -100] == 0).all())


def test_patch_falls_back_to_the_plain_forward_without_cuda():
    patch_chunked_lm_head()
    from transformers import Qwen2Config, Qwen2ForCausalLM

    config = Qwen2Config(
        vocab_size=71,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
    )
    model = Qwen2ForCausalLM(config).eval()
    input_ids = torch.randint(0, config.vocab_size, (2, 9))
    with torch.no_grad():
        outputs = model(input_ids=input_ids, labels=input_ids.clone())

    assert outputs.logits is not None
    assert outputs.logits.shape == (2, 9, config.vocab_size)
    assert outputs.loss is not None and torch.isfinite(outputs.loss)


def test_patch_is_idempotent():
    patch_chunked_lm_head()
    from transformers.models.qwen2 import modeling_qwen2

    first = modeling_qwen2.Qwen2ForCausalLM.forward
    patch_chunked_lm_head()
    assert modeling_qwen2.Qwen2ForCausalLM.forward is first


def test_trl_metrics_are_computed_from_the_step():
    """`_log_trl_metrics` reproduces TRL's entropy, tokens and accuracy."""
    from collections import defaultdict
    from types import SimpleNamespace

    import solution.chunked_lm_head as chunked

    class FakeTrainer:
        _metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
        _total_train_tokens = 0
        accelerator = SimpleNamespace(gather_for_metrics=lambda value: value)

    step = chunked.ChunkedStep(
        token_entropy=torch.tensor([[1.0, 2.0, 3.0]]),
        token_loss=torch.zeros(1, 3),
        correct=torch.tensor(2),
        labelled=torch.tensor(3),
    )
    trainer = FakeTrainer()
    chunked._log_trl_metrics(
        trainer, "train", {"attention_mask": torch.tensor([[1, 1, 0]])}, step
    )

    assert trainer._metrics["train"]["entropy"] == [pytest.approx(1.5)]
    assert trainer._total_train_tokens == 2
    assert trainer._metrics["train"]["num_tokens"] == [2]
    assert trainer._metrics["train"]["mean_token_accuracy"] == [pytest.approx(2 / 3)]


def test_chunked_forward_matches_the_plain_forward(monkeypatch):
    """The whole patched forward, forced on CPU, against the original one."""
    import copy

    import solution.chunked_lm_head as chunked

    patch_chunked_lm_head()
    monkeypatch.setattr(chunked, "_FORCE", True)

    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(0)
    config = Qwen2Config(
        vocab_size=71,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
    )
    model = Qwen2ForCausalLM(config).train()
    # The real setup freezes the tied head and trains LoRA adapters; here one
    # projection stands in for the adapter so the gradient has somewhere to go.
    model.lm_head.weight.requires_grad_(False)
    model.model.layers[0].self_attn.q_proj.weight.requires_grad_(True)

    reference = copy.deepcopy(model)
    original = type(model).forward.__wrapped__

    input_ids = torch.randint(0, config.vocab_size, (2, 11))
    labels = input_ids.clone()
    labels[:, :3] = -100

    ref_out = original(reference, input_ids=input_ids, labels=labels)
    assert ref_out.logits is not None
    ref_out.loss.backward()

    out = model(input_ids=input_ids, labels=labels)
    assert out.logits is None
    out.loss.backward()

    torch.testing.assert_close(
        out.loss.detach(), ref_out.loss.detach(), rtol=1e-4, atol=1e-5
    )
    torch.testing.assert_close(
        model.model.layers[0].self_attn.q_proj.weight.grad,
        reference.model.layers[0].self_attn.q_proj.weight.grad,
        rtol=1e-3,
        atol=1e-4,
    )
    step = chunked._take_step()
    assert step is not None
    assert step.token_loss.shape == labels.shape


# --------------------------------------------------------------------------- #
# Stage A2: the Triton cross-entropy
# --------------------------------------------------------------------------- #

cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="the Triton cross-entropy needs a GPU"
)


def _count_kernel_launches(monkeypatch):
    """Count launches the way the harness does, by kernel name."""
    from triton.runtime.jit import JITFunction

    counts = {}
    original = JITFunction.run

    def run(jit, *args, **kwargs):
        name = getattr(getattr(jit, "fn", None), "__name__", "?")
        counts[name] = counts.get(name, 0) + 1
        return original(jit, *args, **kwargs)

    monkeypatch.setattr(JITFunction, "run", run)
    return counts


@cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("vocab", [4113, 8192])
def test_triton_ce_kernel_matches_pytorch(dtype, vocab):
    """The two kernels against ``F.cross_entropy`` and ``log_softmax``."""
    import solution.chunked_lm_head as chunked

    torch.manual_seed(0)
    rows = 9
    logits = (torch.randn(rows, vocab, device="cuda", dtype=dtype) * 2).detach()
    target = torch.randint(0, vocab, (rows,), device="cuda")
    target[2] = -100
    target[5] = -100

    ref_logits = logits.clone().float().requires_grad_(True)
    ref_total = F.cross_entropy(ref_logits, target, ignore_index=-100, reduction="sum")
    ref_total.backward()
    with torch.no_grad():
        log_probs = F.log_softmax(ref_logits, dim=-1)
        labelled = target != -100
        ref_entropy = -(log_probs.exp() * log_probs).sum(-1)
        ref_token_loss = (
            -log_probs.gather(-1, target.clamp(min=0).unsqueeze(-1)).squeeze(-1)
        ) * labelled
        ref_correct = ((ref_logits.argmax(-1) == target) & labelled).sum()

    work = logits.clone().requires_grad_(True)
    assert chunked._triton_ce_ready(work, target)
    total, entropy, token_loss, correct = chunked._triton_ce_function().apply(
        work, target
    )
    total.backward()

    if dtype is torch.float32:
        rtol, atol = 1e-5, 1e-5
    else:
        rtol, atol = 1e-2, 1e-4
    torch.testing.assert_close(total.detach(), ref_total.detach(), rtol=rtol, atol=atol)
    torch.testing.assert_close(entropy, ref_entropy, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(token_loss, ref_token_loss, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(
        work.grad.float(), ref_logits.grad.to(dtype).float(), rtol=rtol, atol=atol
    )
    assert correct.sum().item() == ref_correct.item()
    # The ignored rows carry no loss and therefore no gradient.
    assert bool((token_loss[~labelled] == 0).all())
    assert bool((work.grad[~labelled] == 0).all())


@cuda
def test_triton_ready_rejects_incompatible_tensors():
    import solution.chunked_lm_head as chunked

    logits = torch.zeros(4, 8, device="cuda")
    target = torch.zeros(4, dtype=torch.long, device="cuda")
    assert chunked._triton_ce_ready(logits, target)
    assert not chunked._triton_ce_ready(logits.cpu(), target.cpu())
    assert not chunked._triton_ce_ready(logits.t().contiguous().t(), target)
    assert not chunked._triton_ce_ready(logits[:, ::2], target)
    assert not chunked._triton_ce_ready(logits.to(torch.float64), target)
    assert not chunked._triton_ce_ready(logits, target.to(torch.int16))
    assert not chunked._triton_ce_ready(logits, target[:3])


@cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("chunk", [1, 4])
def test_chunked_loss_uses_triton_and_matches_full_logits(dtype, chunk, monkeypatch):
    """The whole chunked loss on the GPU against the unchunked reference."""
    import solution.chunked_lm_head as chunked

    counts = _count_kernel_launches(monkeypatch)
    head, hidden, labels = _case((2, 9, 8), 4113, device="cuda")
    head = head.to(dtype)
    hidden = hidden.detach().to(dtype).requires_grad_(True)

    ref_loss, ref_grad, ref_entropy, ref_tl, ref_correct, ref_labelled = _reference(
        hidden, head, labels
    )

    loss, step = chunked_causal_lm_loss(hidden, head, labels, chunk)
    loss.backward()

    assert counts.get("ce_forward_kernel", 0) > 0
    assert counts.get("ce_backward_kernel", 0) > 0
    if dtype is torch.float32:
        rtol, atol = 1e-5, 1e-5
        term_tol = (1e-4, 1e-4)
    else:
        # The head is a different GEMM shape per chunk, so the fp16 logits
        # themselves differ from the unchunked ones in the last bits.
        rtol, atol = 1e-2, 1e-4
        term_tol = (1e-3, 1e-3)
    torch.testing.assert_close(loss.detach(), ref_loss, rtol=rtol, atol=atol)
    torch.testing.assert_close(hidden.grad.float(), ref_grad.float(), rtol=rtol, atol=atol)
    torch.testing.assert_close(
        step.token_entropy, ref_entropy, rtol=term_tol[0], atol=term_tol[1]
    )
    torch.testing.assert_close(
        step.token_loss, ref_tl, rtol=term_tol[0], atol=term_tol[1]
    )
    assert step.correct.item() == ref_correct.item()
    assert step.labelled.item() == ref_labelled.item()
    assert head.weight.grad is None


@cuda
def test_triton_can_be_disabled(monkeypatch):
    """The flag sends the chunk back to the PyTorch path, with the same numbers."""
    import solution.chunked_lm_head as chunked

    monkeypatch.setattr(chunked, "_TRITON_ENABLED", False)
    counts = _count_kernel_launches(monkeypatch)

    head, hidden, labels = _case((2, 9, 8), 4113, device="cuda")
    ref_loss, ref_grad, *_ = _reference(hidden, head, labels)
    loss, _ = chunked_causal_lm_loss(hidden, head, labels, 4)
    loss.backward()

    assert "ce_forward_kernel" not in counts
    torch.testing.assert_close(loss.detach(), ref_loss, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(hidden.grad, ref_grad, rtol=1e-4, atol=1e-5)


@cuda
def test_patched_forward_on_cuda_launches_the_kernels(monkeypatch):
    """The real patched forward on a GPU: Triton kernels run and match the original."""
    import copy

    import solution.chunked_lm_head as chunked

    patch_chunked_lm_head()
    counts = _count_kernel_launches(monkeypatch)

    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(0)
    config = Qwen2Config(
        vocab_size=4200,  # more than one Triton block
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
    )
    model = Qwen2ForCausalLM(config).train().cuda()
    model.lm_head.weight.requires_grad_(False)
    model.model.layers[0].self_attn.q_proj.weight.requires_grad_(True)

    reference = copy.deepcopy(model)
    original = type(model).forward.__wrapped__

    input_ids = torch.randint(0, config.vocab_size, (2, 11), device="cuda")
    labels = input_ids.clone()
    labels[:, :3] = -100

    ref_out = original(reference, input_ids=input_ids, labels=labels)
    ref_out.loss.backward()

    out = model(input_ids=input_ids, labels=labels)
    assert out.logits is None
    out.loss.backward()

    assert counts.get("ce_forward_kernel", 0) > 0
    assert counts.get("ce_backward_kernel", 0) > 0
    torch.testing.assert_close(out.loss.detach(), ref_out.loss.detach(), rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(
        model.model.layers[0].self_attn.q_proj.weight.grad,
        reference.model.layers[0].self_attn.q_proj.weight.grad,
        rtol=1e-3,
        atol=1e-4,
    )
    step = chunked._take_step()
    assert step is not None
    assert step.token_loss.shape == labels.shape
    # Ignored positions stay at exactly 0.0 for Soup's rewind recorder.
    shift = shift_for_next_token(labels)
    assert bool((step.token_loss[shift == -100] == 0).all())
