"""A2 regressions, also runnable on CPU with TRITON_INTERPRET=1.

The reference always uses fp32 CE, as Transformers does. The interpreter is a
mathematical check only; CUDA tests and contest timing still need a real GPU.
"""

import os

import pytest
import torch
import torch.nn.functional as F

import solution.chunked_lm_head as chunked


INTERPRET = os.environ.get("TRITON_INTERPRET") == "1"
kernel = pytest.mark.skipif(
    not torch.cuda.is_available() and not INTERPRET,
    reason="needs CUDA or TRITON_INTERPRET=1",
)
DEVICE = "cpu" if INTERPRET else "cuda"


def _check_terms(logits, target, ignore_index=-100, scale=0.375):
    reference = logits.detach().float().requires_grad_(True)
    expected = F.cross_entropy(
        reference, target.long(), ignore_index=ignore_index, reduction="sum"
    )
    (expected * scale).backward()
    with torch.no_grad():
        log_probs = reference.log_softmax(-1)
        labelled = target != ignore_index
        expected_entropy = -(log_probs.exp() * log_probs).sum(-1)
        safe_target = torch.where(labelled, target, 0).long()
        expected_tokens = -log_probs.gather(1, safe_target[:, None]).squeeze(1)
        expected_tokens = torch.where(labelled, expected_tokens, 0.0)
        expected_correct = (reference.argmax(-1) == target) & labelled

    work = logits.detach().clone().requires_grad_(True)
    # Keep the existing two-argument call covered for the default index.
    args = (work, target) if ignore_index == -100 else (work, target, ignore_index)
    total, entropy, tokens, correct = chunked._triton_ce_function().apply(*args)
    (total * scale).backward()
    torch.testing.assert_close(total, expected, rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(entropy, expected_entropy, rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(tokens, expected_tokens, rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(
        work.grad.float(), reference.grad.to(logits.dtype).float(), rtol=2e-4, atol=2e-6
    )
    assert torch.equal(correct.bool(), expected_correct)
    assert torch.count_nonzero(work.grad[~labelled]) == 0
    assert not any(term.requires_grad for term in (entropy, tokens, correct))


@kernel
@pytest.mark.parametrize("offset", [-1_000_000.0, 1_000_000.0])
def test_ce_is_stable_under_large_common_logit_shift(offset):
    # Exact quarter-unit differences survive the large fp32 offset. The max
    # moves between tiles, exercising the online rescaling of entropy.
    indices = torch.arange(8209, device=DEVICE)
    values = (indices % 29).float() / 4 + (indices // 4096).float() * 2
    logits = torch.stack((values, values.flip(0))) + offset
    target = torch.tensor([17, -100], device=DEVICE)
    _check_terms(logits, target)


@kernel
@pytest.mark.parametrize("ignore_index", [-7, 3])
def test_ce_respects_custom_ignore_index(ignore_index):
    logits = torch.tensor([[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]], device=DEVICE)
    target = torch.tensor([ignore_index, 0], device=DEVICE)
    _check_terms(logits, target, ignore_index=ignore_index)


@kernel
@pytest.mark.parametrize("vocab", [1, 127, 4097, 151936])
def test_ce_handles_vocab_tails_ties_and_ignored_rows(vocab):
    logits = torch.zeros(3, vocab, device=DEVICE)
    # Ties must select the lowest index, including across tile boundaries.
    target = torch.tensor([0, vocab - 1, -100], device=DEVICE)
    _check_terms(logits, target, scale=-0.5)


@kernel
def test_ce_all_ignored_has_zero_loss_and_gradient():
    torch.manual_seed(41)
    logits = torch.randn(3, 73, device=DEVICE)
    target = torch.full((3,), -100, device=DEVICE)
    _check_terms(logits, target)


@kernel
def test_ce_extreme_finite_logits_do_not_overflow():
    logits = torch.tensor([[1000.0, -1000.0, 0.0], [-1000.0, 1000.0, 999.0]], device=DEVICE)
    target = torch.tensor([1, 2], device=DEVICE)
    _check_terms(logits, target)


@kernel
def test_triton_token_losses_match_soup_rewind_rows():
    from soup_cli.trainer.rewind_hf import row_losses_from_token_losses

    torch.manual_seed(31)
    logits = torch.randn(2, 5, 17, device=DEVICE)
    labels = torch.tensor(
        [[3, -100, 4, 7, 0], [-100, -100, -100, -100, -100]], device=DEVICE
    )
    shifted = F.pad(labels, (0, 1), value=-100)[..., 1:]
    _, _, tokens, _ = chunked._triton_ce_function().apply(
        logits.reshape(-1, 17), shifted.reshape(-1)
    )
    means, counts = row_losses_from_token_losses(tokens.reshape(2, 5), labels)
    reference = F.cross_entropy(
        logits[0, :-1].float(), labels[0, 1:], ignore_index=-100
    )
    assert counts == [3, 0]
    assert means == pytest.approx([reference.item(), 0.0], rel=2e-5, abs=2e-5)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=kernel)])
def test_custom_index_reaches_chunk_dispatch_and_denominator(device):
    if INTERPRET and device == "cuda":
        pytest.skip("dispatch requires real CUDA")
    torch.manual_seed(7)
    head = torch.nn.Linear(4, 11, bias=False, device=device).requires_grad_(False)
    hidden = torch.randn(2, 5, 4, device=device, requires_grad=True)
    labels = torch.tensor([[2, -7, 1, 4, 6], [3, 0, 1, -7, 8]], device=device)
    reference = hidden.detach().clone().requires_grad_(True)
    shifted = F.pad(labels, (0, 1), value=-7)[..., 1:]
    expected = F.cross_entropy(
        head(reference).float().reshape(-1, 11), shifted.reshape(-1),
        ignore_index=-7, reduction="sum",
    ) / 19
    (expected * 0.25).backward()
    loss, step = chunked.chunked_causal_lm_loss(
        hidden, head, labels, chunk=2, ignore_index=-7, num_items_in_batch=19
    )
    (loss * 0.25).backward()
    torch.testing.assert_close(loss, expected, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(hidden.grad, reference.grad, rtol=1e-4, atol=1e-6)
    assert torch.count_nonzero(step.token_loss[shifted == -7]) == 0


@pytest.mark.skipif(not torch.cuda.is_available() or INTERPRET, reason="needs real CUDA")
def test_ready_rejects_targets_on_another_device():
    logits = torch.zeros(2, 3, device="cuda")
    target = torch.zeros(2, dtype=torch.long, device="cpu")
    assert not chunked._triton_ce_ready(logits, target)


@pytest.mark.skipif(not torch.cuda.is_available() or INTERPRET, reason="needs real CUDA")
def test_ready_rejects_empty_rows():
    logits = torch.zeros(0, 3, device="cuda")
    target = torch.zeros(0, dtype=torch.long, device="cuda")
    assert not chunked._triton_ce_ready(logits, target)
