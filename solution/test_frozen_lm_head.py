"""A3 correctness and real-launch tests for the frozen head, without training."""

import copy

import pytest
import torch
import torch.nn.functional as F

import solution.chunked_lm_head as chunked
from solution.test_chunked_lm_head import _count_kernel_launches


cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
GEMMS = {
    "lm_head_forward_kernel",
    "lm_head_dhidden_partial_kernel",
    "lm_head_dhidden_reduce_kernel",
}


@cuda
@pytest.mark.parametrize("amp", [False, True])
@pytest.mark.parametrize("chunk", [1, 4])
@pytest.mark.parametrize("vocab", [257, 4109])
def test_chunked_training_uses_own_gemms_and_preserves_ce_rewind(monkeypatch, amp, chunk, vocab):
    from soup_cli.trainer.rewind_hf import row_losses_from_token_losses

    torch.manual_seed(81)
    dtype = torch.float32 if amp else torch.float16
    head = torch.nn.Linear(64, vocab, bias=False, device="cuda", dtype=dtype)
    head.requires_grad_(False)
    hidden = (torch.randn(2, 7, 64, device="cuda", dtype=dtype) * 0.2).requires_grad_(True)
    labels = torch.randint(0, vocab, (2, 7), device="cuda")
    labels[0, :3] = -100
    labels[1] = -100
    shifted = F.pad(labels, (0, 1), value=-100)[..., 1:]
    reference = hidden.detach().clone().requires_grad_(True)
    with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
        logits = head(reference)
        log_probs = logits.detach().float().log_softmax(-1)
        expected = F.cross_entropy(
            logits.float().reshape(-1, vocab), shifted.reshape(-1), reduction="sum"
        ) / 23
    (expected * 0.375).backward()
    counts = _count_kernel_launches(monkeypatch)
    with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
        loss, step = chunked.chunked_causal_lm_loss(
            hidden, head, labels, chunk=chunk, num_items_in_batch=torch.tensor(23, device="cuda")
        )
    (loss * 0.375).backward()

    assert GEMMS.issubset({name for name, count in counts.items() if count > 0})
    assert counts.get("ce_forward_kernel", 0) > 0
    assert counts.get("ce_backward_kernel", 0) > 0
    torch.testing.assert_close(loss, expected, rtol=5e-4, atol=2e-5)
    torch.testing.assert_close(hidden.grad.float(), reference.grad.float(), rtol=8e-3, atol=3e-5)
    expected_tokens = F.cross_entropy(
        logits.detach().float().reshape(-1, vocab), shifted.reshape(-1), reduction="none"
    ).reshape(2, 7)
    torch.testing.assert_close(step.token_loss, expected_tokens, rtol=5e-4, atol=5e-4)
    torch.testing.assert_close(
        step.token_entropy, -(log_probs.exp() * log_probs).sum(-1), rtol=5e-4, atol=5e-4
    )
    assert step.correct == ((logits.argmax(-1) == shifted) & (shifted != -100)).sum()
    means, tokens = row_losses_from_token_losses(step.token_loss, labels)
    assert tokens == [4, 0]
    assert means == pytest.approx([expected_tokens[0].sum().item() / 4, 0.0], rel=5e-4)
    assert torch.count_nonzero(hidden.grad[1]) == 0
    assert head.weight.grad is None


@cuda
def test_real_qwen_forward_amp_launches_a3_and_preserves_adapter_gradient(monkeypatch):
    from transformers import Qwen2Config, Qwen2ForCausalLM
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

    import solution
    solution.apply()
    torch.manual_seed(42)
    model = Qwen2ForCausalLM(Qwen2Config(
        vocab_size=257, hidden_size=64, intermediate_size=128,
        num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=32,
    )).cuda().train()
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
    model = get_peft_model(model, LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"]
    ))
    reference = copy.deepcopy(model)
    ids = torch.randint(0, 257, (2, 9), device="cuda")
    labels = ids.clone()
    labels[:, :3] = -100
    # The PEFT wrapper stays active. Temporarily disable only A1's model patch
    # for the reference, then restore it before the A3 forward.
    monkeypatch.setattr(chunked, "_ENABLED", False)
    with torch.autocast("cuda", dtype=torch.float16):
        ref_out = reference(input_ids=ids, labels=labels)
    ref_out.loss.backward()
    monkeypatch.setattr(chunked, "_ENABLED", True)
    counts = _count_kernel_launches(monkeypatch)
    with torch.autocast("cuda", dtype=torch.float16):
        out = model(input_ids=ids, labels=labels)
    out.loss.backward()
    assert out.logits is None
    assert GEMMS.issubset({name for name, count in counts.items() if count > 0})
    torch.testing.assert_close(out.loss, ref_out.loss, rtol=5e-4, atol=1e-4)
    torch.testing.assert_close(
        model.get_base_model().model.layers[0].self_attn.q_proj.lora_B["default"].weight.grad,
        reference.get_base_model().model.layers[0].self_attn.q_proj.lora_B["default"].weight.grad,
        rtol=0.02, atol=1e-5,
    )
    assert chunked._take_step().token_loss.shape == labels.shape


@cuda
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape", [(3, 37, 131), (5, 96, 4109)])
def test_projection_logits_and_arbitrary_upstream_gradient(monkeypatch, dtype, shape):
    from solution.frozen_lm_head import project_frozen_head

    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("GPU has no bf16 support")
    rows, features, vocab = shape
    torch.manual_seed(55)
    head = torch.nn.Linear(features, vocab, bias=False, device="cuda", dtype=dtype)
    head.requires_grad_(False)
    x = torch.randn(rows, features * 2, device="cuda", dtype=dtype)[:, ::2]
    x = x.detach().requires_grad_(True)
    reference = x.detach().clone().requires_grad_(True)
    upstream = torch.randn(rows, vocab, device="cuda", dtype=dtype) * 0.01
    expected = head(reference)
    expected.backward(upstream)
    counts = _count_kernel_launches(monkeypatch)
    result = project_frozen_head(head, x)
    result.backward(upstream)
    tol = 0.003 if dtype == torch.float16 else 0.02
    atol = 5e-4 if dtype == torch.float16 else 0.008
    torch.testing.assert_close(result, expected, rtol=tol, atol=atol)
    torch.testing.assert_close(x.grad, reference.grad, rtol=tol, atol=atol)
    assert result.dtype == dtype
    assert GEMMS.issubset({name for name, count in counts.items() if count > 0})


@cuda
@pytest.mark.parametrize("case", ["fp32", "bias", "trainable", "hook", "backward_hook", "backward_pre_hook", "override", "disabled"])
def test_unsupported_projection_preserves_head_semantics(monkeypatch, case):
    import solution.frozen_lm_head as frozen

    head = torch.nn.Linear(16, 31, bias=(case == "bias"), device="cuda")
    head.requires_grad_(case == "trainable")
    if case == "hook":
        head.register_forward_hook(lambda module, inputs, output: output + 0.5)
    if case == "backward_hook":
        head.register_full_backward_hook(lambda module, grad_input, grad_output: (grad_input[0] * 0.5,))
    if case == "backward_pre_hook":
        head.register_full_backward_pre_hook(lambda module, grad_output: (grad_output[0] * 0.5,))
    if case == "override":
        original = head.forward
        head.forward = lambda value: original(value) + 0.75
    if case == "disabled":
        monkeypatch.setattr(frozen, "ENABLED", False)
    x = torch.randn(2, 16, device="cuda", requires_grad=True)
    reference = x.detach().clone().requires_grad_(True)
    amp = case != "fp32"
    with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
        expected = head(reference)
    expected.sum().backward()
    expected_weight_grad = None if head.weight.grad is None else head.weight.grad.clone()
    head.zero_grad(set_to_none=True)
    counts = _count_kernel_launches(monkeypatch)
    with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
        result = frozen.project_frozen_head(head, x)
    result.sum().backward()
    torch.testing.assert_close(result, expected, rtol=0, atol=0)
    torch.testing.assert_close(x.grad, reference.grad, rtol=0, atol=0)
    if expected_weight_grad is not None:
        torch.testing.assert_close(head.weight.grad, expected_weight_grad, rtol=0, atol=0)
    assert not GEMMS.intersection(counts)


def test_cpu_projection_falls_back_without_losing_gradients():
    from solution.frozen_lm_head import project_frozen_head

    head = torch.nn.Linear(3, 2, bias=False).requires_grad_(False)
    x = torch.randn(2, 3, requires_grad=True)
    result = project_frozen_head(head, x)
    result.sum().backward()
    torch.testing.assert_close(result, F.linear(x, head.weight), rtol=0, atol=0)
    torch.testing.assert_close(x.grad, head.weight.sum(0).expand_as(x))


@cuda
def test_unsupported_amp_head_fallback_is_reported_once(monkeypatch):
    import solution.frozen_lm_head as frozen

    monkeypatch.setattr(frozen, "_WARNED_FALLBACKS", set(), raising=False)
    head = torch.nn.Linear(16, 31, bias=False, device="cuda").requires_grad_(False)
    head.register_forward_hook(lambda module, inputs, output: output + 0.5)
    hidden = torch.randn(2, 16, device="cuda")
    with torch.autocast("cuda", dtype=torch.float16):
        with pytest.warns(RuntimeWarning, match="A3 frozen head fallback") as records:
            first = frozen.project_frozen_head(head, hidden)
            second = frozen.project_frozen_head(head, hidden)
        expected = head(hidden)
    assert len(records) == 1
    torch.testing.assert_close(first, expected, rtol=0, atol=0)
    torch.testing.assert_close(second, expected, rtol=0, atol=0)
