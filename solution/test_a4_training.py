"""Two real tiny Qwen decoder training steps; no downloaded model or task data."""
import pytest
import torch

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required; includes T4")
def test_qwen_nf4_lora_step(monkeypatch):
    import bitsandbytes as bnb
    from bitsandbytes.autograd._functions import MatMul4Bit
    from transformers import Qwen2Config, Qwen2ForCausalLM
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    import solution
    import solution.chunked_lm_head as chunked
    from solution.test_chunked_lm_head import _count_kernel_launches
    torch.manual_seed(123)
    config = Qwen2Config(vocab_size=257, hidden_size=64, intermediate_size=128,
                         num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                         attention_dropout=0.0, use_cache=False)
    model = Qwen2ForCausalLM(config)
    for layer in model.model.layers:
        for parent in (layer.self_attn, layer.mlp):
            for name, module in list(parent.named_children()):
                if type(module) is torch.nn.Linear:
                    quant = bnb.nn.Linear4bit(module.in_features, module.out_features,
                                             bias=module.bias is not None, compute_dtype=torch.float16,
                                             compress_statistics=True, quant_type="nf4")
                    quant.weight = bnb.nn.Params4bit(module.weight.detach(), requires_grad=False,
                                                    quant_type="nf4", compress_statistics=True)
                    if module.bias is not None:
                        quant.bias = torch.nn.Parameter(module.bias.detach(), requires_grad=False)
                    setattr(parent, name, quant)
    model = model.cuda()
    model.is_loaded_in_4bit = True
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                           target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"))
    # A nonzero B checks gradients for both adapter matrices.
    for name, param in model.named_parameters():
        if "lora_B" in name:
            torch.nn.init.normal_(param, std=0.01)
    ids = torch.randint(0, 257, (4, 19), device="cuda")
    labels = ids.clone()
    labels[:, :5] = -100
    original_backward = MatMul4Bit.backward
    monkeypatch.setattr(chunked, "project_frozen_head", lambda head, x: head(x))
    chunked.patch_chunked_lm_head()
    def step():
        torch.manual_seed(99)  # Identical native dropout stream for the comparison.
        model.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16):
            loss = model(input_ids=ids, labels=labels).loss
        loss.backward()
        return loss.detach(), {name: p.grad.detach().clone() for name, p in model.named_parameters()
                               if p.requires_grad}
    expected_loss, expected_grads = step()
    try:
        solution.apply()
        counts = _count_kernel_launches(monkeypatch)
        loss, grads = step()
        torch.testing.assert_close(loss, expected_loss, rtol=0, atol=0)
        assert counts.get("nf4_dq_cast_kernel", 0) > 0
        assert counts.get("head_cast_kernel", 0) > 0
        assert counts.get("ce_forward_kernel", 0) > 0
        assert not counts.get("lm_head_forward_kernel", 0)
        for name in grads:
            torch.testing.assert_close(grads[name], expected_grads[name], rtol=0, atol=0)
    finally:
        MatMul4Bit.backward = original_backward
