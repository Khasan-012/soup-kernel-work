"""NF4 dequantization must retain native rounding before the compute cast."""
import pytest
import torch

from solution.nf4_backward import dequantize_nf4, patch_nf4_backward

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required; includes T4")

@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("storage", [torch.float16, torch.float32, torch.bfloat16])
@pytest.mark.parametrize("compute", [torch.float16, torch.float32])
@pytest.mark.parametrize("shape", [(17, 67), (1536, 1536)])
def test_dequantization(nested, storage, compute, shape):
    import bitsandbytes.functional as F
    torch.manual_seed(42)
    w = torch.randn(shape, device="cuda", dtype=storage)
    packed, state = F.quantize_4bit(w, quant_type="nf4", compress_statistics=nested)
    expected = F.dequantize_4bit(packed, state).to(compute)
    actual = dequantize_nf4(packed, state, compute)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

@pytest.mark.parametrize("shape", [(2, 7, 67), (4, 128, 1536)])
def test_backward_and_lora_gradients(monkeypatch, shape):
    import bitsandbytes as bnb
    import bitsandbytes.functional as F
    from bitsandbytes.autograd._functions import MatMul4Bit
    from solution.test_chunked_lm_head import _count_kernel_launches
    torch.manual_seed(53)
    width = shape[-1]
    w = torch.randn((97, width), device="cuda", dtype=torch.float16) * 0.1
    packed, state = F.quantize_4bit(w, quant_type="nf4", compress_statistics=True)
    x = torch.randn(shape, device="cuda", dtype=torch.float16, requires_grad=True)
    a = torch.randn((16, width), device="cuda", dtype=torch.float32, requires_grad=True)
    b = torch.randn((97, 16), device="cuda", dtype=torch.float32, requires_grad=True)
    bias = torch.randn(97, device="cuda", dtype=torch.float16, requires_grad=True)
    upstream = torch.randn((*shape[:-1], 97), device="cuda", dtype=torch.float16)
    original = MatMul4Bit.backward
    def run():
        y = bnb.matmul_4bit(x, packed, state, bias=bias)
        y = y + ((x.float() @ a.t()) @ b.t()).to(y.dtype) * 2
        return y, torch.autograd.grad(y, (x, a, b, bias), upstream)
    expected, expected_grads = run()
    patch_nf4_backward()
    counts = _count_kernel_launches(monkeypatch)
    try:
        actual, grads = run()
        assert counts.get("nf4_dq_cast_kernel", 0) > 0
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        for actual_grad, expected_grad in zip(grads, expected_grads):
            torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)
    finally:
        MatMul4Bit.backward = original


def test_cpu_fallback():
    import bitsandbytes.functional as F
    from solution.nf4_backward import supports
    w = torch.randn(32, 64)
    q, state = F.quantize_4bit(w, quant_type="nf4")
    assert not supports(q, state, torch.float32)
