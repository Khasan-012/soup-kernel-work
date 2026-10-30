"""Run explicitly on Kaggle T4 / Triton 3.6.0; environment mismatch is an error."""
import pytest
import torch
from solution.a3_sm75_validation import (
    require_target, fp32_reference_mode, mma_matmul, experimental_projection,
    watch_mma_launches, assert_fp16_close, _launch,
)


@pytest.fixture(scope="module", autouse=True)
def target():
    require_target()


@pytest.mark.parametrize("shape", [(1,1,1), (7,5,3), (15,7,7), (16,8,8),
    (17,9,9), (31,15,17), (32,16,16), (33,17,31), (19,65,13), (65,1536,257)])
@pytest.mark.parametrize("transposed", [True, False])
@pytest.mark.parametrize("out_dtype", [torch.float16, torch.float32])
def test_matrix_tails_and_b_storage_match_fp32(shape, transposed, out_dtype):
    m, k, n = shape
    torch.manual_seed(1000 + m + k + n)
    a = torch.randn(m, k, device="cuda:0", dtype=torch.float16) * 0.125
    logical_b = torch.randn(k, n, device="cuda:0", dtype=torch.float16) * 0.125
    b = logical_b.t().contiguous() if transposed else logical_b.contiguous()
    with fp32_reference_mode():
        expected = a.float() @ logical_b.float()
    result = mma_matmul(a, b, transposed, out_dtype)
    if out_dtype == torch.float16:
        assert_fp16_close(result, expected)
    else:
        assert torch.isfinite(result).all()
        torch.testing.assert_close(result, expected, rtol=2e-4, atol=2e-5)


@pytest.mark.parametrize("transposed", [True, False])
@pytest.mark.parametrize("pattern", ["retain_units", "exceed_half_range", "cancel", "zero"])
def test_fp32_accumulator_stability(transposed, pattern):
    k = 4096 if pattern == "retain_units" else 24
    a = torch.ones(17, k, device="cuda:0", dtype=torch.float16)
    b = torch.ones(k, 9, device="cuda:0", dtype=torch.float16)
    if pattern == "retain_units":
        b[0] = 4096  # 4096 + 4095 units must remain exactly 8191 in FP32.
    elif pattern == "exceed_half_range":
        a.fill_(256)
        b.fill_(256)  # Each product exceeds FP16 range; FP32 result stays finite.
    elif pattern == "cancel":
        a.fill_(256)
        b.fill_(256)
        b[1::2] = -256
        b[-1] = -255  # Large cancelling terms leave an exactly representable residual.
    else:
        a.zero_()
    source = b.t().contiguous() if transposed else b
    with fp32_reference_mode():
        expected = a.float() @ b.float()
    actual = mma_matmul(a, source, transposed, torch.float32)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("shape", [(1,1,1), (17,9,13), (33,65,31), (19,1536,257)])
@pytest.mark.parametrize("amp", [False, True])
def test_hidden_gradient_and_amp_rounding_against_fp32(shape, amp):
    m, h, v = shape
    torch.manual_seed(42)
    dtype = torch.float32 if amp else torch.float16
    x = (torch.randn(m, h, device="cuda:0", dtype=dtype)*0.125).requires_grad_()
    weight = torch.randn(v, h, device="cuda:0", dtype=dtype)*0.125
    dy = torch.randn(m, v, device="cuda:0", dtype=torch.float16)*0.01
    reference_x = x.detach().half().float().requires_grad_()
    with fp32_reference_mode():
        reference = reference_x @ weight.half().float().t()
        reference.backward(dy.float())
    with watch_mma_launches() as launches:
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            actual = experimental_projection(x, weight)
        actual.backward(dy)
        torch.cuda.synchronize()
    assert len(launches) == 2
    assert [r["transposed_storage"] for r in launches] == [True, False]
    assert_fp16_close(actual, reference.detach())
    assert_fp16_close(x.grad, reference_x.grad)
    assert actual.dtype == torch.float16
    assert x.grad.dtype == dtype
    assert weight.grad is None


@pytest.mark.parametrize("shape", [(17,9,13), (31,17,9)])
@pytest.mark.parametrize("transposed", [True, False])
def test_tail_stores_preserve_output_canaries(shape, transposed):
    m, k, n = shape
    torch.manual_seed(81)
    a = torch.randn(m, k, device="cuda:0", dtype=torch.float16)*0.125
    b = torch.randn(k, n, device="cuda:0", dtype=torch.float16)*0.125
    storage = b.t().contiguous() if transposed else b
    backing = torch.full((m*n+64,), 12345.0, device="cuda:0", dtype=torch.float32)
    out = backing[32:32+m*n].view(m, n)
    _launch(a, storage, out, transposed)
    with fp32_reference_mode():
        expected = a.float() @ b.float()
    torch.testing.assert_close(out, expected, rtol=2e-4, atol=2e-5)
    assert torch.all(backing[:32] == 12345.0)
    assert torch.all(backing[32+m*n:] == 12345.0)
