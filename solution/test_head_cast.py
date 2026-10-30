import pytest
import torch
from solution.head_cast import project_head
from solution.test_chunked_lm_head import _count_kernel_launches

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required; includes T4")
@pytest.mark.parametrize("strided", [False, True])
def test_amp_cast_and_gradient(monkeypatch, strided):
    torch.manual_seed(64)
    head = torch.nn.Linear(64, 257, bias=False, device="cuda").requires_grad_(False)
    base = torch.randn(4, 17, 64, device="cuda")
    x = (base[:, 2:13] if strided else base).detach().requires_grad_(True)
    reference = x.detach().clone().requires_grad_(True)
    dy = torch.randn((*x.shape[:-1], 257), device="cuda", dtype=torch.float16)
    with torch.autocast("cuda", dtype=torch.float16):
        expected = head(reference)
    expected.backward(dy)
    counts = _count_kernel_launches(monkeypatch)
    with torch.autocast("cuda", dtype=torch.float16):
        actual = project_head(head, x)
    actual.backward(dy)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(x.grad, reference.grad, rtol=0, atol=0)
    assert counts.get("head_cast_kernel", 0) >= 3


def test_preserves_module_hooks():
    head = torch.nn.Linear(8, 9, bias=False).requires_grad_(False)
    x = torch.randn(2, 8, requires_grad=True)
    handle = head.register_forward_hook(lambda module, args, out: out + 1)
    try:
        torch.testing.assert_close(project_head(head, x), head(x))
    finally:
        handle.remove()
