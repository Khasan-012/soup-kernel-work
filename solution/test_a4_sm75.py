"""Offline SM75 compilation is compatibility evidence, not T4 execution."""
import pytest
import triton
import triton.language as tl
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource
from solution.nf4_backward import nf4_dq_cast_kernel
from solution.head_cast import head_cast_kernel

@pytest.mark.parametrize("nested", [False, True])
def test_nf4_sm75_codegen(nested):
    signature = {name: "*fp32" for name in ("S2", "CODE2", "OFFSET", "CODE4")}
    signature.update(Q="*u8", S="*u8" if nested else "*fp32", OUT="*fp16", N="i32")
    constants = dict(GROUP=64, GROUP2=256, NESTED=nested, STORAGE=tl.float16, BLOCK=1024)
    compiled = triton.compile(ASTSource(nf4_dq_cast_kernel, signature=signature, constexprs=constants),
                              target=GPUTarget("cuda", 75, 32),
                              options={"num_warps": 4, "enable_fp_fusion": False})
    assert ".target sm_75" in compiled.asm["ptx"]
    assert compiled.metadata.shared == 0


def test_head_cast_sm75_codegen():
    compiled = triton.compile(ASTSource(head_cast_kernel,
        signature=dict(X="*fp32", Y="*fp16", N="i32"),
        constexprs=dict(WIDTH=1536, SEQ=128, S0=786432, S1=1536, S2=1, BLOCK=1024)),
        target=GPUTarget("cuda", 75, 32), options={"num_warps": 4})
    assert ".target sm_75" in compiled.asm["ptx"]
    assert compiled.metadata.shared == 0
