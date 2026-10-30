"""The reported training share must exclude example and foreign kernels."""

import json

import pytest

from solution.a3_diagnostics import OWN_NAMES, summarize_training_metrics


def test_training_share_excludes_example_swiglu_even_if_it_dominates(tmp_path):
    names = {name: 1 for name in OWN_NAMES}
    names["example_swiglu_fwd_kernel"] = 100
    names["example_swiglu_bwd_kernel"] = 100
    files = {name: "solution/frozen_lm_head.py" for name in names}
    timings = {
        "ce_forward_kernel": 129.308, "ce_backward_kernel": 235.712,
        "lm_head_dhidden_partial_kernel": 1815.730,
        "example_swiglu_fwd_kernel": 10000.0,
        "example_swiglu_bwd_kernel": 20000.0,
    }
    path = tmp_path / "metrics.json"
    path.write_text(json.dumps({
        "timed_loop_s": 43.615, "own_kernel_ms": timings,
        "triton_launches": names, "triton_files": files,
    }), encoding="utf-8")
    result = summarize_training_metrics(path)
    assert result["own_ms"] == pytest.approx(2180.75)
    assert result["own_share_percent"] == pytest.approx(5.0)
    assert result["missing_own_launches"] == []
    assert not any("example" in name for name in result["own_kernel_ms_excluding_example"])


@pytest.mark.parametrize("bad_field", ["file", "launch"])
def test_unproven_kernel_is_not_counted(tmp_path, bad_field):
    path = tmp_path / "metrics.json"
    path.write_text(json.dumps({
        "timed_loop_s": 1.0,
        "own_kernel_ms": {"lm_head_forward_kernel": 100.0},
        "triton_launches": {"lm_head_forward_kernel": 0 if bad_field == "launch" else 1},
        "triton_files": {"lm_head_forward_kernel": "other/library.py" if bad_field == "file" else "solution/frozen_lm_head.py"},
    }), encoding="utf-8")
    result = summarize_training_metrics(path)
    assert result["own_ms"] == 0
    assert result["own_share_percent"] == 0
    assert "lm_head_forward_kernel" in result["missing_own_launches"]


def test_missing_loop_time_is_reported_as_unknown(tmp_path):
    path = tmp_path / "metrics.json"
    path.write_text("{}", encoding="utf-8")
    result = summarize_training_metrics(path)
    assert result["own_share_percent"] is None
    assert len(result["missing_own_launches"]) == 5
