# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import torch

from vllm.model_executor.layers import linear


def _row_parallel_layer(output_dtype: torch.dtype | None):
    layer = object.__new__(linear.RowParallelLinear)
    torch.nn.Module.__init__(layer)
    layer.input_is_parallel = True
    layer.tp_rank = 0
    layer.tp_size = 2
    layer.skip_bias_add = True
    layer.bias = None
    layer.reduce_results = True
    layer.reduce_output_dtype = output_dtype
    layer.return_bias = False
    layer.quant_method = SimpleNamespace(
        apply=lambda layer, input_tensor, bias: torch.full(
            (input_tensor.shape[0], 4), 40000.0, dtype=torch.float16
        )
    )
    return layer


def test_row_parallel_casts_before_tensor_parallel_reduction(monkeypatch):
    layer = _row_parallel_layer(torch.float32)
    reducer_inputs = []

    monkeypatch.setattr(linear, "_maybe_sm70_dense_forward", lambda *args: None)
    monkeypatch.setattr(
        linear,
        "_maybe_sm70_awq_mlp_down_tile_gemm_reduce",
        lambda *args: (_ for _ in ()).throw(AssertionError("fused reducer used")),
    )
    monkeypatch.setattr(
        linear,
        "_maybe_sm70_awq_mlp_down_tile_all_reduce",
        lambda *args: (_ for _ in ()).throw(AssertionError("fused reducer used")),
    )

    def fake_all_reduce(tensor):
        reducer_inputs.append(tensor)
        return tensor * 2

    monkeypatch.setattr(linear, "tensor_model_parallel_all_reduce", fake_all_reduce)
    output = layer(torch.ones((1, 4), dtype=torch.float16))

    assert reducer_inputs[0].dtype == torch.float32
    assert output.dtype == torch.float32
    torch.testing.assert_close(output, torch.full_like(output, 80000.0))


def test_row_parallel_default_retains_existing_fused_reducer(monkeypatch):
    layer = _row_parallel_layer(None)
    expected = torch.ones((1, 4), dtype=torch.float16)
    monkeypatch.setattr(
        linear,
        "_maybe_sm70_awq_mlp_down_tile_gemm_reduce",
        lambda *args: expected,
    )

    output = layer(torch.ones((1, 4), dtype=torch.float16))

    assert output is expected
