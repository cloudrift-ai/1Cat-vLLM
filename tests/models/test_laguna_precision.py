# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.model_executor.models.laguna import (
    LagunaRMSNorm,
    _laguna_output_dtype,
)


def test_laguna_half_precision_uses_float_output_contract():
    assert _laguna_output_dtype(torch.float16) == torch.float32
    assert _laguna_output_dtype(torch.bfloat16) is None
    assert _laguna_output_dtype(torch.float32) is None


def test_laguna_norm_preserves_large_float_residual():
    norm = LagunaRMSNorm(8, eps=1e-6)
    hidden_states = torch.full((1, 8), 40000.0, dtype=torch.float16)
    residual = torch.full((1, 8), 40000.0, dtype=torch.float32)

    output, residual_output = norm(hidden_states, residual)

    assert output.dtype == torch.float16
    assert residual_output.dtype == torch.float32
    assert torch.isfinite(output).all()
    assert torch.isfinite(residual_output).all()
    torch.testing.assert_close(
        residual_output, torch.full_like(residual_output, 80000.0)
    )


def test_laguna_norm_without_residual_returns_normalized_tensor_only():
    norm = LagunaRMSNorm(8, eps=1e-6)
    output = norm(torch.ones((1, 8), dtype=torch.float16))

    assert isinstance(output, torch.Tensor)
    assert output.dtype == torch.float16
