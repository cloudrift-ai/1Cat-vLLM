# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import torch

from vllm.model_executor.layers.fused_moe import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    MoEActivation,
    RoutingMethodType,
)
from vllm.model_executor.layers.fused_moe import modular_kernel as mk
from vllm.model_executor.layers.fused_moe.experts import triton_moe


def _moe_config(*, out_dtype: torch.dtype | None = None) -> FusedMoEConfig:
    return FusedMoEConfig(
        num_experts=8,
        experts_per_token=2,
        hidden_dim=16,
        intermediate_size_per_partition=8,
        num_local_experts=8,
        num_logical_experts=8,
        activation=MoEActivation.SILU,
        device=torch.device("cpu"),
        routing_method=RoutingMethodType.Default,
        moe_parallel_config=FusedMoEParallelConfig(
            tp_size=1,
            pcp_size=1,
            dp_size=1,
            ep_size=1,
            tp_rank=0,
            pcp_rank=0,
            dp_rank=0,
            ep_rank=0,
            sp_size=1,
            use_ep=False,
            all2all_backend="allgather_reducescatter",
            enable_eplb=False,
        ),
        in_dtype=torch.float16,
        out_dtype=out_dtype,
    )


def test_fused_moe_output_dtype_defaults_to_input_dtype():
    assert _moe_config().out_dtype == torch.float16
    assert _moe_config(out_dtype=torch.float32).out_dtype == torch.float32


def test_modular_moe_keeps_intermediates_half_with_float_output(monkeypatch):
    class FakeExperts:
        moe_config = SimpleNamespace(in_dtype=torch.float16, out_dtype=torch.float32)

        @staticmethod
        def workspace_dtype(dtype):
            assert dtype == torch.float16
            return dtype

        @staticmethod
        def workspace_shapes(*args, **kwargs):
            return (2, 2, 4), (2, 2, 4), (2, 4)

    class FakeWorkspace:
        @staticmethod
        def get_simultaneous(*specs):
            return tuple(torch.empty(shape, dtype=dtype) for shape, dtype in specs)

    monkeypatch.setattr(mk, "current_workspace_manager", lambda: FakeWorkspace())
    impl = object.__new__(mk.FusedMoEKernelModularImpl)
    impl.fused_experts = FakeExperts()

    workspace13, workspace2, output = impl._allocate_buffers(
        torch.float16,
        torch.float32,
        torch.device("cpu"),
        2,
        2,
        8,
        4,
        2,
        8,
        8,
        None,
        MoEActivation.SILU,
    )

    assert workspace13.dtype == torch.float16
    assert workspace2.dtype == torch.float16
    assert output.dtype == torch.float32


def test_triton_moe_accumulates_half_expert_rows_into_float_output(monkeypatch):
    experts = object.__new__(triton_moe.TritonExperts)
    expert_rows = torch.full((1, 10, 4), 40000.0, dtype=torch.float16)
    output = torch.empty((1, 4), dtype=torch.float32)

    monkeypatch.setattr(
        triton_moe.ops,
        "moe_sum",
        lambda *args: (_ for _ in ()).throw(AssertionError("half-only reducer used")),
    )
    experts.moe_sum(expert_rows, output)

    torch.testing.assert_close(output, torch.full_like(output, 400000.0))


def test_triton_moe_retains_existing_same_dtype_reducer(monkeypatch):
    experts = object.__new__(triton_moe.TritonExperts)
    expert_rows = torch.ones((1, 2, 4), dtype=torch.float16)
    output = torch.empty((1, 4), dtype=torch.float16)
    calls = []

    def fake_moe_sum(input_tensor, output_tensor):
        calls.append((input_tensor, output_tensor))

    monkeypatch.setattr(triton_moe.ops, "moe_sum", fake_moe_sum)
    experts.moe_sum(expert_rows, output)

    assert calls == [(expert_rows, output)]
