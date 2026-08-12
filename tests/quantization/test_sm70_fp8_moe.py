# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import struct

import pytest
import torch

from vllm.model_executor.layers.quantization.fp8_sm70_moe import (
    _native_output_dtype_supported,
)
from vllm.platforms import current_platform


def test_sm70_fp8_native_output_kernels_require_matching_dtype():
    expert_output = torch.empty((1, 16), dtype=torch.float16, device="meta")

    assert _native_output_dtype_supported(
        torch.empty_like(expert_output), expert_output
    )
    assert not _native_output_dtype_supported(
        torch.empty((1, 16), dtype=torch.float32, device="meta"), expert_output
    )


def _read_strided_ptr_row(table: torch.Tensor, expert: int) -> tuple[int, int]:
    row = bytes(table[expert * 16 : (expert + 1) * 16].cpu().tolist())
    pointer, stride = struct.unpack_from("=Qi", row)
    return pointer, stride


@pytest.mark.skipif(
    not current_platform.is_cuda()
    or current_platform.get_device_capability() != (7, 0),
    reason="requires NVIDIA V100/SM70",
)
def test_sm70_fp8_strided_ptr_tables_survive_laguna_layer_sequence():
    num_experts = 256
    guard = 32
    cases = (
        ((num_experts, 256, 3072), (num_experts, 2, 24), 3072, 24),
        ((num_experts, 3072, 128), (num_experts, 24, 1), 128, 1),
    )
    retained = []

    for weight_shape, scale_shape, k_ld, q_ld in cases:
        weight_numel = int(torch.tensor(weight_shape).prod().item())
        weight_storage = torch.empty(
            weight_numel + 2 * guard, dtype=torch.uint8, device="cuda"
        )
        weight_storage[:guard] = 0xA5
        weight_storage[-guard:] = 0xA5
        weights = weight_storage[guard:-guard].view(weight_shape)

        scale_numel = int(torch.tensor(scale_shape).prod().item())
        scale_storage = torch.empty(
            scale_numel + 2 * guard, dtype=torch.float16, device="cuda"
        )
        scale_storage[:guard] = -1234.0
        scale_storage[-guard:] = -1234.0
        scales = scale_storage[guard:-guard].view(scale_shape)

        for _ in range(48):
            retained.append(
                torch.ops._C.awq_moe_build_strided_ptrs(
                    weights, scales, k_ld, q_ld, num_experts
                )
            )

        torch.accelerator.synchronize()
        weight_expert_stride = weights.stride(0) * weights.element_size()
        scale_expert_stride = scales.stride(0) * scales.element_size()
        for layer in (0, 31, 32, 47):
            weight_table, scale_table = retained[-48 + layer]
            assert weight_table.shape == (num_experts * 16,)
            assert scale_table.shape == (num_experts * 16,)
            for expert in (0, 1, num_experts - 1):
                weight_ptr, weight_ld = _read_strided_ptr_row(weight_table, expert)
                scale_ptr, scale_ld = _read_strided_ptr_row(scale_table, expert)
                assert weight_ptr == weights.data_ptr() + expert * weight_expert_stride
                assert scale_ptr == scales.data_ptr() + expert * scale_expert_stride
                assert weight_ld == k_ld
                assert scale_ld == q_ld

        assert torch.all(weight_storage[:guard] == 0xA5)
        assert torch.all(weight_storage[-guard:] == 0xA5)
        assert torch.all(scale_storage[:guard] == -1234.0)
        assert torch.all(scale_storage[-guard:] == -1234.0)


@pytest.mark.skipif(
    not current_platform.is_cuda()
    or current_platform.get_device_capability() != (7, 0),
    reason="requires NVIDIA V100/SM70",
)
def test_sm70_fp8_strided_ptr_tables_validate_inputs():
    weights = torch.empty((4, 16, 32), dtype=torch.uint8, device="cuda")
    scales = torch.empty((4, 1, 1), dtype=torch.float16, device="cuda")

    with pytest.raises(RuntimeError, match="weights must have rank >= 1"):
        torch.ops._C.awq_moe_build_strided_ptrs(
            torch.empty((), dtype=torch.uint8, device="cuda"), scales, 32, 1, 4
        )
    with pytest.raises(RuntimeError, match="num_experts must be > 0"):
        torch.ops._C.awq_moe_build_strided_ptrs(weights, scales, 32, 1, 0)
    with pytest.raises(RuntimeError, match="weights dim0 != num_experts"):
        torch.ops._C.awq_moe_build_strided_ptrs(weights, scales, 32, 1, 3)
    with pytest.raises(RuntimeError, match="k_ld must fit a positive int"):
        torch.ops._C.awq_moe_build_strided_ptrs(weights, scales, 0, 1, 4)
    with pytest.raises(RuntimeError, match="q_ld must fit a positive int"):
        torch.ops._C.awq_moe_build_strided_ptrs(weights, scales, 32, 0, 4)


@pytest.mark.skipif(
    not current_platform.is_cuda()
    or current_platform.device_count() < 2
    or any(
        current_platform.get_device_capability(device) != (7, 0) for device in range(2)
    ),
    reason="requires two NVIDIA V100/SM70 devices",
)
def test_sm70_fp8_strided_ptr_tables_use_input_device():
    original_device = torch.accelerator.current_device_index()
    torch.accelerator.set_device_index(0)
    try:
        weights = torch.empty((4, 16, 32), dtype=torch.uint8, device="cuda:1")
        scales = torch.empty((4, 1, 1), dtype=torch.float16, device="cuda:1")

        weight_table, scale_table = torch.ops._C.awq_moe_build_strided_ptrs(
            weights, scales, 32, 1, 4
        )
        torch.accelerator.synchronize(1)

        assert weight_table.device == torch.device("cuda:1")
        assert scale_table.device == torch.device("cuda:1")
        assert torch.accelerator.current_device_index() == 0
        assert _read_strided_ptr_row(weight_table, 3) == (
            weights.data_ptr() + 3 * weights.stride(0) * weights.element_size(),
            32,
        )
        with pytest.raises(RuntimeError, match="same CUDA device"):
            torch.ops._C.awq_moe_build_strided_ptrs(
                weights,
                torch.empty((4, 1, 1), dtype=torch.float16, device="cuda:0"),
                32,
                1,
                4,
            )
    finally:
        torch.accelerator.set_device_index(original_device)


@pytest.mark.slow_test
@pytest.mark.skipif(
    not current_platform.is_cuda()
    or current_platform.get_device_capability() != (7, 0),
    reason="requires NVIDIA V100/SM70",
)
def test_sm70_fp8_strided_ptr_tables_fail_cleanly_under_memory_pressure():
    original_device = torch.accelerator.current_device_index()
    torch.accelerator.set_device_index(0)
    reservations = []
    try:
        torch.accelerator.empty_cache()
        weights = torch.empty((4, 16, 32), dtype=torch.uint8, device="cuda")
        scales = torch.empty((4, 1, 1), dtype=torch.float16, device="cuda")
        free_bytes, _ = current_platform.mem_get_info()
        if free_bytes < 512 * 1024**2:
            pytest.skip("requires at least 512 MiB of initially free GPU memory")

        target_free = 32 * 1024**2
        reserve_margin = 16 * 1024**2
        chunk_limit = 256 * 1024**2
        while free_bytes > target_free + reserve_margin:
            reservations.append(
                torch.empty(
                    min(chunk_limit, free_bytes - target_free - reserve_margin),
                    dtype=torch.uint8,
                    device="cuda",
                )
            )
            free_bytes, _ = current_platform.mem_get_info()

        for _ in range(48):
            weight_table, scale_table = torch.ops._C.awq_moe_build_strided_ptrs(
                weights, scales, 32, 1, 4
            )
            assert weight_table.numel() == scale_table.numel() == 64
        torch.accelerator.synchronize()

        free_bytes, _ = current_platform.mem_get_info()
        with pytest.raises(torch.OutOfMemoryError):
            torch.empty(free_bytes + 64 * 1024**2, dtype=torch.uint8, device="cuda")

        weight_table, scale_table = torch.ops._C.awq_moe_build_strided_ptrs(
            weights, scales, 32, 1, 4
        )
        torch.accelerator.synchronize()
        assert _read_strided_ptr_row(weight_table, 3) == (
            weights.data_ptr() + 3 * weights.stride(0) * weights.element_size(),
            32,
        )
        assert _read_strided_ptr_row(scale_table, 3) == (
            scales.data_ptr() + 3 * scales.stride(0) * scales.element_size(),
            1,
        )
    finally:
        del reservations
        torch.accelerator.empty_cache()
        torch.accelerator.set_device_index(original_device)
