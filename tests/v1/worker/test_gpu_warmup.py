# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import MagicMock

from vllm.v1.worker.gpu.warmup import warmup_kernels


def test_warmup_covers_single_request_short_prefill(monkeypatch):
    model_runner = SimpleNamespace(
        num_speculative_steps=0,
        kv_cache_config=SimpleNamespace(
            kv_cache_groups=[
                SimpleNamespace(kv_cache_spec=SimpleNamespace(block_size=16))
            ],
            num_blocks=128,
        ),
        scheduler_config=SimpleNamespace(max_num_seqs=4, max_num_batched_tokens=32),
        is_pooling_model=False,
        is_last_pp_rank=False,
        kv_connector=MagicMock(),
    )
    execute_outputs = []
    sample_outputs = []
    monkeypatch.setattr("torch.accelerator.synchronize", lambda: None)

    warmup_kernels(model_runner, execute_outputs.append, sample_outputs.append)

    probe_output = execute_outputs[-2]
    assert probe_output.total_num_scheduled_tokens == 6
    assert probe_output.num_scheduled_tokens == {"_warmup_probe_": 6}
    assert len(probe_output.scheduled_new_reqs) == 1
    assert execute_outputs[-1].finished_req_ids == {"_warmup_probe_"}
    assert sample_outputs[-1] is None


def test_warmup_covers_scheduler_sized_serving_prefill(monkeypatch):
    model_runner = SimpleNamespace(
        num_speculative_steps=0,
        kv_cache_config=SimpleNamespace(
            kv_cache_groups=[
                SimpleNamespace(kv_cache_spec=SimpleNamespace(block_size=16))
            ],
            num_blocks=1024,
        ),
        scheduler_config=SimpleNamespace(max_num_seqs=64, max_num_batched_tokens=2112),
        is_pooling_model=False,
        is_last_pp_rank=False,
        kv_connector=MagicMock(),
    )
    execute_outputs = []
    monkeypatch.setattr("torch.accelerator.synchronize", lambda: None)

    warmup_kernels(model_runner, execute_outputs.append, lambda _: None)

    serving_output = next(
        output
        for output in execute_outputs
        if any(
            req_id.startswith("_warmup_serving_")
            for req_id in output.num_scheduled_tokens
        )
    )
    assert len(serving_output.scheduled_new_reqs) == 8
    assert serving_output.total_num_scheduled_tokens == 2048
    assert set(serving_output.num_scheduled_tokens.values()) == {256}
    serving_ids = set(serving_output.num_scheduled_tokens)
    assert any(output.finished_req_ids == serving_ids for output in execute_outputs)
