# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Worker-side unit tests for SimpleCPUOffloadConnector.

Covers the GPU->CPU store cross-stream synchronization: the store copy must be
ordered after the compute stream that writes the KV blocks, otherwise it can
read partially written / stale blocks and silently corrupt the CPU cache.
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest
import torch

from vllm.platforms import current_platform

if not current_platform.is_cuda_alike():
    pytest.skip("Requires CUDA or ROCm", allow_module_level=True)

from tests.v1.attention.utils import dense_kv_cache_views
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheLayout
from vllm.v1.simple_kv_offload.copy_backend import DmaCopyBackend, TransferEvent
from vllm.v1.simple_kv_offload.cuda_mem_ops import (
    CU_MEMCPY_SRC_ACCESS_ORDER_ANY,
    CU_MEMCPY_SRC_ACCESS_ORDER_STREAM,
    build_params,
    pin_tensor,
)
from vllm.v1.simple_kv_offload.metadata import SimpleCPUOffloadMetadata
from vllm.v1.simple_kv_offload.worker import SimpleCPUOffloadWorker

NUM_BLOCKS = 64
BLOCK_BYTES = 4096
ITERS = 30
# Keep the compute stream busy so the KV write lands late; this makes the
# store-vs-compute race deterministic instead of timing-dependent.
SLEEP_CYCLES = 50_000_000


def _make_backend() -> tuple[DmaCopyBackend, torch.Tensor, torch.Tensor]:
    gpu = {"k": torch.zeros((NUM_BLOCKS, BLOCK_BYTES), dtype=torch.int8, device="cuda")}
    cpu = {"k": torch.zeros((NUM_BLOCKS, BLOCK_BYTES), dtype=torch.int8, device="cpu")}
    pin_tensor(cpu["k"])
    low_pri, _ = torch.cuda.Stream.priority_range()
    backend = DmaCopyBackend()
    backend.init(
        gpu,
        cpu,
        gpu["k"].device,
        torch.cuda.Stream(priority=low_pri),
        torch.cuda.Stream(priority=low_pri),
    )
    return backend, gpu["k"], cpu["k"]


def _drive_store(
    backend: DmaCopyBackend,
    gpu: torch.Tensor,
    cpu: torch.Tensor,
    *,
    with_barrier: bool,
) -> int:
    """Run ITERS store cycles; return how many landed corrupted in the CPU pool.

    Each cycle writes a unique value on a compute stream (after a deliberate
    delay) and then issues the GPU->CPU store. The store is issued *after* the
    write in host program order, mirroring the connector's deferred-store
    assumption. Only the compute-done event creates a real device-side
    happens-before edge.
    """
    block_ids = list(range(gpu.shape[0]))
    compute_stream = torch.cuda.Stream()
    corrupt = 0
    for it in range(ITERS):
        val = (it % 126) + 1  # 1..126; distinct from the zero-initialized pool
        with torch.cuda.stream(compute_stream):
            torch.cuda._sleep(SLEEP_CYCLES)
            gpu.fill_(val)

        wait_event = None
        if with_barrier:
            wait_event = torch.Event()
            wait_event.record(compute_stream)

        store_events: list[TransferEvent] = []
        backend.launch_copy(
            block_ids,
            block_ids,
            is_store=True,
            event_idx=it,
            events_list=store_events,
            wait_event=wait_event,
        )

        deadline = time.time() + 10.0
        while not store_events and time.time() < deadline:
            time.sleep(0.0005)
        assert store_events, "background copy was never enqueued"
        store_events[0][1].synchronize()

        if int((cpu[:, 0].to(torch.int32) != val).sum().item()):
            corrupt += 1

    # Drain the compute stream before returning: in the no-barrier control
    # phase the store never waits on compute, so the host loop runs far ahead
    # and leaves a backlog of sleep+fill kernels in flight. Without this, the
    # leftover control-phase fills race the barrier phase's fill->copy window
    # on the shared gpu tensor and flakily corrupt one iteration.
    compute_stream.synchronize()
    return corrupt


def test_store_orders_after_compute_write():
    """The store must wait for the compute event; without it, it races.

    Asserts both directions so the test is self-validating: the no-barrier
    control must actually corrupt (proving the race window is exercised), and
    the fixed path with the compute-done event must be clean.
    """
    backend, gpu, cpu = _make_backend()
    try:
        control = _drive_store(backend, gpu, cpu, with_barrier=False)
        fixed = _drive_store(backend, gpu, cpu, with_barrier=True)
    finally:
        backend.shutdown()

    assert control > 0, (
        "no-barrier store did not race the compute write; the test no longer "
        "exercises the hazard it is meant to guard"
    )
    assert fixed == 0, f"store raced compute even with the barrier: {fixed} corrupt"


class _RecordingBackend:
    """Captures launch_copy calls without touching the GPU."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def launch_copy(
        self,
        src_blocks,
        dst_blocks,
        is_store,
        event_idx,
        events_list,
        wait_event=None,
        num_bytes=0,
    ) -> None:
        self.calls.append(
            {"is_store": is_store, "wait_event": wait_event, "num_bytes": num_bytes}
        )


def test_get_finished_passes_wait_event_for_store_only():
    """get_finished gates stores on a compute-done event but not loads."""
    worker = SimpleCPUOffloadWorker(
        vllm_config=None, kv_cache_config=None, cpu_capacity_bytes=0
    )
    recording = _RecordingBackend()
    worker._backend = recording
    worker._connector_metadata = SimpleCPUOffloadMetadata(
        load_event=0,
        load_gpu_blocks=[0],
        load_cpu_blocks=[0],
        store_event=1,
        store_gpu_blocks=[1],
        store_cpu_blocks=[1],
    )

    worker.get_finished(set())

    store_calls = [c for c in recording.calls if c["is_store"]]
    load_calls = [c for c in recording.calls if not c["is_store"]]
    assert len(store_calls) == 1
    assert len(load_calls) == 1
    assert isinstance(store_calls[0]["wait_event"], torch.Event)
    assert load_calls[0]["wait_event"] is None


def test_get_finished_reports_transfer_bytes():
    """Launched transfers are sized as block count * bytes per block."""
    worker = SimpleCPUOffloadWorker(
        vllm_config=None, kv_cache_config=None, cpu_capacity_bytes=0
    )
    recording = _RecordingBackend()
    worker._backend = recording
    worker.total_bytes_per_block = 4096
    worker._connector_metadata = SimpleCPUOffloadMetadata(
        load_event=0,
        load_gpu_blocks=[0, 1],
        load_cpu_blocks=[0, 1],
        store_event=1,
        store_gpu_blocks=[2, 3, 4],
        store_cpu_blocks=[2, 3, 4],
    )

    worker.get_finished(set())

    by_dir = {c["is_store"]: c["num_bytes"] for c in recording.calls}
    assert by_dir[False] == 2 * 4096
    assert by_dir[True] == 3 * 4096


def test_worker_records_transfer_stats():
    """Real DMA transfers report their bytes and nonzero device seconds."""
    backend, gpu, cpu = _make_backend()
    worker = SimpleCPUOffloadWorker(
        vllm_config=None, kv_cache_config=None, cpu_capacity_bytes=0
    )
    worker._backend = backend
    worker.total_bytes_per_block = BLOCK_BYTES
    block_ids = list(range(NUM_BLOCKS))
    worker.bind_connector_metadata(
        SimpleCPUOffloadMetadata(
            load_event=0,
            load_gpu_blocks=block_ids,
            load_cpu_blocks=block_ids,
            store_event=1,
            store_gpu_blocks=block_ids,
            store_cpu_blocks=block_ids,
        )
    )

    worker.get_finished(set())
    # Drop the metadata so the flush loop below cannot relaunch the transfers.
    worker.clear_connector_metadata()

    # The backend enqueues its events from a background thread, so wait for both
    # directions to land instead of assuming they are visible on the first pass.
    deadline = time.time() + 10.0
    while time.time() < deadline:
        worker._flush_and_sync_all()
        seen = worker._stats.reduce()
        if "store transfers" in seen and "load transfers" in seen:
            break
        time.sleep(0.001)

    stats = worker.get_stats()
    assert stats is not None, "no transfer stats were recorded"
    reduced = stats.reduce()
    expected_gib = round(NUM_BLOCKS * BLOCK_BYTES / float(1024**3), 3)
    for direction in ("store", "load"):
        assert reduced[f"{direction} transfers"] == 1
        assert reduced[f"{direction} GiB"] == expected_gib
        assert reduced[f"{direction} seconds"] > 0.0

    # get_stats() hands ownership to the framework and starts a fresh interval.
    assert worker.get_stats() is None

    backend.shutdown()


def test_build_params_src_access_order():
    """build_params defaults to ANY and honors an explicit STREAM override."""
    gpu = {"k": torch.zeros((4, 64), dtype=torch.int8, device="cuda")}
    cpu = {"k": torch.zeros((4, 64), dtype=torch.int8, device="cpu")}
    stream = torch.cuda.Stream()

    default = build_params(gpu, cpu, stream)
    assert default.attrs.srcAccessOrder == CU_MEMCPY_SRC_ACCESS_ORDER_ANY

    ordered = build_params(
        gpu, cpu, stream, src_access_order=CU_MEMCPY_SRC_ACCESS_ORDER_STREAM
    )
    assert ordered.attrs.srcAccessOrder == CU_MEMCPY_SRC_ACCESS_ORDER_STREAM


@pytest.mark.parametrize("layout", list(KVCacheLayout))
def test_register_shared_kv_cache_storage(monkeypatch, layout: KVCacheLayout):
    num_blocks = 4
    num_layers = 2
    spec = FullAttentionSpec(
        block_size=2,
        num_kv_heads=2,
        head_size=2,
        dtype=torch.float16,
    )
    raw = torch.zeros(
        num_blocks * num_layers * spec.page_size_bytes,
        dtype=torch.int8,
        device="cuda",
    )
    caches = dense_kv_cache_views(raw, spec, num_blocks, num_layers, layout)
    cache_config = MagicMock(
        num_blocks=num_blocks,
        kv_cache_tensors=[MagicMock(size=raw.nbytes)],
    )
    worker = SimpleCPUOffloadWorker(
        vllm_config=None,
        kv_cache_config=cache_config,
        cpu_capacity_bytes=raw.nbytes,
    )
    worker._backend = MagicMock()
    monkeypatch.setattr("vllm.v1.simple_kv_offload.worker.PIN_MEMORY", False)

    worker.register_kv_caches(
        {f"layer.{layer_idx}": cache for layer_idx, cache in enumerate(caches)}
    )

    assert worker.gpu_kv_caches is not None
    if layout.is_layer_compact and not layout.is_block_compact:
        expected_regions = num_layers * spec.num_heads
        expected_block_bytes = spec.page_size_bytes // spec.num_heads
    elif layout.is_layer_compact:
        expected_regions = num_layers
        expected_block_bytes = spec.page_size_bytes
    else:
        expected_regions = 1
        expected_block_bytes = spec.page_size_bytes * num_layers
    assert len(worker.gpu_kv_caches) == expected_regions
    assert {cache.shape for cache in worker.gpu_kv_caches.values()} == {
        (num_blocks, expected_block_bytes)
    }


def test_register_kv_cache_storage_with_trailing_padding(monkeypatch):
    num_blocks = 4
    block_bytes = 32
    cache_bytes = num_blocks * block_bytes
    raw = torch.zeros(4096, dtype=torch.int8, device="cuda")
    cache = raw[:cache_bytes].view(num_blocks, block_bytes)
    worker = SimpleCPUOffloadWorker(
        vllm_config=None,
        kv_cache_config=MagicMock(
            num_blocks=num_blocks,
            kv_cache_tensors=[MagicMock(size=cache_bytes)],
        ),
        cpu_capacity_bytes=cache_bytes,
    )
    worker._backend = MagicMock()
    monkeypatch.setattr("vllm.v1.simple_kv_offload.worker.PIN_MEMORY", False)

    worker.register_kv_caches({"layer.0": cache})

    assert worker.gpu_kv_caches is not None
    assert list(worker.gpu_kv_caches) == ["layer.0"]
    assert worker.gpu_kv_caches["layer.0"].shape == (num_blocks, block_bytes)


def test_register_separate_kv_head_groups(monkeypatch):
    # LHBNC hoists the K/V head groups outside the block dim, so each layer's
    # blocks are registered as one region per group (K, V).
    layout = KVCacheLayout.LHBNC
    num_blocks = 4
    num_layers = 2
    spec = FullAttentionSpec(
        block_size=2,
        num_kv_heads=2,
        head_size=2,
        dtype=torch.float16,
        num_head_slots=2,
        state_content_bytes=2 * 2 * 2,
    )
    raw = torch.zeros(
        num_blocks * num_layers * spec.page_size_bytes,
        dtype=torch.int8,
        device="cuda",
    )
    caches = dense_kv_cache_views(raw, spec, num_blocks, num_layers, layout)
    worker = SimpleCPUOffloadWorker(
        vllm_config=None,
        kv_cache_config=MagicMock(
            num_blocks=num_blocks,
            kv_cache_tensors=[MagicMock(size=raw.nbytes)],
        ),
        cpu_capacity_bytes=raw.nbytes,
    )
    worker._backend = MagicMock()
    monkeypatch.setattr("vllm.v1.simple_kv_offload.worker.PIN_MEMORY", False)

    worker.register_kv_caches(
        {f"layer.{layer_idx}": cache for layer_idx, cache in enumerate(caches)}
    )

    assert worker.gpu_kv_caches is not None
    assert len(worker.gpu_kv_caches) == num_layers * spec.num_heads
    per_group_block_bytes = (
        spec.num_kv_heads * spec.block_size * spec.head_size * spec.dtype.itemsize
    )
    assert {cache.shape for cache in worker.gpu_kv_caches.values()} == {
        (num_blocks, per_group_block_bytes)
    }
