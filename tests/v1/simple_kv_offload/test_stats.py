# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

from prometheus_client import Counter, Gauge, Histogram

from vllm.distributed.kv_transfer.kv_connector.v1.offloading.metrics import (
    _TransferMetricName as _NativeTransferMetricName,
)
from vllm.v1.simple_kv_offload.stats import (
    SimpleCPUOffloadPromMetrics,
    SimpleCPUOffloadStats,
    _TransferMetricName,
)

MiB = 1024**2
GiB = 1024**3


class _FakeMetric:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.children: list[_FakeMetric] = []
        self.observed: list[int | float] = []
        self.increments: list[int | float] = []

    def labels(self, *labelvalues):
        child = _FakeMetric(**self.kwargs)
        self.children.append(child)
        return child

    def observe(self, value):
        self.observed.append(value)

    def inc(self, value):
        self.increments.append(value)


def _build_prom_metrics() -> SimpleCPUOffloadPromMetrics:
    vllm_config = SimpleNamespace(kv_transfer_config=SimpleNamespace())
    metric_types = {Gauge: _FakeMetric, Counter: _FakeMetric, Histogram: _FakeMetric}
    return SimpleCPUOffloadPromMetrics(
        vllm_config, metric_types, ["engine"], {0: ["0"]}
    )


def test_metric_names_match_offloading_connector():
    """Both offload paths must expose the same names for the same concepts."""
    for attr in ("LOAD_BYTES", "LOAD_TIME", "LOAD_SIZE"):
        assert getattr(_TransferMetricName, attr) == getattr(
            _NativeTransferMetricName, attr
        )
    for attr in ("STORE_BYTES", "STORE_TIME", "STORE_SIZE"):
        assert getattr(_TransferMetricName, attr) == getattr(
            _NativeTransferMetricName, attr
        )


def test_empty_stats():
    stats = SimpleCPUOffloadStats()
    assert stats.is_empty()
    assert stats.reduce() == {}


def test_record_and_reduce():
    stats = SimpleCPUOffloadStats()
    stats.record_transfer(is_store=True, num_bytes=GiB, seconds=0.5)
    stats.record_transfer(is_store=True, num_bytes=GiB, seconds=0.5)
    stats.record_transfer(is_store=False, num_bytes=2 * GiB, seconds=0.5)

    assert not stats.is_empty()
    reduced = stats.reduce()
    assert reduced["store transfers"] == 2
    assert reduced["store GiB"] == 2.0
    assert reduced["store seconds"] == 1.0
    assert reduced["store GiB/s per rank"] == 2.0
    assert reduced["load transfers"] == 1
    assert reduced["load GiB"] == 2.0
    assert reduced["load GiB/s per rank"] == 4.0


def test_reduce_omits_idle_direction():
    stats = SimpleCPUOffloadStats()
    stats.record_transfer(is_store=True, num_bytes=MiB, seconds=0.01)
    reduced = stats.reduce()
    assert "store GiB" in reduced
    assert not any(key.startswith("load") for key in reduced)


def test_aggregate_sums_across_ranks():
    rank0 = SimpleCPUOffloadStats()
    rank0.record_transfer(is_store=True, num_bytes=GiB, seconds=0.25)
    rank1 = SimpleCPUOffloadStats()
    rank1.record_transfer(is_store=True, num_bytes=GiB, seconds=0.25)

    aggregated = rank0.aggregate(rank1)
    reduced = aggregated.reduce()
    # Bytes total the node; seconds total the concurrent per-rank spend, so the
    # rate stays a per-rank average rather than doubling.
    assert reduced["store transfers"] == 2
    assert reduced["store GiB"] == 2.0
    assert reduced["store seconds"] == 0.5
    assert reduced["store GiB/s per rank"] == 4.0


def test_aggregate_with_empty_is_noop():
    stats = SimpleCPUOffloadStats()
    stats.record_transfer(is_store=False, num_bytes=MiB, seconds=0.01)
    before = stats.reduce()
    assert stats.aggregate(SimpleCPUOffloadStats()).reduce() == before


def test_clone_and_reset():
    stats = SimpleCPUOffloadStats()
    stats.record_transfer(is_store=True, num_bytes=MiB, seconds=0.01)
    snapshot = stats.clone_and_reset()

    assert not snapshot.is_empty()
    assert stats.is_empty()


def test_prom_metrics_observe():
    prom = _build_prom_metrics()
    stats = SimpleCPUOffloadStats()
    stats.record_transfer(is_store=True, num_bytes=4 * MiB, seconds=0.02)
    stats.record_transfer(is_store=True, num_bytes=8 * MiB, seconds=0.03)
    stats.record_transfer(is_store=False, num_bytes=2 * MiB, seconds=0.01)

    prom.observe(stats.data, engine_idx=0)

    # Counters take the interval sum; the size histogram sees each transfer.
    assert prom.counter_store_bytes[0].increments == [12 * MiB]
    assert prom.counter_store_time[0].increments == [0.05]
    assert prom.histogram_store_size[0].observed == [4 * MiB, 8 * MiB]
    assert prom.counter_load_bytes[0].increments == [2 * MiB]
    assert prom.histogram_load_size[0].observed == [2 * MiB]


def test_prom_metrics_observe_empty_payload_is_noop():
    prom = _build_prom_metrics()
    prom.observe(SimpleCPUOffloadStats().data, engine_idx=0)

    assert prom.counter_store_bytes[0].increments == []
    assert prom.counter_store_time[0].increments == []
    assert prom.histogram_store_size[0].observed == []


def test_prom_metrics_registers_expected_names():
    prom = _build_prom_metrics()
    assert prom.counter_store_bytes[0].kwargs["name"] == (
        _TransferMetricName.STORE_BYTES
    )
    assert prom.counter_load_time[0].kwargs["name"] == _TransferMetricName.LOAD_TIME
    assert prom.histogram_load_size[0].kwargs["buckets"][0] == 1e6
