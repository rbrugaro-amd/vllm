# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stats and Prometheus metrics for SimpleCPUOffloadConnector.

Metric names and semantics match the OffloadingConnector
(``kv_connector/v1/offloading/metrics.py``), so both offload paths are
interchangeable on a dashboard: ``store`` is GPU -> offload medium, ``load``
is offload medium -> GPU.

Stats are collected worker-side, so every rank reports and ``KVOutputAggregator``
sums the payloads across the TP/PP world. Bytes therefore total the whole node,
while time totals the per-rank DMA seconds that ranks spend concurrently.
``bytes / time`` is consequently the *average per-rank* bandwidth; multiply by
``world_size`` to estimate the node aggregate.
"""

import copy
from dataclasses import dataclass
from typing import Any

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.metrics import (
    KVConnectorPromMetrics,
    KVConnectorStats,
    PromMetric,
    PromMetricT,
)
from vllm.v1.metrics.utils import create_metric_per_engine


class _TransferMetricName:
    """Flat metric names, shared with the OffloadingConnector."""

    LOAD_BYTES = "vllm:kv_offload_load_bytes"
    LOAD_TIME = "vllm:kv_offload_load_time"
    LOAD_SIZE = "vllm:kv_offload_load_size"
    STORE_BYTES = "vllm:kv_offload_store_bytes"
    STORE_TIME = "vllm:kv_offload_store_time"
    STORE_SIZE = "vllm:kv_offload_store_size"


# Matches OffloadingConnector's TRANSFER_SIZE_BUCKETS so the size histograms
# are comparable between the two offload paths.
TRANSFER_SIZE_BUCKETS = (
    1e6,
    5e6,
    10e6,
    20e6,
    40e6,
    60e6,
    80e6,
    100e6,
    150e6,
    200e6,
)

_LOAD_BYTES = "load_bytes"
_LOAD_TIME = "load_time"
_STORE_BYTES = "store_bytes"
_STORE_TIME = "store_time"

_GIB = float(1024**3)


@dataclass
class SimpleCPUOffloadStats(KVConnectorStats):
    """Per-transfer byte and duration observations, split by direction.

    One list entry per completed transfer. The byte lists double as the
    ``*_size`` histogram observations, and their sums are the ``*_bytes``
    counters.
    """

    def __post_init__(self):
        if not self.data:
            self.reset()

    def reset(self):
        # Must be serializable: plain lists of numbers only.
        self.data: dict[str, list[float | int]] = {
            _LOAD_BYTES: [],
            _LOAD_TIME: [],
            _STORE_BYTES: [],
            _STORE_TIME: [],
        }

    def record_transfer(self, is_store: bool, num_bytes: int, seconds: float) -> None:
        bytes_key, time_key = (
            (_STORE_BYTES, _STORE_TIME) if is_store else (_LOAD_BYTES, _LOAD_TIME)
        )
        self.data[bytes_key].append(num_bytes)
        self.data[time_key].append(seconds)

    def clone_and_reset(self) -> "SimpleCPUOffloadStats":
        old = copy.copy(self)
        self.reset()
        return old

    def is_empty(self) -> bool:
        return not any(self.data.values())

    def aggregate(self, other: KVConnectorStats) -> KVConnectorStats:
        if not other.is_empty():
            for key, values in other.data.items():
                self.data[key].extend(values)
        return self

    def reduce(self) -> dict[str, int | float]:
        out: dict[str, int | float] = {}
        for direction, bytes_key, time_key in (
            ("store", _STORE_BYTES, _STORE_TIME),
            ("load", _LOAD_BYTES, _LOAD_TIME),
        ):
            transfers = self.data[bytes_key]
            if not transfers:
                continue
            total_bytes = sum(transfers)
            seconds = sum(self.data[time_key])
            out[f"{direction} transfers"] = len(transfers)
            out[f"{direction} GiB"] = round(total_bytes / _GIB, 3)
            out[f"{direction} seconds"] = round(seconds, 4)
            if seconds > 0:
                # Per-rank average: numerator and denominator are both summed
                # over ranks that transfer concurrently.
                out[f"{direction} GiB/s per rank"] = round(
                    total_bytes / _GIB / seconds, 2
                )
        return out


class SimpleCPUOffloadPromMetrics(KVConnectorPromMetrics):
    def __init__(
        self,
        vllm_config: VllmConfig,
        metric_types: dict[type[PromMetric], type[PromMetricT]],
        labelnames: list[str],
        per_engine_labelvalues: dict[int, list[object]],
    ):
        super().__init__(vllm_config, metric_types, labelnames, per_engine_labelvalues)

        def counter(name: str, documentation: str):
            return create_metric_per_engine(
                self._counter_cls(
                    name=name,
                    documentation=documentation,
                    labelnames=labelnames,
                ),
                self.per_engine_labelvalues,
            )

        def size_histogram(name: str, documentation: str):
            return create_metric_per_engine(
                self._histogram_cls(
                    name=name,
                    documentation=documentation,
                    buckets=TRANSFER_SIZE_BUCKETS,
                    labelnames=labelnames,
                ),
                self.per_engine_labelvalues,
            )

        self.counter_store_bytes = counter(
            _TransferMetricName.STORE_BYTES,
            "Total bytes stored from GPU to offload storage.",
        )
        self.counter_store_time = counter(
            _TransferMetricName.STORE_TIME,
            "Total store time from GPU to offload storage, in seconds. Summed "
            "over ranks, so bytes/time is the average per-rank bandwidth.",
        )
        self.histogram_store_size = size_histogram(
            _TransferMetricName.STORE_SIZE,
            "Histogram of KV offload store operation size, in bytes.",
        )
        self.counter_load_bytes = counter(
            _TransferMetricName.LOAD_BYTES,
            "Total bytes loaded from offload storage to GPU.",
        )
        self.counter_load_time = counter(
            _TransferMetricName.LOAD_TIME,
            "Total load time from offload storage to GPU, in seconds. Summed "
            "over ranks, so bytes/time is the average per-rank bandwidth.",
        )
        self.histogram_load_size = size_histogram(
            _TransferMetricName.LOAD_SIZE,
            "Histogram of KV offload load operation size, in bytes.",
        )

    def observe(self, transfer_stats_data: dict[str, Any], engine_idx: int = 0):
        for bytes_key, time_key, bytes_counter, time_counter, size_histogram in (
            (
                _STORE_BYTES,
                _STORE_TIME,
                self.counter_store_bytes,
                self.counter_store_time,
                self.histogram_store_size,
            ),
            (
                _LOAD_BYTES,
                _LOAD_TIME,
                self.counter_load_bytes,
                self.counter_load_time,
                self.histogram_load_size,
            ),
        ):
            transfers = transfer_stats_data.get(bytes_key) or []
            if transfers:
                bytes_counter[engine_idx].inc(sum(transfers))
                for num_bytes in transfers:
                    size_histogram[engine_idx].observe(num_bytes)
            seconds = sum(transfer_stats_data.get(time_key) or [])
            if seconds:
                time_counter[engine_idx].inc(seconds)
