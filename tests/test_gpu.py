import time
from types import SimpleNamespace

import pytest

import camera_bench.gpu as gpu_module
from camera_bench.gpu import GpuMonitor


class _NVMLError(Exception):
    pass


def _fake_nvml(monkeypatch, *, fail_decoder=False, fail_all=False, handle_indexes=(0,)):
    """Install a stub pynvml into camera_bench.gpu and return its call counters."""
    calls = {"handle": 0, "memory": 0, "util": 0, "decoder": 0}

    def nvmlDeviceGetHandleByIndex(index):
        calls["handle"] += 1
        if index not in handle_indexes:
            raise _NVMLError(f"Invalid index: {index}")
        return f"handle{index}"

    def nvmlDeviceGetMemoryInfo(handle):
        calls["memory"] += 1
        if fail_all:
            raise _NVMLError("GPU is lost")
        return SimpleNamespace(used=100 * 2**20)

    def nvmlDeviceGetUtilizationRates(handle):
        calls["util"] += 1
        if fail_all:
            raise _NVMLError("GPU is lost")
        return SimpleNamespace(gpu=50)

    def nvmlDeviceGetDecoderUtilization(handle):
        calls["decoder"] += 1
        if fail_all or fail_decoder:
            raise _NVMLError("Not Supported")
        return (7, 2)

    fake = SimpleNamespace(
        NVMLError=_NVMLError,
        nvmlDeviceGetHandleByIndex=nvmlDeviceGetHandleByIndex,
        nvmlDeviceGetMemoryInfo=nvmlDeviceGetMemoryInfo,
        nvmlDeviceGetUtilizationRates=nvmlDeviceGetUtilizationRates,
        nvmlDeviceGetDecoderUtilization=nvmlDeviceGetDecoderUtilization,
    )
    monkeypatch.setattr(gpu_module, "pynvml", fake)
    monkeypatch.setattr(gpu_module, "_NVML", True)
    return calls


def _collect(monitor, seconds=0.05):
    with monitor:
        time.sleep(seconds)


def test_failing_decoder_query_keeps_other_metrics(monkeypatch):
    _fake_nvml(monkeypatch, fail_decoder=True)
    monitor = GpuMonitor(index=0, interval=0.005)
    _collect(monitor)

    assert monitor.samples, "loop must keep sampling"
    assert all(s.memory_used_mb == pytest.approx(100.0) for s in monitor.samples)
    assert all(s.gpu_util == 50 for s in monitor.samples)
    assert all(s.decoder_util is None for s in monitor.samples)
    assert monitor.peak_memory_mb() == pytest.approx(100.0)
    assert monitor.mean_gpu_util() == 50
    assert monitor.mean_decoder_util() is None


def test_all_queries_failed_stops_loop(monkeypatch):
    calls = _fake_nvml(monkeypatch, fail_all=True)
    monitor = GpuMonitor(index=0, interval=0.005)
    _collect(monitor)

    assert monitor.samples == []
    assert calls["memory"] == 1  # the loop stopped after the first dead sample


def test_invalid_index_gives_clear_error(monkeypatch):
    _fake_nvml(monkeypatch, handle_indexes=(0,))
    monitor = GpuMonitor(index=9)
    with pytest.raises(ValueError, match="GPU 9 is not available"):
        monitor.__enter__()
    with pytest.raises(ValueError, match="GPU 9 is not available"):
        monitor.snapshot_memory_mb()


def test_none_index_disables_telemetry_without_nvml_calls(monkeypatch):
    calls = _fake_nvml(monkeypatch)
    monitor = GpuMonitor(index=None, interval=0.005)
    _collect(monitor)

    assert not monitor.available
    assert calls == {"handle": 0, "memory": 0, "util": 0, "decoder": 0}
    assert monitor.snapshot_memory_mb() is None
    assert monitor.peak_memory_mb() is None
    assert monitor.mean_gpu_util() is None
    assert monitor.mean_decoder_util() is None
