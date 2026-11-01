"""GPU identification and background sampling through NVML.

NVML ships with the NVIDIA driver and is reached through ``nvidia-ml-py``. On a
machine without an NVIDIA GPU every function here degrades gracefully: device
info is ``None`` and the monitor records nothing.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from types import TracebackType
from typing import Any

__all__ = ["DeviceInfo", "GpuMonitor", "GpuSample", "device_info"]

try:  # pragma: no cover - depends on the machine
    import pynvml

    pynvml.nvmlInit()
    _NVML = True
except Exception:  # noqa: BLE001 - any NVML failure means "no GPU telemetry"
    pynvml = None
    _NVML = False


@dataclass(frozen=True)
class DeviceInfo:
    """Static properties of one GPU.

    Attributes
    ----------
    index : int
        NVML device index, the same as ``CUDA_VISIBLE_DEVICES`` ordering by PCI bus.
    name : str
        Marketing name, e.g. ``"NVIDIA GeForce RTX 3090 Ti"``.
    driver : str
        Driver version.
    memory_mb : float
        Total memory, MiB.
    """

    index: int
    name: str
    driver: str
    memory_mb: float


@dataclass(frozen=True)
class GpuSample:
    """One telemetry sample.

    Attributes
    ----------
    t : float
        ``time.perf_counter()`` at sampling.
    memory_used_mb : float or None
        Device memory in use, MiB; ``None`` when the NVML query failed.
    gpu_util, decoder_util : int or None
        SM and NVDEC utilization, percent; ``None`` when the query failed.
    """

    t: float
    memory_used_mb: float | None
    gpu_util: int | None
    decoder_util: int | None


def device_info(index: int = 0) -> DeviceInfo | None:
    """Describe a GPU, or return ``None`` when NVML is unavailable.

    Parameters
    ----------
    index : int, default 0
        NVML device index.

    Returns
    -------
    DeviceInfo or None
        Device properties.
    """
    if not _NVML:
        return None
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(index)
        name = pynvml.nvmlDeviceGetName(handle)
        driver = pynvml.nvmlSystemGetDriverVersion()
        memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
    except pynvml.NVMLError:
        return None
    return DeviceInfo(
        index=index,
        name=name.decode() if isinstance(name, bytes) else name,
        driver=driver.decode() if isinstance(driver, bytes) else driver,
        memory_mb=memory.total / 2**20,
    )


@dataclass
class GpuMonitor:
    """Sample memory and utilization in a background thread.

    Use as a context manager around the measured section; :meth:`mark` starts a
    new window so warm-up samples can be dropped.

    Parameters
    ----------
    index : int or None, default 0
        NVML device index; ``None`` disables telemetry entirely, e.g. for CPU runs.
    interval : float, default 0.2
        Seconds between samples.

    Examples
    --------
    >>> with GpuMonitor() as monitor:   # doctest: +SKIP
    ...     run_workload()
    >>> monitor.peak_memory_mb()         # doctest: +SKIP
    """

    index: int | None = 0
    interval: float = 0.2
    samples: list[GpuSample] = field(default_factory=list, init=False)
    _since: float = field(default=0.0, init=False)
    _stop: threading.Event = field(default_factory=threading.Event, init=False)
    _thread: threading.Thread | None = field(default=None, init=False)

    @property
    def available(self) -> bool:
        """Whether NVML telemetry is collected."""
        return _NVML and self.index is not None

    def __enter__(self) -> GpuMonitor:
        if not self.available:
            return self
        try:
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(self.index)
        except pynvml.NVMLError as exc:
            raise ValueError(f"GPU {self.index} is not available: {exc}") from exc
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()

    def mark(self) -> None:
        """Start a new measurement window; earlier samples are ignored by summaries."""
        self._since = time.perf_counter()

    def snapshot_memory_mb(self) -> float | None:
        """Read device memory in use right now, MiB."""
        if not self.available:
            return None
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(self.index)
            return pynvml.nvmlDeviceGetMemoryInfo(handle).used / 2**20
        except pynvml.NVMLError as exc:
            raise ValueError(f"GPU {self.index} is not available: {exc}") from exc

    def _window(self) -> list[GpuSample]:
        return [sample for sample in self.samples if sample.t >= self._since]

    def peak_memory_mb(self) -> float | None:
        """Highest memory use in the current window, MiB."""
        window = [s.memory_used_mb for s in self._window() if s.memory_used_mb is not None]
        return max(window) if window else None

    def mean_decoder_util(self) -> float | None:
        """Mean NVDEC utilization in the current window, percent."""
        window = [s.decoder_util for s in self._window() if s.decoder_util is not None]
        return sum(window) / len(window) if window else None

    def mean_gpu_util(self) -> float | None:
        """Mean SM utilization in the current window, percent."""
        window = [s.gpu_util for s in self._window() if s.gpu_util is not None]
        return sum(window) / len(window) if window else None

    def _loop(self) -> None:
        while not self._stop.is_set():
            sample = GpuSample(
                time.perf_counter(),
                self._query(lambda: pynvml.nvmlDeviceGetMemoryInfo(self._handle).used / 2**20),
                self._query(lambda: pynvml.nvmlDeviceGetUtilizationRates(self._handle).gpu),
                self._query(lambda: pynvml.nvmlDeviceGetDecoderUtilization(self._handle)[0]),
            )
            if (
                sample.memory_used_mb is None
                and sample.gpu_util is None
                and sample.decoder_util is None
            ):
                break  # the handle is dead; nothing left to sample
            self.samples.append(sample)
            self._stop.wait(self.interval)

    @staticmethod
    def _query(call: Callable[[], Any]) -> Any:
        try:
            return call()
        except pynvml.NVMLError:
            return None
