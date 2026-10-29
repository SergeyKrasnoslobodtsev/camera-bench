"""GPU identification and background sampling through NVML.

NVML ships with the NVIDIA driver and is reached through ``nvidia-ml-py``. On a
machine without an NVIDIA GPU every function here degrades gracefully: device
info is ``None`` and the monitor records nothing.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from types import TracebackType

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
    memory_used_mb : float
        Device memory in use, MiB.
    gpu_util, decoder_util : int
        SM and NVDEC utilization, percent.
    """

    t: float
    memory_used_mb: float
    gpu_util: int
    decoder_util: int


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
    index : int, default 0
        NVML device index.
    interval : float, default 0.2
        Seconds between samples.

    Examples
    --------
    >>> with GpuMonitor() as monitor:   # doctest: +SKIP
    ...     run_workload()
    >>> monitor.peak_memory_mb()         # doctest: +SKIP
    """

    index: int = 0
    interval: float = 0.2
    samples: list[GpuSample] = field(default_factory=list, init=False)
    _since: float = field(default=0.0, init=False)
    _stop: threading.Event = field(default_factory=threading.Event, init=False)
    _thread: threading.Thread | None = field(default=None, init=False)

    @property
    def available(self) -> bool:
        """Whether NVML telemetry is collected."""
        return _NVML

    def __enter__(self) -> GpuMonitor:
        if _NVML:
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(self.index)
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
        if not _NVML:
            return None
        handle = pynvml.nvmlDeviceGetHandleByIndex(self.index)
        return pynvml.nvmlDeviceGetMemoryInfo(handle).used / 2**20

    def _window(self) -> list[GpuSample]:
        return [sample for sample in self.samples if sample.t >= self._since]

    def peak_memory_mb(self) -> float | None:
        """Highest memory use in the current window, MiB."""
        window = self._window()
        return max(sample.memory_used_mb for sample in window) if window else None

    def mean_decoder_util(self) -> float | None:
        """Mean NVDEC utilization in the current window, percent."""
        window = self._window()
        return sum(s.decoder_util for s in window) / len(window) if window else None

    def mean_gpu_util(self) -> float | None:
        """Mean SM utilization in the current window, percent."""
        window = self._window()
        return sum(s.gpu_util for s in window) / len(window) if window else None

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                memory = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
                util = pynvml.nvmlDeviceGetUtilizationRates(self._handle)
                decoder, _ = pynvml.nvmlDeviceGetDecoderUtilization(self._handle)
            except pynvml.NVMLError:
                break
            self.samples.append(
                GpuSample(time.perf_counter(), memory.used / 2**20, util.gpu, decoder)
            )
            self._stop.wait(self.interval)
