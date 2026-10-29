"""Time inference sessions and compare them with camera-calc.

Two modes are measured for every configuration:

``engine``
    Bare forward pass on a preprocessed batch already on the GPU. This is what the
    Ultralytics latency table reports and what the calculator scales.
``pipeline``
    Raw camera frames in, boxes out: resize and normalisation, forward pass and
    NMS. This is closer to what a video analytics server actually spends.
"""

from __future__ import annotations

import statistics
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

from camera_bench.gpu import GpuMonitor
from camera_bench.inference.base import InferenceBackend, ModelConfig, Session
from camera_bench.reference import find_model, predicted_model_fps

__all__ = ["InferRow", "Mode", "engine_memory_mb", "measure_session", "time_calls"]

Mode = Literal["engine", "pipeline"]

# camera-calc engine memory model (src/domain/compute-system/gpu-load.ts).
_ENGINE_BASE_GB = 0.15
_ENGINE_BATCH = 16
_BYTES_PER_INPUT_PIXEL = 6
_ENGINE_GFLOPS_DIVISOR = 4


@dataclass(frozen=True)
class InferRow:
    """Timing of one model configuration in one mode.

    Attributes
    ----------
    gpu, backend : str
        Device and backend names.
    model : str
        Backend model name.
    calc_id : str or None
        camera-calc catalog id, when the model is in the catalog.
    format, precision : str
        Runtime format and precision.
    input_size, batch : int
        Network input, px, and images per pass.
    mode : {"engine", "pipeline"}
        What was timed, see the module docstring.
    latency_ms, latency_p95_ms : float
        Median and 95th percentile time of one batch, ms.
    fps : float
        Images per second, ``batch * calls / total time``.
    memory_mb : float or None
        Device memory held by the loaded model after warm-up, MiB.
    calc_memory_mb : float or None
        camera-calc's estimate for this engine. The calculator always assumes
        batch 16, so compare it with batch-16 rows.
    predicted_fps : float or None
        camera-calc's throughput for this GPU before its efficiency factor;
        needs ``tflops``.
    measured_to_predicted : float or None
        ``fps / predicted_fps``: the efficiency factor this measurement implies.
    """

    gpu: str
    backend: str
    model: str
    calc_id: str | None
    format: str
    precision: str
    input_size: int
    batch: int
    mode: Mode
    latency_ms: float
    latency_p95_ms: float
    fps: float
    memory_mb: float | None
    calc_memory_mb: float | None
    predicted_fps: float | None
    measured_to_predicted: float | None


def engine_memory_mb(gflops: float, input_size: int) -> float:
    """Estimate engine memory the way camera-calc does, MiB.

    Parameters
    ----------
    gflops : float
        Model GFLOPs at 640 px.
    input_size : int
        Network input, px.

    Returns
    -------
    float
        ``(0.15 + 16 * input^2 * 6 * gflops / 4 / 1e9) GB`` expressed in MiB.
    """
    gigabytes = (
        _ENGINE_BASE_GB
        + (_ENGINE_BATCH * input_size**2 * _BYTES_PER_INPUT_PIXEL * gflops / _ENGINE_GFLOPS_DIVISOR)
        / 1e9
    )
    return gigabytes * 1e9 / 2**20


def time_calls(call: Callable[[], None], *, warmup: int, iters: int) -> list[float]:
    """Time a blocking call repeatedly.

    Parameters
    ----------
    call : callable
        Function that returns once the work is finished.
    warmup : int
        Untimed calls first, for allocator, clocks and lazy initialisation.
    iters : int
        Timed calls.

    Returns
    -------
    list of float
        Duration of every timed call, seconds.
    """
    for _ in range(warmup):
        call()
    durations = []
    for _ in range(iters):
        start = time.perf_counter()
        call()
        durations.append(time.perf_counter() - start)
    return durations


def _percentile(values: Sequence[float], q: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(q * (len(ordered) - 1))))
    return ordered[index]


def measure_session(
    backend: InferenceBackend,
    config: ModelConfig,
    *,
    backend_name: str,
    gpu_name: str,
    modes: Sequence[Mode] = ("engine", "pipeline"),
    frames: Sequence[object] | None = None,
    warmup: int = 20,
    iters: int = 200,
    tflops: float | None = None,
    device_index: int = 0,
) -> list[InferRow]:
    """Load one configuration and time it in the requested modes.

    Parameters
    ----------
    backend : InferenceBackend
        Backend that loads the model; :meth:`InferenceBackend.check` must have run.
    config : ModelConfig
        Model configuration.
    backend_name, gpu_name : str
        Labels copied into the rows.
    modes : sequence of {"engine", "pipeline"}, default both
        What to time. ``pipeline`` is skipped for sessions that do not support it.
    frames : sequence of numpy.ndarray, optional
        Raw frames for ``pipeline``; required when that mode is requested.
    warmup, iters : int
        Untimed and timed calls per mode.
    tflops : float, optional
        FP16 Tensor dense TFLOPS of this GPU, enables the calculator prediction.
    device_index : int, default 0
        NVML index of the GPU, for memory readings.

    Returns
    -------
    list of InferRow
        One row per measured mode.
    """
    catalog = find_model(config.model)
    predicted = (
        predicted_model_fps(catalog, config.input_size, tflops) if catalog and tflops else None
    )
    calc_memory = engine_memory_mb(catalog.gflops, config.input_size) if catalog else None

    rows: list[InferRow] = []
    with GpuMonitor(index=device_index) as monitor:
        baseline = monitor.snapshot_memory_mb()
        session: Session = backend.load(config)
        with session:
            for mode in modes:
                if mode == "pipeline":
                    if not session.supports_pipeline:
                        continue
                    if frames is None:
                        raise ValueError("pipeline mode needs frames")
                    batch = list(frames[: config.batch])

                    def call(batch: list[object] = batch) -> None:
                        session.predict(batch)  # type: ignore[arg-type]
                else:
                    call = session.infer
                durations = time_calls(call, warmup=warmup, iters=iters)
                used = monitor.snapshot_memory_mb()
                fps = config.batch * len(durations) / sum(durations)
                rows.append(
                    InferRow(
                        gpu=gpu_name,
                        backend=backend_name,
                        model=config.model,
                        calc_id=catalog.id if catalog else None,
                        format=config.format,
                        precision=config.precision,
                        input_size=config.input_size,
                        batch=config.batch,
                        mode=mode,
                        latency_ms=round(statistics.median(durations) * 1e3, 3),
                        latency_p95_ms=round(_percentile(durations, 0.95) * 1e3, 3),
                        fps=round(fps, 1),
                        memory_mb=round(used - baseline, 1)
                        if used is not None and baseline is not None
                        else None,
                        calc_memory_mb=round(calc_memory, 1) if calc_memory else None,
                        predicted_fps=round(predicted, 1) if predicted else None,
                        measured_to_predicted=round(fps / predicted, 2) if predicted else None,
                    )
                )
    return rows
