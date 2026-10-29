"""Find how many real-time streams a decoder sustains.

For each clip the runner opens 1, 2, 4, ... parallel streams, each decoding as
fast as it can, and measures the total decoded frame rate. Once adding streams no
longer raises the total, the decoder is saturated; that total is its capacity.

Capacity is reported in two units:

- real-time streams of the clip itself, ``floor(capacity_fps / clip.fps)``;
- 1080p@30 reference streams, the unit camera-calc uses for decoder load.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

from camera_bench.clips import Clip
from camera_bench.decode.base import DecodeBackend, DecodeStream
from camera_bench.gpu import GpuMonitor
from camera_bench.reference import GpuArch, predicted_decode_streams, reference_streams

__all__ = ["CapacityRow", "SweepRow", "measure_capacity", "stream_counts"]

#: Total throughput must grow by at least this share to keep adding streams.
MIN_GAIN = 0.05
#: NVDEC utilization below this, percent, means the hardware decoder was not used.
IDLE_DECODER_UTIL = 5.0


@dataclass(frozen=True)
class SweepRow:
    """Throughput at one stream count.

    Attributes
    ----------
    backend, clip : str
        Backend name and clip label.
    streams : int
        Parallel streams.
    aggregate_fps : float
        Frames per second decoded by all streams together.
    per_stream_fps : float
        ``aggregate_fps / streams``.
    realtime : bool
        Whether every stream kept up with the clip frame rate.
    decoder_util, gpu_util : float or None
        Mean NVDEC and SM utilization, percent.
    memory_mb_per_stream : float or None
        Extra device memory per stream, MiB, including each process's CUDA context.
    """

    backend: str
    clip: str
    streams: int
    aggregate_fps: float
    per_stream_fps: float
    realtime: bool
    decoder_util: float | None
    gpu_util: float | None
    memory_mb_per_stream: float | None


@dataclass(frozen=True)
class CapacityRow:
    """Decoder capacity for one clip.

    Attributes
    ----------
    gpu, backend : str
        Device name and backend name.
    codec : str
        Clip codec.
    width, height : int
        Frame size, px.
    fps : float
        Clip frame rate.
    capacity_fps : float
        Highest total decoded frame rate across the sweep.
    realtime_streams : int
        Streams of this clip the decoder sustains in real time.
    ref_streams : float
        Capacity in 1080p@30 streams.
    predicted_ref_streams : float or None
        What camera-calc assumes for this GPU, when ``arch`` and ``nvdec`` are given.
    measured_to_predicted : float or None
        ``ref_streams / predicted_ref_streams``.
    saturated_at : int
        Stream count that reached ``capacity_fps``.
    memory_mb_per_stream : float or None
        Extra device memory per stream at saturation, MiB.
    note : str
        Warnings, e.g. a software fallback.
    """

    gpu: str
    backend: str
    codec: str
    width: int
    height: int
    fps: float
    capacity_fps: float
    realtime_streams: int
    ref_streams: float
    predicted_ref_streams: float | None
    measured_to_predicted: float | None
    saturated_at: int
    memory_mb_per_stream: float | None
    note: str


def stream_counts(max_streams: int) -> Iterator[int]:
    """Yield 1, 2, 4, ... up to and including ``max_streams``.

    Examples
    --------
    >>> list(stream_counts(12))
    [1, 2, 4, 8, 12]
    """
    count = 1
    while count < max_streams:
        yield count
        count *= 2
    yield max_streams


def _measure_once(
    backend: DecodeBackend,
    backend_name: str,
    clip: Clip,
    clips_dir: Path,
    streams: int,
    warmup_s: float,
    window_s: float,
    device: int,
) -> SweepRow:
    with GpuMonitor(index=device) as monitor:
        baseline_mb = monitor.snapshot_memory_mb()
        opened: list[DecodeStream] = []
        try:
            opened = [backend.open(clip, clips_dir) for _ in range(streams)]
            time.sleep(warmup_s)
            _raise_on_failure(opened)
            start_frames = sum(stream.frames for stream in opened)
            start = time.perf_counter()
            monitor.mark()
            time.sleep(window_s)
            frames = sum(stream.frames for stream in opened) - start_frames
            elapsed = time.perf_counter() - start
            _raise_on_failure(opened)
            peak_mb = monitor.peak_memory_mb()
        finally:
            for stream in opened:
                stream.close()

    aggregate = frames / elapsed
    per_stream = aggregate / streams
    memory = (
        (peak_mb - baseline_mb) / streams
        if peak_mb is not None and baseline_mb is not None
        else None
    )
    return SweepRow(
        backend=backend_name,
        clip=clip.name,
        streams=streams,
        aggregate_fps=round(aggregate, 1),
        per_stream_fps=round(per_stream, 1),
        realtime=per_stream >= clip.fps,
        decoder_util=monitor.mean_decoder_util(),
        gpu_util=monitor.mean_gpu_util(),
        memory_mb_per_stream=round(memory, 1) if memory is not None else None,
    )


def _raise_on_failure(streams: list[DecodeStream]) -> None:
    for stream in streams:
        if stream.error is not None:
            raise RuntimeError(f"decode stream stopped:\n{stream.error}")


def measure_capacity(
    backend: DecodeBackend,
    clip: Clip,
    clips_dir: Path,
    *,
    backend_name: str,
    gpu_name: str,
    max_streams: int = 64,
    warmup_s: float = 3.0,
    window_s: float = 10.0,
    arch: GpuArch | None = None,
    nvdec: int | None = None,
    on_step: Callable[[SweepRow], None] | None = None,
) -> tuple[CapacityRow, list[SweepRow]]:
    """Sweep stream counts for one clip until the decoder saturates.

    Parameters
    ----------
    backend : DecodeBackend
        Backend that opens the streams.
    clip : Clip
        Clip to decode.
    clips_dir : Path
        Directory the clip path is relative to.
    backend_name, gpu_name : str
        Labels copied into the rows.
    max_streams : int, default 64
        Upper bound for the sweep. Each ffmpeg stream holds a CUDA context of a few
        hundred MiB, so very high counts can run out of memory before NVDEC saturates.
    warmup_s : float, default 3.0
        Seconds to let streams start before measuring.
    window_s : float, default 10.0
        Measurement window, seconds.
    arch : {"Turing", "Hopper", "Ampere", "Ada", "Blackwell"}, optional
        GPU architecture for the calculator prediction.
    nvdec : int, optional
        Number of NVDEC engines for the calculator prediction.
    on_step : callable, optional
        Called with each :class:`SweepRow` as soon as it is measured.

    Returns
    -------
    capacity : CapacityRow
        Summary for the clip.
    sweep : list of SweepRow
        Every measured stream count.
    """
    sweep: list[SweepRow] = []
    best: SweepRow | None = None
    for streams in stream_counts(max_streams):
        row = _measure_once(
            backend, backend_name, clip, clips_dir, streams, warmup_s, window_s, backend.device
        )
        sweep.append(row)
        if on_step is not None:
            on_step(row)
        if best is not None and row.aggregate_fps < best.aggregate_fps * (1 + MIN_GAIN):
            if row.aggregate_fps > best.aggregate_fps:
                best = row
            break
        best = row
    assert best is not None

    ref = reference_streams(best.aggregate_fps * clip.width * clip.height)
    predicted = (
        predicted_decode_streams(arch, nvdec, clip.codec)
        if arch is not None and nvdec is not None
        else None
    )
    notes = []
    if backend.hardware and best.decoder_util is not None and best.decoder_util < IDLE_DECODER_UTIL:
        notes.append("NVDEC idle: software fallback?")
    if best.streams == max_streams:
        notes.append("not saturated: raise --max-streams")
    capacity = CapacityRow(
        gpu=gpu_name,
        backend=backend_name,
        codec=clip.codec,
        width=clip.width,
        height=clip.height,
        fps=clip.fps,
        capacity_fps=best.aggregate_fps,
        realtime_streams=math.floor(best.aggregate_fps / clip.fps),
        ref_streams=round(ref, 1),
        predicted_ref_streams=round(predicted, 1) if predicted is not None else None,
        measured_to_predicted=round(ref / predicted, 2) if predicted else None,
        saturated_at=best.streams,
        memory_mb_per_stream=best.memory_mb_per_stream,
        note="; ".join(notes),
    )
    return capacity, sweep
