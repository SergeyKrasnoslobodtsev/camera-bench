"""Bitrate of real camera recordings compared with the calculator formula.

The calculator estimates a stream as ``width * height * fps * bpp * scene``. Probing
recordings of real cameras gives the measured ``bpp`` and, divided by the calculator
value, the scene factor to enter in the form for that kind of scene.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

from camera_bench.media import probe_video
from camera_bench.reference import BITS_PER_PIXEL

__all__ = ["VIDEO_SUFFIXES", "BitrateRow", "find_videos", "measure_bitrate"]

#: File suffixes treated as video when a directory is given.
VIDEO_SUFFIXES = frozenset(
    {".mp4", ".mkv", ".mov", ".avi", ".ts", ".m2ts", ".h264", ".264", ".h265", ".265", ".hevc"}
)


@dataclass(frozen=True)
class BitrateRow:
    """Measured stream properties of one recording.

    Attributes
    ----------
    file : str
        Path as given or found.
    codec : str
        ``"h264"``, ``"h265"`` or another ffprobe codec name.
    width, height : int
        Frame size, px.
    fps : float
        Frame rate.
    duration_s : float
        Duration, seconds.
    bitrate_mbps : float
        Mean video bitrate, Mbit/s.
    bpp : float
        Measured bits per pixel.
    calc_bpp : float or None
        Bits per pixel the calculator uses for this codec.
    scene_factor : float or None
        ``bpp / calc_bpp``: the calculator's scene factor that reproduces this file.
    """

    file: str
    codec: str
    width: int
    height: int
    fps: float
    duration_s: float
    bitrate_mbps: float
    bpp: float
    calc_bpp: float | None
    scene_factor: float | None


def find_videos(paths: Iterable[Path]) -> Iterator[Path]:
    """Expand directories into the video files they contain, recursively.

    Parameters
    ----------
    paths : iterable of Path
        Files and directories. Files are yielded as is, whatever their suffix.

    Yields
    ------
    Path
        Video files in sorted order within each directory.
    """
    for path in paths:
        if path.is_dir():
            yield from sorted(
                item
                for item in path.rglob("*")
                if item.is_file() and item.suffix.lower() in VIDEO_SUFFIXES
            )
        else:
            yield path


def measure_bitrate(path: Path) -> BitrateRow:
    """Probe one recording and compare it with the calculator.

    Parameters
    ----------
    path : Path
        Video file.

    Returns
    -------
    BitrateRow
        Measured values; ``calc_bpp`` and ``scene_factor`` are ``None`` for codecs
        the calculator does not model.
    """
    info = probe_video(path)
    calc_bpp = BITS_PER_PIXEL.get(info.codec)  # type: ignore[call-overload]
    bpp = info.bits_per_pixel
    return BitrateRow(
        file=str(path),
        codec=info.codec,
        width=info.width,
        height=info.height,
        fps=round(info.fps, 3),
        duration_s=round(info.duration_s, 1),
        bitrate_mbps=round(info.bitrate_mbps, 3),
        bpp=round(bpp, 4),
        calc_bpp=calc_bpp,
        scene_factor=round(bpp / calc_bpp, 2) if calc_bpp else None,
    )
