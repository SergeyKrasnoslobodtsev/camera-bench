"""Thin wrappers around the ``ffmpeg`` and ``ffprobe`` command-line tools."""

from __future__ import annotations

import functools
import json
import shutil
import subprocess
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

from camera_bench.reference import Codec

__all__ = [
    "FFmpegError",
    "VideoInfo",
    "encoder_works",
    "ffmpeg_version",
    "hwaccels",
    "probe_video",
    "require_tool",
    "run_ffmpeg",
]

# ffprobe codec names mapped to the two codecs the calculator knows about.
_CODEC_NAMES: dict[str, Codec] = {"h264": "h264", "hevc": "h265"}


class FFmpegError(RuntimeError):
    """Raised when ``ffmpeg`` or ``ffprobe`` is missing or exits with an error."""


@dataclass(frozen=True)
class VideoInfo:
    """Video stream properties measured from the file.

    Attributes
    ----------
    path : Path
        Probed file.
    codec : str
        ``"h264"`` or ``"h265"`` for the codecs the calculator models, otherwise
        the ffprobe codec name.
    width, height : int
        Frame size, px.
    fps : float
        Average frame rate.
    frames : int
        Number of video packets, one per frame for camera streams.
    duration_s : float
        Duration derived from ``frames / fps``.
    video_bytes : int
        Total size of video packets, container overhead excluded.
    """

    path: Path
    codec: str
    width: int
    height: int
    fps: float
    frames: int
    duration_s: float
    video_bytes: int

    @property
    def bitrate_mbps(self) -> float:
        """Mean video bitrate, Mbit/s."""
        return self.video_bytes * 8 / self.duration_s / 1e6 if self.duration_s else 0.0

    @property
    def bits_per_pixel(self) -> float:
        """Mean bits per pixel, the quantity the calculator calls ``bpp``."""
        pixel_rate = self.width * self.height * self.fps
        return self.bitrate_mbps * 1e6 / pixel_rate if pixel_rate else 0.0


def require_tool(name: str) -> str:
    """Return the full path of an executable on ``PATH``.

    Parameters
    ----------
    name : str
        Executable name, e.g. ``"ffmpeg"``.

    Returns
    -------
    str
        Absolute path to the executable.

    Raises
    ------
    FFmpegError
        If the executable is not found.
    """
    path = shutil.which(name)
    if path is None:
        raise FFmpegError(f"{name} not found on PATH")
    return path


def run_ffmpeg(args: list[str], *, tool: str = "ffmpeg") -> str:
    """Run ``ffmpeg`` or ``ffprobe`` and return its standard output.

    Parameters
    ----------
    args : list of str
        Arguments after the executable name.
    tool : {"ffmpeg", "ffprobe"}, default "ffmpeg"
        Which executable to run.

    Returns
    -------
    str
        Captured standard output.

    Raises
    ------
    FFmpegError
        If the process exits with a non-zero code. The message carries the tail
        of standard error.
    """
    cmd = [require_tool(tool), "-hide_banner", *args]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        tail = "\n".join(proc.stderr.strip().splitlines()[-8:])
        raise FFmpegError(f"{tool} exited with {proc.returncode}:\n{tail}")
    return proc.stdout


@functools.cache
def ffmpeg_version() -> str:
    """Return the first line of ``ffmpeg -version``."""
    return run_ffmpeg(["-version"]).splitlines()[0]


@functools.cache
def hwaccels() -> tuple[str, ...]:
    """Return hardware acceleration methods compiled into ``ffmpeg``."""
    lines = run_ffmpeg(["-hwaccels"]).splitlines()
    return tuple(line.strip() for line in lines[1:] if line.strip())


@functools.cache
def encoder_works(encoder: str) -> bool:
    """Check that an encoder actually runs on this machine.

    ``ffmpeg -encoders`` lists NVENC even without an NVIDIA GPU, so the check
    encodes a few blank frames instead.

    Parameters
    ----------
    encoder : str
        ffmpeg encoder name, e.g. ``"hevc_nvenc"``.

    Returns
    -------
    bool
        ``True`` if a test encode succeeded.
    """
    try:
        run_ffmpeg(
            [
                "-loglevel",
                "error",
                "-f",
                "lavfi",
                "-i",
                "color=c=black:s=640x360:r=25",
                "-frames:v",
                "5",
                "-c:v",
                encoder,
                "-f",
                "null",
                "-",
            ]
        )
    except FFmpegError:
        return False
    return True


def probe_video(path: Path) -> VideoInfo:
    """Measure codec, frame size, frame rate and bitrate of the first video stream.

    Bitrate is computed from packet sizes rather than the container header, so it
    is correct for raw ``.h264``/``.h265`` dumps and for files with audio tracks.

    Parameters
    ----------
    path : Path
        Video file.

    Returns
    -------
    VideoInfo
        Measured properties.

    Raises
    ------
    FFmpegError
        If the file has no video stream or ffprobe fails.

    Notes
    -----
    Every packet is read, so probing a day-long recording takes a while.
    Raw elementary streams carry no timing; ffprobe then reports 25 fps.
    """
    stream_json = run_ffmpeg(
        [
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name,width,height,avg_frame_rate,r_frame_rate",
            "-of",
            "json",
            str(path),
        ],
        tool="ffprobe",
    )
    streams = json.loads(stream_json).get("streams", [])
    if not streams:
        raise FFmpegError(f"no video stream in {path}")
    stream = streams[0]
    fps = _parse_rate(stream.get("avg_frame_rate")) or _parse_rate(stream.get("r_frame_rate"))

    sizes = run_ffmpeg(
        [
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "packet=size",
            "-of",
            "csv=p=0",
            str(path),
        ],
        tool="ffprobe",
    ).split()
    frames = len(sizes)
    codec = stream.get("codec_name", "unknown")
    return VideoInfo(
        path=path,
        codec=_CODEC_NAMES.get(codec, codec),
        width=int(stream.get("width", 0)),
        height=int(stream.get("height", 0)),
        fps=fps,
        frames=frames,
        duration_s=frames / fps if fps else 0.0,
        video_bytes=sum(int(size) for size in sizes if size.isdigit()),
    )


def _parse_rate(value: str | None) -> float:
    """Parse an ffprobe rate such as ``"30000/1001"``; return 0 when unknown."""
    if not value or value in ("0/0", "0"):
        return 0.0
    try:
        return float(Fraction(value))
    except (ValueError, ZeroDivisionError):
        return 0.0
