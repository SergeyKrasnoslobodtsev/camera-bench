"""Decoder backends that run one ``ffmpeg`` process per stream.

Each process has its own CUDA context, so per-stream memory is overstated compared
with a single-process pipeline such as DeepStream, and context switching can cost a
few percent of NVDEC throughput at high stream counts.
"""

from __future__ import annotations

import collections
import subprocess
import threading
from pathlib import Path
from typing import ClassVar

from camera_bench.clips import Clip
from camera_bench.decode.base import BackendUnavailableError, DecodeBackend, DecodeStream
from camera_bench.media import FFmpegError, hwaccels, require_tool, run_ffmpeg

__all__ = ["FFmpegBackend", "FFmpegCpuBackend", "FFmpegCudaBackend", "FFmpegCuvidBackend"]


class FFmpegStream(DecodeStream):
    """An ``ffmpeg`` process that decodes a clip in a loop into the null muxer.

    Progress is read from ``-progress pipe:1``, which prints ``frame=N`` lines
    several times per second.

    Parameters
    ----------
    cmd : list of str
        Full command line.
    """

    def __init__(self, cmd: list[str]) -> None:
        self._frames = 0
        self._stderr: collections.deque[str] = collections.deque(maxlen=8)
        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self._readers = [
            threading.Thread(target=self._read_progress, daemon=True),
            threading.Thread(target=self._read_stderr, daemon=True),
        ]
        for reader in self._readers:
            reader.start()

    def _read_progress(self) -> None:
        assert self._proc.stdout is not None
        for line in self._proc.stdout:
            if line.startswith("frame="):
                value = line[6:].strip()
                if value.isdigit():
                    self._frames = int(value)

    def _read_stderr(self) -> None:
        assert self._proc.stderr is not None
        for line in self._proc.stderr:
            self._stderr.append(line.rstrip())

    @property
    def frames(self) -> int:
        """Frames decoded since the process started."""
        return self._frames

    @property
    def error(self) -> str | None:
        """Last lines of standard error if the process exited."""
        if self._proc.poll() is None:
            return None
        return "\n".join(self._stderr) or f"ffmpeg exited with {self._proc.returncode}"

    def close(self) -> None:
        """Terminate the process and wait for it."""
        if self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait()
        for reader in self._readers:
            reader.join(timeout=1)


class FFmpegBackend(DecodeBackend):
    """Base for ffmpeg backends; subclasses provide the decoder arguments."""

    def check(self) -> None:
        """Verify that ``ffmpeg`` is on ``PATH``."""
        try:
            require_tool("ffmpeg")
        except FFmpegError as exc:
            raise BackendUnavailableError(str(exc)) from exc

    def decoder_args(self, clip: Clip) -> list[str]:
        """Return input options that select the decoder for ``clip``."""
        return []

    def open(self, clip: Clip, clips_dir: Path) -> FFmpegStream:
        """Start an ffmpeg process that decodes ``clip`` endlessly."""
        cmd = [
            require_tool("ffmpeg"),
            "-hide_banner",
            "-nostdin",
            "-loglevel",
            "error",
            *self.decoder_args(clip),
            "-stream_loop",
            "-1",
            "-i",
            str(clips_dir / clip.path),
            "-map",
            "0:v:0",
            "-f",
            "null",
            "-",
            "-nostats",
            "-progress",
            "pipe:1",
            "-stats_period",
            "0.25",
        ]
        return FFmpegStream(cmd)


class FFmpegCpuBackend(FFmpegBackend):
    """Software decoding, one thread per stream by default.

    Parameters
    ----------
    threads : int, default 1
        Decoder threads per stream.
    device : int, default 0
        Ignored.
    """

    def __init__(self, *, threads: int = 1, device: int = 0) -> None:
        super().__init__(device=device)
        self.threads = threads

    def decoder_args(self, clip: Clip) -> list[str]:
        """Limit decoder threads so throughput scales with the number of streams."""
        return ["-threads", str(self.threads)]


class FFmpegCudaBackend(FFmpegBackend):
    """NVDEC through ``-hwaccel cuda``; frames stay in GPU memory.

    Notes
    -----
    If hardware initialisation fails, ffmpeg silently falls back to software
    decoding. :meth:`check` therefore decodes a short test clip and the runner
    warns when NVDEC utilization stays near zero.
    """

    hardware: ClassVar[bool] = True

    def check(self) -> None:
        """Verify that ffmpeg supports CUDA and can open the device."""
        super().check()
        if "cuda" not in hwaccels():
            raise BackendUnavailableError("ffmpeg is built without the cuda hwaccel")
        try:
            run_ffmpeg(
                [
                    "-loglevel",
                    "error",
                    "-init_hw_device",
                    f"cuda=gpu:{self.device}",
                    "-f",
                    "lavfi",
                    "-i",
                    "nullsrc=s=64x64",
                    "-frames:v",
                    "1",
                    "-f",
                    "null",
                    "-",
                ]
            )
        except FFmpegError as exc:
            raise BackendUnavailableError(f"cannot open CUDA device {self.device}: {exc}") from exc

    def decoder_args(self, clip: Clip) -> list[str]:
        """Select the CUDA hwaccel and keep decoded frames on the GPU."""
        return [
            "-hwaccel",
            "cuda",
            "-hwaccel_device",
            str(self.device),
            "-hwaccel_output_format",
            "cuda",
        ]


class FFmpegCuvidBackend(FFmpegCudaBackend):
    """NVDEC through the ``h264_cuvid``/``hevc_cuvid`` decoders.

    Unlike :class:`FFmpegCudaBackend` this path fails loudly instead of falling
    back to software, which makes it a useful cross-check.
    """

    _DECODERS: ClassVar[dict[str, str]] = {"h264": "h264_cuvid", "h265": "hevc_cuvid"}

    def decoder_args(self, clip: Clip) -> list[str]:
        """Select the cuvid decoder matching the clip codec."""
        return ["-c:v", self._DECODERS[clip.codec], "-gpu", str(self.device)]
