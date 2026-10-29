"""Interfaces every decoder backend implements."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from types import TracebackType
from typing import ClassVar

from camera_bench.clips import Clip

__all__ = ["BackendUnavailableError", "DecodeBackend", "DecodeStream"]


class BackendUnavailableError(RuntimeError):
    """Raised when a backend cannot run on this machine, e.g. no GPU or no library."""


class DecodeStream(ABC):
    """One decoding session running in the background.

    A stream decodes its clip in a loop as fast as the hardware allows until
    :meth:`close` is called. The runner reads :attr:`frames` twice and divides the
    difference by the elapsed time, so implementations only need a frame counter.
    """

    @property
    @abstractmethod
    def frames(self) -> int:
        """Frames decoded since the stream was opened."""

    @property
    def error(self) -> str | None:
        """Error message if the stream stopped on its own, otherwise ``None``."""
        return None

    @abstractmethod
    def close(self) -> None:
        """Stop decoding and release resources. Must be safe to call twice."""

    def __enter__(self) -> DecodeStream:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


class DecodeBackend(ABC):
    """Factory of :class:`DecodeStream` objects for one decoding method.

    Parameters
    ----------
    device : int, default 0
        GPU index for hardware backends; ignored by CPU backends.

    Attributes
    ----------
    hardware : bool
        Whether the backend uses NVDEC. The runner reads NVDEC utilization only
        for hardware backends and warns when it stays at zero.

    Notes
    -----
    To add a backend, subclass this class, implement :meth:`open` and register it::

        from camera_bench.decode import DECODE_BACKENDS

        @DECODE_BACKENDS.register("pynvvideocodec")
        class PyNvBackend(DecodeBackend):
            hardware = True

            def open(self, clip, clips_dir):
                ...
    """

    hardware: ClassVar[bool] = False

    def __init__(self, *, device: int = 0) -> None:
        self.device = device

    def check(self) -> None:  # noqa: B027 - optional hook, a no-op by default
        """Verify the backend can run here.

        Raises
        ------
        BackendUnavailableError
            If a required tool, library or device is missing.
        """

    @abstractmethod
    def open(self, clip: Clip, clips_dir: Path) -> DecodeStream:
        """Start decoding ``clip`` in a loop.

        Parameters
        ----------
        clip : Clip
            Clip from the manifest.
        clips_dir : Path
            Directory the clip path is relative to.

        Returns
        -------
        DecodeStream
            Running stream; the caller closes it.
        """
