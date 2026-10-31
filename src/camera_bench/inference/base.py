"""Interfaces every inference backend implements."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING, ClassVar, Literal

if TYPE_CHECKING:
    import numpy as np

__all__ = ["BackendUnavailableError", "InferenceBackend", "ModelConfig", "Precision", "Session"]

Precision = Literal["fp16", "fp32"]


class BackendUnavailableError(RuntimeError):
    """Raised when a backend cannot run on this machine, e.g. a missing package."""


@dataclass(frozen=True)
class ModelConfig:
    """What to load and how.

    Attributes
    ----------
    model : str
        Backend-specific model name, e.g. ``"yolo11n"`` for Ultralytics.
    input_size : int
        Square network input, px.
    batch : int
        Images per forward pass. Static engines are built for exactly this batch.
    precision : {"fp16", "fp32"}
        Compute precision. FP16 requires a GPU; CPU sessions always run FP32.
    format : str
        Backend-specific runtime format, e.g. ``"engine"`` for TensorRT.
    """

    model: str
    input_size: int
    batch: int
    precision: Precision = "fp16"
    format: str = "engine"

    @property
    def tag(self) -> str:
        """Short unique label, also used for cache file names."""
        name = Path(self.model).stem if self.model.endswith(".onnx") else self.model
        return f"{name}_{self.input_size}_b{self.batch}_{self.precision}"


class Session(ABC):
    """A loaded model ready to be timed.

    Both methods must return only after the work is complete on the device, so that
    wall-clock timing around a call is correct. For CUDA this means synchronizing.

    Parameters
    ----------
    config : ModelConfig
        Configuration the session was loaded with.
    """

    def __init__(self, config: ModelConfig) -> None:
        self.config = config

    @abstractmethod
    def infer(self) -> None:
        """Run one forward pass on a preprocessed batch already on the device.

        This is the quantity published model benchmarks report: no image decoding,
        resizing or box post-processing.
        """

    def predict(self, frames: Sequence[np.ndarray]) -> None:
        """Run the full pipeline on raw frames: preprocess, forward, post-process.

        Parameters
        ----------
        frames : sequence of numpy.ndarray
            ``config.batch`` frames, HxWx3 ``uint8`` in BGR order.

        Raises
        ------
        NotImplementedError
            If the backend measures only the forward pass.
        """
        raise NotImplementedError

    @property
    def supports_pipeline(self) -> bool:
        """Whether :meth:`predict` is implemented."""
        return type(self).predict is not Session.predict

    def close(self) -> None:  # noqa: B027 - optional hook, a no-op by default
        """Release device memory held by the session."""

    def __enter__(self) -> Session:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


class InferenceBackend(ABC):
    """Factory of :class:`Session` objects for one inference framework.

    Parameters
    ----------
    device : str, default "0"
        GPU index as a string, or ``"cpu"``.
    workdir : Path, default ``Path(".cache")``
        Directory for downloaded weights and built engines. Building a TensorRT
        engine takes minutes, so engines are reused between runs.

    Attributes
    ----------
    formats : tuple of str
        Runtime formats the backend accepts in :attr:`ModelConfig.format`.

    Notes
    -----
    To add a backend, subclass this class, implement :meth:`load` and register it::

        from camera_bench.inference import INFERENCE_BACKENDS

        @INFERENCE_BACKENDS.register("onnxruntime")
        class OrtBackend(InferenceBackend):
            formats = ("onnx",)

            def load(self, config):
                ...
    """

    formats: ClassVar[tuple[str, ...]] = ()

    def __init__(self, *, device: str = "0", workdir: Path = Path(".cache")) -> None:
        self.device = device
        self.workdir = workdir

    def check(self) -> None:  # noqa: B027 - optional hook, a no-op by default
        """Verify the backend can run and initialise the device.

        Called once before any :meth:`load`, so that device memory measured after it
        includes the runtime context but no model.

        Raises
        ------
        BackendUnavailableError
            If a required package or device is missing.
        """

    def describe(self) -> str:
        """Return backend settings worth recording next to results, e.g. ``"cuda"``.

        The CLI labels rows ``"<backend>:<describe()>"`` so runs with different
        settings stay distinguishable in one CSV. Empty by default.
        """
        return ""

    def resolve_model(self, name: str) -> str:
        """Map a user-facing model name to the backend's own name.

        Parameters
        ----------
        name : str
            Name from the command line, e.g. a camera-calc id like ``"y11n"``.

        Returns
        -------
        str
            Name passed to :meth:`load` through :attr:`ModelConfig.model`.
        """
        return name

    @abstractmethod
    def load(self, config: ModelConfig) -> Session:
        """Prepare and load a model.

        Parameters
        ----------
        config : ModelConfig
            Model, input size, batch, precision and format.

        Returns
        -------
        Session
            Loaded model; the caller closes it.

        Raises
        ------
        ValueError
            If ``config.format`` is not in :attr:`formats`.
        """
