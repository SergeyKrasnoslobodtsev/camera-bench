"""Ultralytics YOLO models on PyTorch, TensorRT or ONNX Runtime.

Weights are downloaded into ``workdir/weights`` and exported artifacts are cached
in ``workdir/engines`` under a name that encodes input size, batch and precision.
TensorRT engines are tied to the GPU, driver and TensorRT version: delete the cache
after upgrading any of them.
"""

from __future__ import annotations

import gc
import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from camera_bench.inference.base import (
    BackendUnavailableError,
    InferenceBackend,
    ModelConfig,
    Session,
)
from camera_bench.reference import find_model

if TYPE_CHECKING:
    import numpy as np

__all__ = ["UltralyticsBackend", "UltralyticsSession", "download_weights", "export_model"]

# Ultralytics export ``format`` names and the suffix of the file each one produces.
_EXPORTS: dict[str, str] = {"engine": ".engine", "onnx": ".onnx"}


def _precision_kwargs(fp16: bool) -> dict[str, Any]:
    """Precision arguments for both old (``half``) and new (``quantize``) Ultralytics."""
    from ultralytics.cfg import DEFAULT_CFG_DICT

    if "quantize" in DEFAULT_CFG_DICT:
        return {"quantize": 16 if fp16 else None}
    return {"half": fp16}


class UltralyticsSession(Session):
    """A YOLO model loaded through ``ultralytics.YOLO``.

    Parameters
    ----------
    model : ultralytics.YOLO
        Loaded model.
    config : ModelConfig
        Configuration the model was loaded with.
    device : str
        ``"0"``-style GPU index or ``"cpu"``.
    """

    def __init__(self, model: Any, config: ModelConfig, device: str) -> None:
        super().__init__(config)
        import torch

        self._torch = torch
        self._model = model
        self._device = device
        self._cuda = device != "cpu"
        self._kwargs = {
            "imgsz": config.input_size,
            "device": device,
            "verbose": False,
            "batch": config.batch,
            **_precision_kwargs(config.precision == "fp16" and self._cuda),
        }
        self._tensor: Any = None

    def _sync(self) -> None:
        if self._cuda:
            self._torch.cuda.synchronize()

    def predict(self, frames: Sequence[np.ndarray]) -> None:
        """Run ``YOLO.predict`` on raw frames, including resize and NMS."""
        self._model.predict(list(frames), **self._kwargs)
        self._sync()

    def infer(self) -> None:
        """Run the bare network on a preprocessed tensor already on the device."""
        if self._tensor is None:
            self._prepare_tensor()
        self._backend(self._tensor)
        self._sync()

    def _prepare_tensor(self) -> None:
        # The predictor builds an AutoBackend on first use; reuse it for bare forwards.
        if getattr(self._model, "predictor", None) is None:
            import numpy as np

            size = self.config.input_size
            blank = np.zeros((size, size, 3), dtype=np.uint8)
            self.predict([blank] * self.config.batch)
        self._backend = self._model.predictor.model
        device = self._backend.device
        dtype = (
            self._torch.float16 if getattr(self._backend, "fp16", False) else self._torch.float32
        )
        size = self.config.input_size
        self._tensor = self._torch.rand(
            (self.config.batch, 3, size, size), device=device, dtype=dtype
        )

    def close(self) -> None:
        """Drop the model and return cached CUDA memory to the driver."""
        self._model = None
        self._backend = None
        self._tensor = None
        gc.collect()
        if self._cuda:
            self._torch.cuda.empty_cache()


class UltralyticsBackend(InferenceBackend):
    """Ultralytics YOLO detectors and pose models.

    Model names are Ultralytics weights names (``"yolo11n"``, ``"yolo26s-pose"``) or
    camera-calc ids (``"y11n"``, ``"y26sp"``).

    Parameters
    ----------
    device : str, default "0"
        GPU index as a string, or ``"cpu"``.
    workdir : Path, default ``Path(".cache")``
        Cache for weights and exported engines.

    Notes
    -----
    Formats:

    - ``"engine"`` -- TensorRT, static input and batch; needs the ``tensorrt`` extra.
    - ``"onnx"`` -- ONNX Runtime; needs ``onnxruntime-gpu`` for CUDA.
    - ``"torch"`` -- the ``.pt`` checkpoint in PyTorch, no export.
    """

    formats: ClassVar[tuple[str, ...]] = ("engine", "onnx", "torch")

    def check(self) -> None:
        """Import Ultralytics and create the CUDA context."""
        try:
            import torch
            import ultralytics  # noqa: F401
        except ImportError as exc:
            raise BackendUnavailableError(
                "ultralytics is not installed; run `uv sync --extra ultralytics`"
            ) from exc
        if self.device != "cpu":
            if not torch.cuda.is_available():
                raise BackendUnavailableError(
                    "torch has no CUDA; on Windows reinstall with `uv sync --extra ultralytics`"
                    " so torch comes from the PyTorch CUDA index"
                )
            torch.zeros(1, device=f"cuda:{self.device}")

    def resolve_model(self, name: str) -> str:
        """Translate camera-calc ids to weights names; other names pass through."""
        catalog = find_model(name)
        return catalog.weights if catalog else name.removesuffix(".pt")

    def load(self, config: ModelConfig) -> UltralyticsSession:
        """Download weights, export if needed and load the model."""
        if config.format not in self.formats:
            raise ValueError(f"format must be one of {self.formats}, got {config.format!r}")
        if config.format == "engine" and self.device == "cpu":
            raise ValueError("TensorRT engines need a GPU")
        if config.precision == "fp16" and self.device == "cpu":
            raise ValueError("fp16 needs a GPU; on CPU pass --precision fp32")
        from ultralytics import YOLO

        weights = download_weights(config.model, self.workdir)
        artifact = (
            weights
            if config.format == "torch"
            else export_model(weights, config, self.workdir, device=self.device)
        )
        return UltralyticsSession(YOLO(str(artifact)), config, self.device)


def download_weights(name: str, workdir: Path) -> Path:
    """Return the path to official Ultralytics weights, downloading them if needed.

    Parameters
    ----------
    name : str
        Weights name without extension, e.g. ``"yolo11n"``.
    workdir : Path
        Cache directory; weights go to ``workdir / "weights"``.

    Returns
    -------
    Path
        Local ``.pt`` file.
    """
    path = workdir / "weights" / f"{name}.pt"
    if not path.exists():
        from ultralytics import YOLO

        path.parent.mkdir(parents=True, exist_ok=True)
        YOLO(str(path))  # downloads official weights to exactly this path
    return path


def export_model(
    weights: Path,
    config: ModelConfig,
    workdir: Path,
    *,
    device: str,
    fp16: bool | None = None,
) -> Path:
    """Export weights with static input and batch, reusing a cached artifact.

    Parameters
    ----------
    weights : Path
        ``.pt`` checkpoint.
    config : ModelConfig
        Input size, batch and target format (``"engine"`` or ``"onnx"``).
    workdir : Path
        Cache directory; artifacts go to ``workdir / "engines"``.
    device : str
        Device used for export. FP16 export requires a GPU.
    fp16 : bool, optional
        Store weights in FP16. Defaults to ``config.precision == "fp16"``.

    Returns
    -------
    Path
        Exported file named after :attr:`ModelConfig.tag` and the precision stored.
    """
    if fp16 is None:
        fp16 = config.precision == "fp16"
    stored = "fp16" if fp16 else "fp32"
    tag = f"{config.model}_{config.input_size}_b{config.batch}_{stored}"
    cached = workdir / "engines" / f"{tag}{_EXPORTS[config.format]}"
    if cached.exists():
        return cached
    from ultralytics import YOLO

    exported = YOLO(str(weights)).export(
        format=config.format,
        imgsz=config.input_size,
        batch=config.batch,
        dynamic=False,
        device=device,
        verbose=False,
        **_precision_kwargs(fp16),
    )
    cached.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(exported), cached)
    return cached
