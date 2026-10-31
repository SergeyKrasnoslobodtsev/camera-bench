"""ONNX models on ONNX Runtime, the way a production server runs them.

The model is either an ``.onnx`` file (any detector, including custom ones) or an
Ultralytics name such as ``"yolo11n"`` / ``"y11n"``; names are exported to ONNX once
through Ultralytics and cached. Running the benchmark itself needs only
``onnxruntime-gpu``.

Execution providers
-------------------
``cuda``
    CUDA execution provider. With ``precision="fp16"`` the model is exported with
    FP16 weights, because the CUDA provider runs whatever types the graph has.
``tensorrt``
    TensorRT execution provider. The graph stays FP32 and TensorRT is asked for
    FP16 kernels; its engines are cached in ``workdir/trt-ep``.
``cpu``
    CPU execution provider, for reference and for testing without a GPU.
    FP32 only: an fp16 config is rejected with an explicit error.
"""

from __future__ import annotations

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

__all__ = ["OnnxRuntimeBackend", "OnnxRuntimeSession"]

_PROVIDERS: dict[str, str] = {
    "cuda": "CUDAExecutionProvider",
    "tensorrt": "TensorrtExecutionProvider",
    "cpu": "CPUExecutionProvider",
}


class OnnxRuntimeSession(Session):
    """An ``onnxruntime.InferenceSession`` with a fixed input batch.

    Parameters
    ----------
    session : onnxruntime.InferenceSession
        Created session.
    config : ModelConfig
        Configuration the session was created for.
    device_type : {"cuda", "cpu"}
        Where inputs and outputs are bound in :meth:`infer`.
    device_id : int
        GPU index for CUDA bindings.

    Notes
    -----
    :meth:`infer` keeps input and outputs in device memory through IO binding, so it
    times the graph alone. :meth:`predict` resizes and normalises frames on the CPU,
    copies the batch to the device and the raw outputs back, as a server does.
    Box decoding and NMS are model-specific and are not included.
    """

    def __init__(self, session: Any, config: ModelConfig, device_type: str, device_id: int) -> None:
        super().__init__(config)
        import numpy as np
        import onnxruntime as ort

        self._session = session
        self._np = np
        model_input = session.get_inputs()[0]
        self._input_name = model_input.name
        self._dtype = np.float16 if model_input.type == "tensor(float16)" else np.float32
        size = config.input_size
        sample = np.random.default_rng(0).random((config.batch, 3, size, size)).astype(self._dtype)

        self._binding = session.io_binding()
        self._binding.bind_ortvalue_input(
            self._input_name, ort.OrtValue.ortvalue_from_numpy(sample, device_type, device_id)
        )
        for output in session.get_outputs():
            self._binding.bind_output(output.name, device_type, device_id)

    def infer(self) -> None:
        """Run the graph on a batch already in device memory.

        ``run_with_iobinding`` synchronizes the execution provider before returning.
        """
        self._session.run_with_iobinding(self._binding)

    def predict(self, frames: Sequence[np.ndarray]) -> None:
        """Preprocess raw frames on the CPU and run the graph with host copies."""
        import cv2

        np = self._np
        size = self.config.input_size
        batch = np.empty((len(frames), 3, size, size), dtype=self._dtype)
        for index, frame in enumerate(frames):
            resized = cv2.resize(frame, (size, size), interpolation=cv2.INTER_LINEAR)
            rgb = resized[:, :, ::-1].transpose(2, 0, 1)
            batch[index] = rgb.astype(self._dtype) / self._dtype(255)
        self._session.run(None, {self._input_name: batch})

    def close(self) -> None:
        """Release the session and its memory arena."""
        self._binding = None
        self._session = None


class OnnxRuntimeBackend(InferenceBackend):
    """ONNX Runtime with the CUDA, TensorRT or CPU execution provider.

    Parameters
    ----------
    device : str, default "0"
        GPU index as a string, or ``"cpu"`` (forces the CPU provider).
    workdir : Path, default ``Path(".cache")``
        Cache for exported ONNX files and TensorRT provider engines.
    provider : {"cuda", "tensorrt", "cpu"}, default "cuda"
        Execution provider.
    threads : int, default 0
        Intra-op threads for CPU work; 0 lets ONNX Runtime decide.
    cudnn_search : {"EXHAUSTIVE", "HEURISTIC", "DEFAULT"}, default "EXHAUSTIVE"
        cuDNN convolution algorithm search of the CUDA provider. Match production.
    """

    formats: ClassVar[tuple[str, ...]] = ("onnx",)

    def __init__(
        self,
        *,
        device: str = "0",
        workdir: Path = Path(".cache"),
        provider: str = "cuda",
        threads: int | str = 0,
        cudnn_search: str = "EXHAUSTIVE",
    ) -> None:
        super().__init__(device=device, workdir=workdir)
        if provider not in _PROVIDERS:
            raise ValueError(f"provider must be one of {', '.join(_PROVIDERS)}, got {provider!r}")
        self.provider = "cpu" if device == "cpu" else provider
        self.threads = int(threads)
        self.cudnn_search = cudnn_search

    @property
    def _device_id(self) -> int:
        return 0 if self.device == "cpu" else int(self.device)

    def describe(self) -> str:
        """Return the execution provider, e.g. ``"cuda"`` or ``"tensorrt"``."""
        return self.provider

    def check(self) -> None:
        """Import ONNX Runtime and verify the provider is available."""
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise BackendUnavailableError(
                "onnxruntime is not installed; run `uv sync --extra onnx`"
            ) from exc
        # Wheels with the [cuda,cudnn] extras ship CUDA and cuDNN as pip packages.
        if self.provider != "cpu" and hasattr(ort, "preload_dlls"):
            ort.preload_dlls()
        available = ort.get_available_providers()
        if _PROVIDERS[self.provider] not in available:
            raise BackendUnavailableError(
                f"{_PROVIDERS[self.provider]} is not available; ONNX Runtime offers "
                f"{', '.join(available)}. Install onnxruntime-gpu instead of onnxruntime."
            )

    def resolve_model(self, name: str) -> str:
        """Keep ``.onnx`` paths; translate camera-calc ids to Ultralytics names."""
        if name.endswith(".onnx"):
            return name
        catalog = find_model(name)
        return catalog.weights if catalog else name.removesuffix(".pt")

    def load(self, config: ModelConfig) -> OnnxRuntimeSession:
        """Export the model if needed and create an inference session."""
        if config.format not in self.formats:
            raise ValueError(f"format must be one of {self.formats}, got {config.format!r}")
        if config.precision == "fp16" and self.provider == "cpu":
            raise ValueError("fp16 needs a GPU; on CPU pass --precision fp32")
        import onnxruntime as ort

        path = self._onnx_path(config)
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.log_severity_level = 3
        if self.threads:
            options.intra_op_num_threads = self.threads
        session = ort.InferenceSession(
            str(path), sess_options=options, providers=self._providers(config)
        )
        actual = session.get_providers()[0]
        if actual != _PROVIDERS[self.provider]:
            raise BackendUnavailableError(f"session fell back to {actual}")
        device_type = "cpu" if self.provider == "cpu" else "cuda"
        return OnnxRuntimeSession(session, config, device_type, self._device_id)

    def _providers(self, config: ModelConfig) -> list[Any]:
        device_id = self._device_id
        if self.provider == "cpu":
            return ["CPUExecutionProvider"]
        cuda = (
            "CUDAExecutionProvider",
            {"device_id": device_id, "cudnn_conv_algo_search": self.cudnn_search},
        )
        if self.provider == "cuda":
            return [cuda, "CPUExecutionProvider"]
        cache = self.workdir / "trt-ep"
        cache.mkdir(parents=True, exist_ok=True)
        tensorrt = (
            "TensorrtExecutionProvider",
            {
                "device_id": device_id,
                "trt_fp16_enable": config.precision == "fp16",
                "trt_engine_cache_enable": True,
                "trt_engine_cache_path": str(cache),
            },
        )
        return [tensorrt, cuda, "CPUExecutionProvider"]

    def _onnx_path(self, config: ModelConfig) -> Path:
        if config.model.endswith(".onnx"):
            path = Path(config.model)
            if not path.exists():
                raise FileNotFoundError(path)
            return path
        try:
            from camera_bench.inference.ultralytics import download_weights, export_model
        except ImportError as exc:  # pragma: no cover - module itself has no hard deps
            raise BackendUnavailableError(str(exc)) from exc
        try:
            import ultralytics  # noqa: F401
        except ImportError as exc:
            raise BackendUnavailableError(
                f"{config.model!r} is not an .onnx file and exporting it needs "
                "`uv sync --extra ultralytics`"
            ) from exc
        # FP16 weights for the CUDA provider; the TensorRT provider converts FP32 itself.
        fp16 = config.precision == "fp16" and self.provider == "cuda"
        weights = download_weights(config.model, self.workdir)
        export_device = "cpu" if not fp16 else str(self._device_id)
        return export_model(weights, config, self.workdir, device=export_device, fp16=fp16)
