"""Detector throughput benchmark.

Backends are created by name through :data:`INFERENCE_BACKENDS`::

    backend = INFERENCE_BACKENDS.create("ultralytics", device="0")
"""

from camera_bench.inference.base import (
    BackendUnavailableError,
    InferenceBackend,
    ModelConfig,
    Precision,
    Session,
)
from camera_bench.registry import Registry

__all__ = [
    "INFERENCE_BACKENDS",
    "BackendUnavailableError",
    "InferenceBackend",
    "ModelConfig",
    "Precision",
    "Session",
]

#: Registry of inference backends; plugins use the ``camera_bench.inference`` entry point group.
INFERENCE_BACKENDS: Registry[InferenceBackend] = Registry(
    "inference", base=InferenceBackend, group="camera_bench.inference"
)
INFERENCE_BACKENDS.register("ultralytics", "camera_bench.inference.ultralytics:UltralyticsBackend")
INFERENCE_BACKENDS.register("onnxruntime", "camera_bench.inference.onnxruntime:OnnxRuntimeBackend")
