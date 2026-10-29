"""Decoder throughput benchmark.

Backends are created by name through :data:`DECODE_BACKENDS`::

    backend = DECODE_BACKENDS.create("ffmpeg-cuda", device=0)
"""

from camera_bench.decode.base import BackendUnavailableError, DecodeBackend, DecodeStream
from camera_bench.registry import Registry

__all__ = [
    "DECODE_BACKENDS",
    "BackendUnavailableError",
    "DecodeBackend",
    "DecodeStream",
]

#: Registry of decoder backends; plugins use the ``camera_bench.decode`` entry point group.
DECODE_BACKENDS: Registry[DecodeBackend] = Registry(
    "decode", base=DecodeBackend, group="camera_bench.decode"
)
DECODE_BACKENDS.register("ffmpeg-cuda", "camera_bench.decode.ffmpeg:FFmpegCudaBackend")
DECODE_BACKENDS.register("ffmpeg-cuvid", "camera_bench.decode.ffmpeg:FFmpegCuvidBackend")
DECODE_BACKENDS.register("ffmpeg-cpu", "camera_bench.decode.ffmpeg:FFmpegCpuBackend")
