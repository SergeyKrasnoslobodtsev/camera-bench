"""Constants and formulas mirrored from camera-calc.

Every benchmark reports what the calculator predicts next to what was measured, so
the two can be compared row by row. Values here must match the calculator sources:

- ``src/domain/references/resolutions.ts`` -- :data:`RESOLUTIONS`
- ``src/domain/references/models.ts`` -- :data:`CATALOG_MODELS`, :data:`T4_TFLOPS`
- ``src/domain/references/gpu-catalog.ts`` -- :data:`NVDEC_FPS`
- ``src/domain/compute-type/stream.ts`` -- :data:`BITS_PER_PIXEL`, :func:`stream_bitrate_mbps`
- ``src/domain/compute-system/gpu-load.ts`` -- :func:`predicted_model_fps`

When the calculator changes, update this module and the tests in
``tests/test_reference.py`` together.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

__all__ = [
    "BITS_PER_PIXEL",
    "CATALOG_MODELS",
    "NVDEC_FPS",
    "REFERENCE_PIXEL_RATE",
    "RESOLUTIONS",
    "T4_TFLOPS",
    "CatalogModel",
    "Codec",
    "Resolution",
    "find_model",
    "parse_resolution",
    "predicted_decode_streams",
    "predicted_model_fps",
    "reference_streams",
    "stream_bitrate_mbps",
]

Codec = Literal["h264", "h265"]
GpuArch = Literal["Turing", "Hopper", "Ampere", "Ada", "Blackwell"]


@dataclass(frozen=True)
class Resolution:
    """Camera stream resolution.

    Attributes
    ----------
    width, height : int
        Frame size in pixels.
    megapixels : int
        Marketing megapixel count used as a short label, e.g. ``4`` for 2560x1440.
    """

    width: int
    height: int
    megapixels: int

    @property
    def pixels(self) -> int:
        """Number of pixels in one frame."""
        return self.width * self.height

    @property
    def label(self) -> str:
        """Short label such as ``"2560x1440"``."""
        return f"{self.width}x{self.height}"


@dataclass(frozen=True)
class CatalogModel:
    """Detector from the calculator catalog.

    Attributes
    ----------
    id : str
        Calculator identifier, e.g. ``"y11n"``.
    weights : str
        Ultralytics weights name without extension, e.g. ``"yolo11n"``.
    gflops : float
        Model GFLOPs at 640 px input.
    latency_ms : float
        Ultralytics latency on T4, TensorRT FP16, batch 1, 640 px input.
    """

    id: str
    weights: str
    gflops: float
    latency_ms: float


# Stream resolutions, ascending by frame width.
RESOLUTIONS: tuple[Resolution, ...] = (
    Resolution(1920, 1080, 2),
    Resolution(2560, 1440, 4),
    Resolution(2592, 1944, 5),
    Resolution(2880, 1620, 5),
    Resolution(3072, 2048, 6),
    Resolution(3840, 2160, 8),
    Resolution(4000, 3000, 12),
)

# GFLOPs and T4 TensorRT FP16 latency at batch 1 and 640 px (docs.ultralytics.com).
CATALOG_MODELS: tuple[CatalogModel, ...] = (
    CatalogModel("y26n", "yolo26n", 5.5, 1.7),
    CatalogModel("y26s", "yolo26s", 20.9, 2.5),
    CatalogModel("y26m", "yolo26m", 68.4, 4.7),
    CatalogModel("y26l", "yolo26l", 86.8, 6.2),
    CatalogModel("y26x", "yolo26x", 194.4, 11.8),
    CatalogModel("y26np", "yolo26n-pose", 7.6, 1.8),
    CatalogModel("y26sp", "yolo26s-pose", 24.1, 2.7),
    CatalogModel("y26mp", "yolo26m-pose", 73.3, 5.0),
    CatalogModel("y26lp", "yolo26l-pose", 91.7, 6.5),
    CatalogModel("y26xp", "yolo26x-pose", 202.3, 12.2),
    CatalogModel("y11n", "yolo11n", 6.5, 1.5),
    CatalogModel("y11s", "yolo11s", 21.6, 2.5),
    CatalogModel("y11m", "yolo11m", 68.1, 4.7),
    CatalogModel("y11l", "yolo11l", 87.2, 6.2),
    CatalogModel("y11x", "yolo11x", 195.3, 11.3),
)

#: FP16 Tensor TFLOPS of the T4 the catalog latencies were measured on.
T4_TFLOPS = 65.0
#: Input size the catalog latencies were measured at, px.
CATALOG_INPUT_PX = 640

#: Bits per pixel of a camera stream, chosen by codec.
BITS_PER_PIXEL: dict[Codec, float] = {"h264": 0.07, "h265": 0.04}

#: The decoder reference stream is 1080p at 30 fps.
REFERENCE_PIXEL_RATE = 1920 * 1080 * 30

# 1080p frames per second for one NVDEC engine (NVIDIA Video Codec SDK 13.0,
# NVDEC Application Note). Hopper is assumed equal to Turing.
NVDEC_FPS: dict[GpuArch, dict[Codec, float]] = {
    "Turing": {"h264": 771, "h265": 1316},
    "Hopper": {"h264": 771, "h265": 1316},
    "Ampere": {"h264": 748, "h265": 1415},
    "Ada": {"h264": 903, "h265": 1641},
    "Blackwell": {"h264": 2172, "h265": 1872},
}

_RESOLUTION_RE = re.compile(r"^\s*(\d+)\s*[xX×]\s*(\d+)\s*$")
_MEGAPIXEL_RE = re.compile(r"^\s*(\d+)\s*mp\s*$", re.IGNORECASE)


def parse_resolution(text: str) -> Resolution:
    """Parse ``"2560x1440"`` or a megapixel label such as ``"4mp"``.

    Parameters
    ----------
    text : str
        Explicit frame size or megapixel label. Megapixel labels pick the first
        calculator resolution with that count, so ``"5mp"`` is 2592x1944.

    Returns
    -------
    Resolution
        Matching calculator resolution, or an ad-hoc one for an explicit size.

    Raises
    ------
    ValueError
        If the text is neither form, or no calculator resolution has that count.

    Examples
    --------
    >>> parse_resolution("4mp").label
    '2560x1440'
    >>> parse_resolution("1280x720").megapixels
    1
    """
    if match := _RESOLUTION_RE.match(text):
        width, height = int(match[1]), int(match[2])
        for known in RESOLUTIONS:
            if (known.width, known.height) == (width, height):
                return known
        return Resolution(width, height, max(1, round(width * height / 1e6)))
    if match := _MEGAPIXEL_RE.match(text):
        megapixels = int(match[1])
        for known in RESOLUTIONS:
            if known.megapixels == megapixels:
                return known
        raise ValueError(f"no calculator resolution with {megapixels} MP")
    raise ValueError(f"expected WIDTHxHEIGHT or <N>mp, got {text!r}")


def find_model(name: str) -> CatalogModel | None:
    """Find a catalog model by calculator id or weights name.

    Parameters
    ----------
    name : str
        ``"y11n"``, ``"yolo11n"`` or ``"yolo11n.pt"``.

    Returns
    -------
    CatalogModel or None
        The model, or ``None`` when it is not in the calculator catalog.
    """
    stem = name.removesuffix(".pt")
    for model in CATALOG_MODELS:
        if stem in (model.id, model.weights):
            return model
    return None


def stream_bitrate_mbps(
    width: int, height: int, fps: float, codec: Codec, scene_factor: float = 1.0
) -> float:
    """Estimate camera bitrate the way the calculator does.

    Parameters
    ----------
    width, height : int
        Frame size, px.
    fps : float
        Stream frame rate.
    codec : {"h264", "h265"}
        Stream codec, selects bits per pixel from :data:`BITS_PER_PIXEL`.
    scene_factor : float, default 1.0
        Scene complexity multiplier from the calculator form.

    Returns
    -------
    float
        Bitrate in Mbit/s, ``width * height * fps * bpp * scene_factor / 1e6``.
    """
    return width * height * fps * BITS_PER_PIXEL[codec] * scene_factor / 1e6


def reference_streams(pixel_rate: float) -> float:
    """Express a decoded pixel rate in 1080p@30 reference streams."""
    return pixel_rate / REFERENCE_PIXEL_RATE


def predicted_decode_streams(arch: GpuArch, nvdec: int, codec: Codec) -> float:
    """Return the decoder capacity the calculator assumes, in 1080p@30 streams.

    Parameters
    ----------
    arch : {"Turing", "Hopper", "Ampere", "Ada", "Blackwell"}
        GPU architecture, selects the per-engine rate from :data:`NVDEC_FPS`.
    nvdec : int
        Number of NVDEC engines on the card.
    codec : {"h264", "h265"}
        Stream codec.

    Returns
    -------
    float
        ``nvdec * NVDEC_FPS[arch][codec] / 30``.
    """
    return nvdec * NVDEC_FPS[arch][codec] / 30


def predicted_model_fps(model: CatalogModel, input_size: int, tflops: float) -> float:
    """Return the model throughput the calculator assumes, before its efficiency factor.

    Parameters
    ----------
    model : CatalogModel
        Catalog detector.
    input_size : int
        Square model input, px.
    tflops : float
        FP16 Tensor dense TFLOPS of the target GPU. For GeForce cards the
        calculator uses the figure with FP32 accumulation.

    Returns
    -------
    float
        Images per second: ``1000 / latency * (640 / input)^2 * tflops / 65``.
        The calculator ignores batch size, so measured/predicted at batch > 1
        shows how much batching adds.
    """
    input_scale = (CATALOG_INPUT_PX / input_size) ** 2
    return 1000 / model.latency_ms * input_scale * tflops / T4_TFLOPS
