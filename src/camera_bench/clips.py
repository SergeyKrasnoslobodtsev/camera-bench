"""Camera-like test clips for the decoder benchmark.

Video from the web is encoded with long GOPs, B-frames and two-pass rate control,
none of which an IP camera does. :func:`prepare_clips` re-encodes any source (or a
synthetic pattern) the way cameras typically stream: fixed GOP, no B-frames,
constant bitrate taken from the calculator formula.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from camera_bench.media import encoder_works, probe_video, run_ffmpeg
from camera_bench.reference import Codec, Resolution, stream_bitrate_mbps

__all__ = ["MANIFEST_NAME", "Clip", "EncoderFamily", "load_manifest", "prepare_clips"]

MANIFEST_NAME = "manifest.json"

EncoderFamily = Literal["auto", "nvenc", "cpu"]

_ENCODERS: dict[str, dict[Codec, str]] = {
    "nvenc": {"h264": "h264_nvenc", "h265": "hevc_nvenc"},
    "cpu": {"h264": "libx264", "h265": "libx265"},
}


@dataclass(frozen=True)
class Clip:
    """A prepared clip and its measured properties.

    Attributes
    ----------
    path : str
        File name relative to the manifest directory.
    codec : {"h264", "h265"}
        Stream codec.
    width, height : int
        Frame size, px.
    fps : float
        Frame rate.
    frames : int
        Number of frames in the file.
    gop : int
        Keyframe interval, frames.
    encoder : str
        ffmpeg encoder that produced the clip.
    target_mbps : float
        Bitrate requested from the encoder (the calculator estimate).
    bitrate_mbps : float
        Bitrate measured from the packets.
    """

    path: str
    codec: Codec
    width: int
    height: int
    fps: float
    frames: int
    gop: int
    encoder: str
    target_mbps: float
    bitrate_mbps: float

    @property
    def name(self) -> str:
        """Short label, e.g. ``"h265 2560x1440@25"``."""
        return f"{self.codec} {self.width}x{self.height}@{self.fps:g}"

    @property
    def pixel_rate(self) -> float:
        """Pixels per second of one real-time stream."""
        return self.width * self.height * self.fps


def _encoder_args(encoder: str, codec: Codec, gop: int, mbps: float) -> list[str]:
    """Build encoder options that mimic a camera stream: CBR, fixed GOP, no B-frames."""
    rate = f"{mbps * 1000:.0f}k"
    common = ["-b:v", rate, "-maxrate", rate, "-bufsize", f"{mbps * 2000:.0f}k", "-g", str(gop)]
    if encoder.endswith("_nvenc"):
        return [
            "-c:v",
            encoder,
            "-preset",
            "p4",
            "-rc",
            "cbr",
            "-bf",
            "0",
            "-forced-idr",
            "1",
            *common,
        ]
    if encoder == "libx264":
        return [
            "-c:v",
            encoder,
            "-preset",
            "fast",
            "-profile:v",
            "main",
            "-bf",
            "0",
            "-sc_threshold",
            "0",
            "-keyint_min",
            str(gop),
            *common,
        ]
    # libx265 takes GOP structure through its own parameter string.
    params = f"keyint={gop}:min-keyint={gop}:scenecut=0:bframes=0:log-level=error"
    return ["-c:v", encoder, "-preset", "fast", "-x265-params", params, "-tag:v", "hvc1", *common]


def _pick_encoder(family: EncoderFamily, codec: Codec) -> str:
    if family != "auto":
        return _ENCODERS[family][codec]
    nvenc = _ENCODERS["nvenc"][codec]
    return nvenc if encoder_works(nvenc) else _ENCODERS["cpu"][codec]


def prepare_clips(
    out_dir: Path,
    resolutions: Iterable[Resolution],
    codecs: Iterable[Codec],
    *,
    source: Path | None = None,
    fps: float = 25,
    seconds: float = 20,
    gop_seconds: float = 2,
    scene_factor: float = 1.0,
    encoder: EncoderFamily = "auto",
    on_clip: Callable[[Clip], None] | None = None,
) -> list[Clip]:
    """Encode one clip per resolution and codec and write a manifest.

    Parameters
    ----------
    out_dir : Path
        Directory for the clips and ``manifest.json``.
    resolutions : iterable of Resolution
        Frame sizes to produce. The source is scaled to cover the frame and
        center-cropped, so 4:3 sizes do not get black bars.
    codecs : iterable of {"h264", "h265"}
        Codecs to produce.
    source : Path, optional
        Source video, looped if shorter than ``seconds``. Without it a noisy test
        pattern is used, which is fine for decoder load but not for bitrate studies.
    fps : float, default 25
        Output frame rate.
    seconds : float, default 20
        Clip duration. The decoder benchmark loops clips, so length only affects
        how representative the content is.
    gop_seconds : float, default 2
        Keyframe interval in seconds, typical for cameras.
    scene_factor : float, default 1.0
        Multiplier on the calculator bitrate, as in the calculator form.
    encoder : {"auto", "nvenc", "cpu"}, default "auto"
        Encoder family. ``"auto"`` uses NVENC when it works on this machine.
    on_clip : callable, optional
        Called with each finished :class:`Clip`, e.g. to report progress.

    Returns
    -------
    list of Clip
        Prepared clips, also stored in ``out_dir / "manifest.json"``.

    Raises
    ------
    FFmpegError
        If an encode fails.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    gop = max(1, round(fps * gop_seconds))
    clips: list[Clip] = []
    for resolution in resolutions:
        for codec in codecs:
            clip = _encode_one(
                out_dir,
                resolution,
                codec,
                source,
                fps,
                seconds,
                gop,
                scene_factor,
                _pick_encoder(encoder, codec),
            )
            clips.append(clip)
            if on_clip is not None:
                on_clip(clip)
    _write_manifest(out_dir, clips)
    return clips


def _encode_one(
    out_dir: Path,
    resolution: Resolution,
    codec: Codec,
    source: Path | None,
    fps: float,
    seconds: float,
    gop: int,
    scene_factor: float,
    encoder: str,
) -> Clip:
    width, height = resolution.width, resolution.height
    target = stream_bitrate_mbps(width, height, fps, codec, scene_factor)
    out_path = out_dir / f"{codec}_{width}x{height}_{fps:g}fps.mp4"

    if source is None:
        # Temporal noise keeps every frame different, like sensor noise on a real camera.
        pattern = f"testsrc2=size={width}x{height}:rate={fps:g},noise=alls=16:allf=t"
        inputs = ["-f", "lavfi", "-i", pattern]
        video_filter = "format=yuv420p"
    else:
        inputs = ["-stream_loop", "-1", "-i", str(source)]
        video_filter = (
            f"scale={width}:{height}:force_original_aspect_ratio=increase,"
            f"crop={width}:{height},fps={fps:g},format=yuv420p"
        )
    run_ffmpeg(
        [
            "-y",
            "-loglevel",
            "error",
            *inputs,
            "-t",
            f"{seconds:g}",
            "-an",
            "-vf",
            video_filter,
            *_encoder_args(encoder, codec, gop, target),
            str(out_path),
        ]
    )
    info = probe_video(out_path)
    return Clip(
        path=out_path.name,
        codec=codec,
        width=width,
        height=height,
        fps=fps,
        frames=info.frames,
        gop=gop,
        encoder=encoder,
        target_mbps=round(target, 3),
        bitrate_mbps=round(info.bitrate_mbps, 3),
    )


def _write_manifest(out_dir: Path, clips: list[Clip]) -> None:
    # Merge with an existing manifest so clips can be prepared in several runs.
    known = (
        {clip.path: clip for clip in load_manifest(out_dir)}
        if (out_dir / MANIFEST_NAME).exists()
        else {}
    )
    known.update({clip.path: clip for clip in clips})
    payload = [asdict(clip) for clip in sorted(known.values(), key=lambda c: (c.width, c.codec))]
    (out_dir / MANIFEST_NAME).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_manifest(clips_dir: Path) -> list[Clip]:
    """Read clips listed in ``clips_dir / "manifest.json"``.

    Parameters
    ----------
    clips_dir : Path
        Directory written by :func:`prepare_clips`.

    Returns
    -------
    list of Clip
        Clips in manifest order.

    Raises
    ------
    FileNotFoundError
        If there is no manifest in ``clips_dir``.
    """
    records = json.loads((clips_dir / MANIFEST_NAME).read_text(encoding="utf-8"))
    return [Clip(**record) for record in records]
