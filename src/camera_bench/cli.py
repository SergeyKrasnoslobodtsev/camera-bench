"""Command-line interface.

Run ``camera-bench --help`` for the list of commands. Every command prints a table
and appends its rows to a CSV file under ``--out-dir`` so runs on different machines
can be collected into one dataset.
"""

from __future__ import annotations

import platform
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Annotated

import typer

from camera_bench import bitrate as bitrate_mod
from camera_bench.clips import load_manifest, prepare_clips
from camera_bench.decode import DECODE_BACKENDS
from camera_bench.decode import BackendUnavailableError as DecodeUnavailable
from camera_bench.decode.runner import measure_capacity
from camera_bench.gpu import device_info
from camera_bench.inference import INFERENCE_BACKENDS, ModelConfig
from camera_bench.inference import BackendUnavailableError as InferenceUnavailable
from camera_bench.inference.runner import measure_session
from camera_bench.media import FFmpegError, encoder_works, ffmpeg_version, hwaccels
from camera_bench.reference import CATALOG_MODELS, NVDEC_FPS, RESOLUTIONS, parse_resolution
from camera_bench.registry import UnknownBackendError
from camera_bench.results import console, print_table, write_csv

app = typer.Typer(
    name="camera-bench",
    help="Field measurements for camera-calc: bitrate, NVDEC capacity, detector throughput.",
    no_args_is_help=True,
    add_completion=False,
    pretty_exceptions_show_locals=False,
)

OutDir = Annotated[
    Path, typer.Option("--out-dir", "-o", help="Directory for CSV results; rows are appended.")
]


def _split(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _ints(value: str) -> list[int]:
    try:
        return [int(item) for item in _split(value)]
    except ValueError:
        raise typer.BadParameter(f"expected comma-separated integers, got {value!r}") from None


def _fail(message: str) -> typer.Exit:
    console.print(f"[bold red]error:[/] {message}")
    return typer.Exit(code=1)


def _gpu_name(device: str | int) -> str:
    if str(device) == "cpu":
        return f"cpu ({platform.processor() or platform.machine()})"
    info = device_info(int(device))
    return info.name if info else "unknown"


@app.command()
def info() -> None:
    """Show GPU, ffmpeg capabilities and available backends."""
    gpu = device_info(0)
    console.print(f"[bold]GPU[/]             {gpu.name if gpu else 'not found (no NVML)'}")
    if gpu:
        console.print(f"                driver {gpu.driver}, {gpu.memory_mb / 1024:.1f} GiB")
    try:
        console.print(f"[bold]ffmpeg[/]          {ffmpeg_version()}")
        console.print(f"                hwaccels: {', '.join(hwaccels()) or 'none'}")
        working = [name for name in ("h264_nvenc", "hevc_nvenc") if encoder_works(name)]
        console.print(f"                working NVENC: {', '.join(working) or 'none'}")
    except FFmpegError as exc:
        console.print(f"[bold]ffmpeg[/]          [red]{exc}[/]")
    for package in ("ultralytics", "torch", "tensorrt", "onnxruntime-gpu"):
        try:
            console.print(f"[bold]{package:<16}[/]{version(package)}")
        except PackageNotFoundError:
            console.print(f"[bold]{package:<16}[/][dim]not installed[/]")
    console.print(f"[bold]decode[/]          {', '.join(DECODE_BACKENDS.names())}")
    console.print(f"[bold]inference[/]       {', '.join(INFERENCE_BACKENDS.names())}")


@app.command()
def prepare(
    out: Annotated[
        Path, typer.Option("--out", help="Directory for clips and manifest.json.")
    ] = Path("clips"),
    source: Annotated[
        Path | None,
        typer.Option(
            "--source",
            "-s",
            exists=True,
            dir_okay=False,
            help="Source video; a noisy test pattern is used when omitted.",
        ),
    ] = None,
    resolutions: Annotated[
        str,
        typer.Option(
            "--resolutions",
            "-r",
            help="Comma-separated WxH or <N>mp; 'all' is every calculator resolution.",
        ),
    ] = "all",
    codecs: Annotated[
        str, typer.Option("--codecs", "-c", help="h264, h265 or both.")
    ] = "h264,h265",
    fps: Annotated[float, typer.Option(help="Output frame rate.")] = 25,
    seconds: Annotated[float, typer.Option(help="Clip duration, s.")] = 20,
    gop_seconds: Annotated[float, typer.Option(help="Keyframe interval, s.")] = 2,
    scene_factor: Annotated[
        float, typer.Option(help="Multiplier on the calculator bitrate.")
    ] = 1.0,
    encoder: Annotated[str, typer.Option(help="auto, nvenc or cpu.")] = "auto",
) -> None:
    """Encode camera-like clips: fixed GOP, no B-frames, CBR at the calculator bitrate."""
    sizes = (
        list(RESOLUTIONS)
        if resolutions == "all"
        else [parse_resolution(item) for item in _split(resolutions)]
    )
    codec_list = _split(codecs)
    if not set(codec_list) <= {"h264", "h265"} or encoder not in ("auto", "nvenc", "cpu"):
        raise _fail("codecs must be h264/h265 and encoder auto/nvenc/cpu")
    try:
        clips = prepare_clips(
            out,
            sizes,
            codec_list,
            source=source,
            fps=fps,
            seconds=seconds,  # type: ignore[arg-type]
            gop_seconds=gop_seconds,
            scene_factor=scene_factor,
            encoder=encoder,  # type: ignore[arg-type]
            on_clip=lambda clip: console.print(
                f"  {clip.name:<22} {clip.encoder:<11} {clip.bitrate_mbps:6.2f} Mbit/s"
            ),
        )
    except FFmpegError as exc:
        raise _fail(str(exc)) from exc
    console.print(f"[green]{len(clips)} clips[/] in {out}")


@app.command()
def bitrate(
    paths: Annotated[list[Path], typer.Argument(exists=True, help="Recordings or directories.")],
    out_dir: OutDir = Path("results"),
) -> None:
    """Measure bitrate and bits per pixel of real camera recordings.

    The scene_factor column is the value to enter in the calculator form so that
    its bitrate matches the recording.
    """
    rows = []
    for path in bitrate_mod.find_videos(paths):
        try:
            rows.append(bitrate_mod.measure_bitrate(path))
        except FFmpegError as exc:
            console.print(f"[yellow]skip[/] {path}: {exc}")
    if not rows:
        raise _fail("no readable video files")
    print_table(
        rows,
        [
            "file",
            "codec",
            "width",
            "height",
            "fps",
            "bitrate_mbps",
            "bpp",
            "calc_bpp",
            "scene_factor",
        ],
        title="Recorded bitrate vs camera-calc",
    )
    console.print(f"saved to {write_csv(rows, out_dir / 'bitrate.csv')}")


@app.command()
def decode(
    clips_dir: Annotated[Path, typer.Option("--clips", help="Directory made by `prepare`.")] = Path(
        "clips"
    ),
    backend_name: Annotated[
        str, typer.Option("--backend", "-b", help="Decoder backend, see `info`.")
    ] = "ffmpeg-cuda",
    device: Annotated[int, typer.Option(help="GPU index.")] = 0,
    codecs: Annotated[
        str, typer.Option("--codecs", "-c", help="Filter clips by codec.")
    ] = "h264,h265",
    resolutions: Annotated[
        str, typer.Option("--resolutions", "-r", help="Filter clips: WxH or <N>mp list, or 'all'.")
    ] = "all",
    max_streams: Annotated[int, typer.Option(help="Upper bound for parallel streams.")] = 64,
    warmup: Annotated[float, typer.Option(help="Seconds before measuring.")] = 3.0,
    window: Annotated[float, typer.Option(help="Measurement window, s.")] = 10.0,
    arch: Annotated[
        str | None,
        typer.Option(
            help=f"GPU architecture for the calculator prediction: {', '.join(NVDEC_FPS)}."
        ),
    ] = None,
    nvdec: Annotated[
        int | None, typer.Option(help="NVDEC engines on the card, for the prediction.")
    ] = None,
    out_dir: OutDir = Path("results"),
) -> None:
    """Find how many real-time streams the decoder sustains per clip."""
    if arch is not None and arch not in NVDEC_FPS:
        raise _fail(f"--arch must be one of {', '.join(NVDEC_FPS)}")
    try:
        backend = DECODE_BACKENDS.create(backend_name, device=device)
        clips = load_manifest(clips_dir)
        backend.check()
    except (UnknownBackendError, DecodeUnavailable) as exc:
        raise _fail(str(exc)) from exc
    except FileNotFoundError:
        raise _fail(f"no manifest in {clips_dir}; run `camera-bench prepare` first") from None

    wanted_codecs = set(_split(codecs))
    wanted_sizes = (
        None
        if resolutions == "all"
        else {parse_resolution(item).label for item in _split(resolutions)}
    )
    clips = [
        clip
        for clip in clips
        if clip.codec in wanted_codecs
        and (wanted_sizes is None or f"{clip.width}x{clip.height}" in wanted_sizes)
    ]
    if not clips:
        raise _fail("no clips match the filters")

    gpu = _gpu_name(device if backend.hardware else "cpu")
    summary = []
    for clip in clips:
        console.print(f"[bold]{clip.name}[/]")
        try:
            capacity, steps = measure_capacity(
                backend,
                clip,
                clips_dir,
                backend_name=backend_name,
                gpu_name=gpu,
                max_streams=max_streams,
                warmup_s=warmup,
                window_s=window,
                arch=arch,
                nvdec=nvdec,  # type: ignore[arg-type]
                on_step=lambda row: console.print(
                    f"  {row.streams:>3} streams  {row.aggregate_fps:9.1f} fps total  "
                    f"{row.per_stream_fps:8.1f} fps each"
                    + (f"  NVDEC {row.decoder_util:.0f}%" if row.decoder_util is not None else "")
                ),
            )
        except RuntimeError as exc:
            console.print(f"  [red]failed:[/] {exc}")
            continue
        summary.append(capacity)
        write_csv(steps, out_dir / "decode_sweep.csv")  # save as we go: a full run is long
        write_csv([capacity], out_dir / "decode.csv")

    if not summary:
        raise _fail("nothing measured")
    print_table(
        summary,
        [
            "codec",
            "width",
            "height",
            "fps",
            "capacity_fps",
            "realtime_streams",
            "ref_streams",
            "predicted_ref_streams",
            "measured_to_predicted",
            "memory_mb_per_stream",
            "note",
        ],
        title=f"Decoder capacity, {gpu}, {backend_name}",
    )
    console.print(f"saved to {out_dir / 'decode.csv'}")


@app.command()
def infer(
    backend_name: Annotated[
        str, typer.Option("--backend", "-b", help="Inference backend, see `info`.")
    ] = "onnxruntime",
    backend_options: Annotated[
        list[str] | None,
        typer.Option(
            "--opt", "-O", help="Backend option KEY=VALUE, repeatable, e.g. -O provider=tensorrt."
        ),
    ] = None,
    models: Annotated[
        str,
        typer.Option(
            "--models",
            "-m",
            help="Comma-separated names (yolo11n or calculator id y11n); "
            "'catalog' is every calculator model.",
        ),
    ] = "catalog",
    sizes: Annotated[str, typer.Option("--sizes", help="Network inputs, px.")] = "640",
    batches: Annotated[str, typer.Option("--batches", help="Batch sizes.")] = "1,16",
    fmt: Annotated[
        str | None,
        typer.Option(
            "--format", "-f", help="Backend runtime format; default is the backend's first."
        ),
    ] = None,
    precision: Annotated[str, typer.Option(help="fp16 or fp32.")] = "fp16",
    modes: Annotated[str, typer.Option(help="engine, pipeline or both.")] = "engine,pipeline",
    device: Annotated[str, typer.Option(help="GPU index or 'cpu'.")] = "0",
    tflops: Annotated[
        float | None,
        typer.Option(help="FP16 Tensor dense TFLOPS of this GPU, as in the calculator catalog."),
    ] = None,
    frame_size: Annotated[
        str, typer.Option(help="Camera frame for pipeline mode, WxH or <N>mp.")
    ] = "1920x1080",
    image: Annotated[
        Path | None,
        typer.Option(exists=True, dir_okay=False, help="Real frame for pipeline mode."),
    ] = None,
    warmup: Annotated[int, typer.Option(help="Untimed calls per mode.")] = 20,
    iters: Annotated[int, typer.Option(help="Timed calls per mode.")] = 200,
    workdir: Annotated[Path, typer.Option(help="Cache for weights and engines.")] = Path(".cache"),
    out_dir: OutDir = Path("results"),
) -> None:
    """Measure detector throughput and compare it with the calculator estimate."""
    mode_list = _split(modes)
    if not set(mode_list) <= {"engine", "pipeline"} or precision not in ("fp16", "fp32"):
        raise _fail("modes must be engine/pipeline and precision fp16/fp32")
    if device == "cpu" and precision == "fp16":
        raise _fail("fp16 needs a GPU; on CPU pass --precision fp32")
    try:
        backend = INFERENCE_BACKENDS.create(
            backend_name, device=device, workdir=workdir, **_options(backend_options)
        )
        backend.check()
    except (UnknownBackendError, InferenceUnavailable, ValueError) as exc:
        raise _fail(str(exc)) from exc
    except TypeError as exc:
        raise _fail(f"bad --opt for {backend_name}: {exc}") from exc
    fmt = fmt or backend.formats[0]
    label = f"{backend_name}:{backend.describe()}" if backend.describe() else backend_name
    if fmt not in backend.formats:
        raise _fail(f"{backend_name} supports formats {', '.join(backend.formats)}")

    names = [model.weights for model in CATALOG_MODELS] if models == "catalog" else _split(models)
    frames = _frames(frame_size, image, max(_ints(batches))) if "pipeline" in mode_list else None
    gpu = _gpu_name(device)
    device_index = 0 if device == "cpu" else int(device)

    rows = []
    for name in names:
        for size in _ints(sizes):
            for batch in _ints(batches):
                config = ModelConfig(backend.resolve_model(name), size, batch, precision, fmt)  # type: ignore[arg-type]
                console.print(f"[bold]{config.tag}[/] {fmt}")
                try:
                    measured = measure_session(
                        backend,
                        config,
                        backend_name=label,
                        gpu_name=gpu,
                        modes=mode_list,
                        frames=frames,
                        warmup=warmup,
                        iters=iters,  # type: ignore[arg-type]
                        tflops=tflops,
                        device_index=device_index,
                    )
                except Exception as exc:  # noqa: BLE001 - one bad config must not stop the run
                    console.print(f"  [red]failed:[/] {exc}")
                    continue
                for row in measured:
                    console.print(
                        f"  {row.mode:<8} {row.fps:9.1f} img/s  {row.latency_ms:7.2f} ms/batch"
                        + (
                            f"  x{row.measured_to_predicted} of calc"
                            if row.measured_to_predicted
                            else ""
                        )
                    )
                rows.extend(measured)
                write_csv(measured, out_dir / "infer.csv")  # save as we go: engine builds are slow

    if not rows:
        raise _fail("nothing measured")
    print_table(
        rows,
        [
            "model",
            "input_size",
            "batch",
            "mode",
            "latency_ms",
            "fps",
            "predicted_fps",
            "measured_to_predicted",
            "memory_mb",
            "calc_memory_mb",
        ],
        title=f"Inference, {gpu}, {label} {fmt} {precision}",
    )
    console.print(f"saved to {out_dir / 'infer.csv'}")


def _options(items: list[str] | None) -> dict[str, str]:
    """Parse repeated ``KEY=VALUE`` options into constructor keyword arguments."""
    options = {}
    for item in items or []:
        key, sep, value = item.partition("=")
        if not sep or not key:
            raise _fail(f"--opt expects KEY=VALUE, got {item!r}")
        options[key.strip().replace("-", "_")] = value.strip()
    return options


def _frames(frame_size: str, image: Path | None, count: int) -> list[object]:
    """Raw BGR frames for pipeline mode: a real image or seeded noise."""
    import numpy as np

    if image is not None:
        import cv2

        frame = cv2.imread(str(image))
        if frame is None:
            raise _fail(f"cannot read image {image}")
    else:
        size = parse_resolution(frame_size)
        rng = np.random.default_rng(0)
        frame = rng.integers(0, 256, (size.height, size.width, 3), dtype=np.uint8)
    return [frame.copy() for _ in range(count)]


if __name__ == "__main__":
    app()
