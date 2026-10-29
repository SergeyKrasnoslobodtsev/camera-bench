from conftest import needs_ffmpeg

from camera_bench.clips import load_manifest
from camera_bench.decode import DECODE_BACKENDS
from camera_bench.decode.runner import measure_capacity, stream_counts


def test_stream_counts():
    assert list(stream_counts(1)) == [1]
    assert list(stream_counts(8)) == [1, 2, 4, 8]
    assert list(stream_counts(10)) == [1, 2, 4, 8, 10]


@needs_ffmpeg
def test_cpu_sweep_reports_capacity(small_clips):
    clip = next(c for c in load_manifest(small_clips) if c.codec == "h264")
    backend = DECODE_BACKENDS.create("ffmpeg-cpu")
    backend.check()
    capacity, sweep = measure_capacity(
        backend,
        clip,
        small_clips,
        backend_name="ffmpeg-cpu",
        gpu_name="cpu",
        max_streams=2,
        warmup_s=0.5,
        window_s=1.0,
        arch="Ampere",
        nvdec=1,
    )
    assert [row.streams for row in sweep] == [1, 2][: len(sweep)]
    assert capacity.capacity_fps > clip.fps  # a 320x240 clip decodes far above real time
    assert capacity.realtime_streams >= 1
    assert capacity.predicted_ref_streams == round(1 * 748 / 30, 1)
    assert capacity.measured_to_predicted is not None
