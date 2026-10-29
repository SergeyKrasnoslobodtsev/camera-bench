import json
import time
from dataclasses import asdict

from typer.testing import CliRunner

from camera_bench.cli import app
from camera_bench.clips import MANIFEST_NAME, Clip
from camera_bench.decode import DECODE_BACKENDS, DecodeBackend, DecodeStream

runner = CliRunner()


def test_help_lists_commands():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in ("info", "prepare", "bitrate", "decode", "infer"):
        assert command in result.output


def test_info_runs_without_gpu():
    result = runner.invoke(app, ["info"])
    assert result.exit_code == 0
    assert "onnxruntime" in result.output


def test_unknown_backend_fails_cleanly():
    result = runner.invoke(app, ["infer", "-b", "nope"])
    assert result.exit_code == 1
    assert "unknown inference backend" in result.output


def test_bad_option_syntax():
    result = runner.invoke(app, ["infer", "--device", "cpu", "-O", "provider"])
    assert result.exit_code == 1
    assert "KEY=VALUE" in result.output


def test_decode_without_manifest(tmp_path):
    result = runner.invoke(app, ["decode", "-b", "ffmpeg-cpu", "--clips", str(tmp_path)])
    assert result.exit_code == 1
    assert "prepare" in result.output


class _FakeStream(DecodeStream):
    def __init__(self):
        self._start = time.perf_counter()

    @property
    def frames(self):
        return int((time.perf_counter() - self._start) * 1000)

    def close(self):
        pass


class _FlakyBackend(DecodeBackend):
    """Decodes h264 clips and fails to open h265 ones."""

    def open(self, clip, clips_dir):
        if clip.codec == "h265":
            raise RuntimeError("out of memory")
        return _FakeStream()


DECODE_BACKENDS.register("flaky-test", _FlakyBackend)


def _write_manifest(directory, codecs):
    clips = [Clip(f"{c}.mp4", c, 320, 240, 25, 50, 50, "libx264", 1.0, 1.0) for c in codecs]
    (directory / MANIFEST_NAME).write_text(json.dumps([asdict(c) for c in clips]))


def _decode(tmp_path, codecs):
    _write_manifest(tmp_path, codecs)
    args = ["decode", "-b", "flaky-test", "--clips", str(tmp_path), "-o", str(tmp_path / "out")]
    return runner.invoke(app, [*args, "--max-streams", "2", "--warmup", "0.01", "--window", "0.05"])


def test_decode_keeps_results_when_a_later_clip_fails(tmp_path):
    result = _decode(tmp_path, ["h264", "h265"])
    assert result.exit_code == 0
    assert "out of memory" in result.output
    rows = (tmp_path / "out" / "decode.csv").read_text().splitlines()
    assert len(rows) == 2 and ",h264," in rows[1]
    assert (tmp_path / "out" / "decode_sweep.csv").exists()


def test_decode_fails_when_no_clip_is_measured(tmp_path):
    result = _decode(tmp_path, ["h265"])
    assert result.exit_code == 1
    assert "nothing measured" in result.output
