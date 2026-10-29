from typer.testing import CliRunner

from camera_bench.cli import app

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
