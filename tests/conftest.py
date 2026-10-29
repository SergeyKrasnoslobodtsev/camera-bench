import shutil
from pathlib import Path

import pytest

from camera_bench.clips import prepare_clips
from camera_bench.reference import Resolution

needs_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg and ffprobe are required",
)


@pytest.fixture(scope="session")
def small_clips(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Two short 320x240 clips encoded on the CPU."""
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg is required")
    out = tmp_path_factory.mktemp("clips")
    prepare_clips(out, [Resolution(320, 240, 0)], ["h264", "h265"], seconds=2, encoder="cpu")
    return out
