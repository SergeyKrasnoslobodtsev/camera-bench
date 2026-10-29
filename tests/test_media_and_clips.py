import json

import pytest
from conftest import needs_ffmpeg

from camera_bench.bitrate import find_videos, measure_bitrate
from camera_bench.clips import MANIFEST_NAME, load_manifest
from camera_bench.media import probe_video

pytestmark = needs_ffmpeg


def test_manifest_lists_clips(small_clips):
    clips = load_manifest(small_clips)
    assert {(c.codec, c.width, c.height) for c in clips} == {("h264", 320, 240), ("h265", 320, 240)}
    assert all(c.frames == 50 and c.gop == 50 for c in clips)
    assert json.loads((small_clips / MANIFEST_NAME).read_text())


def test_clips_have_no_b_frames(small_clips):
    from camera_bench.media import run_ffmpeg

    for clip in load_manifest(small_clips):
        types = run_ffmpeg(
            [
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "frame=pict_type",
                "-of",
                "csv=p=0",
                str(small_clips / clip.path),
            ],
            tool="ffprobe",
        ).split()
        types = [frame.strip(",") for frame in types]  # ffprobe may append empty side-data fields
        assert "B" not in types
        assert types[0] == "I"


def test_probe_and_bitrate_agree(small_clips):
    clip = load_manifest(small_clips)[0]
    info = probe_video(small_clips / clip.path)
    assert (info.width, info.height, info.fps) == (320, 240, 25)
    row = measure_bitrate(small_clips / clip.path)
    assert row.bitrate_mbps == pytest.approx(info.bitrate_mbps, abs=1e-3)
    assert row.scene_factor == pytest.approx(row.bpp / row.calc_bpp, rel=1e-2)


def test_find_videos_skips_non_video(small_clips):
    found = list(find_videos([small_clips]))
    assert found and all(path.suffix == ".mp4" for path in found)
