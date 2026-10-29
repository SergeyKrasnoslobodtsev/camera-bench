"""Pin calculator constants: a failure here means camera-calc and this mirror diverged."""

import pytest

from camera_bench.reference import (
    CATALOG_MODELS,
    RESOLUTIONS,
    find_model,
    parse_resolution,
    predicted_decode_streams,
    predicted_model_fps,
    reference_streams,
    stream_bitrate_mbps,
)


def test_resolutions_match_calculator():
    assert [r.label for r in RESOLUTIONS] == [
        "1920x1080",
        "2560x1440",
        "2592x1944",
        "2880x1620",
        "3072x2048",
        "3840x2160",
        "4000x3000",
    ]
    assert len(CATALOG_MODELS) == 15


@pytest.mark.parametrize(
    ("text", "label"),
    [
        ("4mp", "2560x1440"),
        ("5MP", "2592x1944"),
        ("1280x720", "1280x720"),
        ("2880×1620", "2880x1620"),
    ],
)
def test_parse_resolution(text, label):
    assert parse_resolution(text).label == label


@pytest.mark.parametrize("text", ["7mp", "720p", ""])
def test_parse_resolution_rejects(text):
    with pytest.raises(ValueError):
        parse_resolution(text)


def test_find_model_accepts_ids_and_weights():
    assert find_model("y11n") is find_model("yolo11n") is find_model("yolo11n.pt")
    assert find_model("y26sp").weights == "yolo26s-pose"
    assert find_model("rtdetr-l") is None


def test_bitrate_formula():
    # 4 MP H.265 at 25 fps: 2560*1440*25*0.04 bit/s
    assert stream_bitrate_mbps(2560, 1440, 25, "h265") == pytest.approx(3.6864)
    assert stream_bitrate_mbps(1920, 1080, 25, "h264", 2.0) == pytest.approx(7.2576)


def test_decoder_prediction():
    assert predicted_decode_streams("Ampere", 1, "h265") == pytest.approx(1415 / 30)
    assert reference_streams(1920 * 1080 * 60) == pytest.approx(2.0)


def test_model_prediction_scales_with_input_and_tflops():
    model = find_model("y11n")
    at_t4 = predicted_model_fps(model, 640, 65)
    assert at_t4 == pytest.approx(1000 / 1.5)
    assert predicted_model_fps(model, 1280, 130) == pytest.approx(at_t4 / 2)
