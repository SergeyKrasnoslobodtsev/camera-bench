from pathlib import Path

import pytest

from camera_bench.inference import INFERENCE_BACKENDS, InferenceBackend, ModelConfig, Session
from camera_bench.inference.runner import engine_memory_mb, measure_session, time_calls


class FakeSession(Session):
    def __init__(self, config):
        super().__init__(config)
        self.calls = 0

    def infer(self):
        self.calls += 1


class FakeBackend(InferenceBackend):
    formats = ("fake",)

    def load(self, config):
        return FakeSession(config)


def test_time_calls_counts():
    calls = []
    durations = time_calls(lambda: calls.append(1), warmup=3, iters=5)
    assert len(calls) == 8 and len(durations) == 5


def test_measure_session_engine_only_and_prediction():
    backend = FakeBackend(device="cpu")
    config = ModelConfig("yolo11n", 640, 4, "fp16", "fake")
    rows = measure_session(
        backend,
        config,
        backend_name="fake",
        gpu_name="cpu",
        modes=("engine", "pipeline"),
        warmup=1,
        iters=10,
        tflops=65,
    )
    assert [row.mode for row in rows] == ["engine"]  # pipeline unsupported -> skipped
    row = rows[0]
    assert row.calc_id == "y11n"
    assert row.predicted_fps == pytest.approx(1000 / 1.5, rel=1e-3)
    assert row.measured_to_predicted == pytest.approx(row.fps / row.predicted_fps, rel=1e-2)


def test_engine_memory_formula():
    # 0.15 GB + 16 * 640^2 * 6 * 6.5 / 4 / 1e9 GB
    expected_gb = 0.15 + 16 * 640**2 * 6 * 6.5 / 4 / 1e9
    assert engine_memory_mb(6.5, 640) == pytest.approx(expected_gb * 1e9 / 2**20)


def test_config_tag_uses_onnx_stem():
    assert ModelConfig("/models/my-det.onnx", 960, 8).tag == "my-det_960_b8_fp16"


def test_onnxruntime_rejects_unknown_provider():
    with pytest.raises(ValueError):
        INFERENCE_BACKENDS.create("onnxruntime", provider="openvino")


def test_fp16_rejected_on_cpu():
    # the guard runs before any heavy import, so no extras are needed
    config = ModelConfig("yolo11n", 640, 1, "fp16", "onnx")
    for name in ("onnxruntime", "ultralytics"):
        backend = INFERENCE_BACKENDS.create(name, device="cpu")
        with pytest.raises(ValueError, match="--precision fp32"):
            backend.load(config)


def _tiny_onnx(path: Path) -> Path:
    np = pytest.importorskip("numpy")
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper, numpy_helper

    weight = numpy_helper.from_array(np.ones((4, 3, 3, 3), dtype=np.float32), "w")
    graph = helper.make_graph(
        [helper.make_node("Conv", ["images", "w"], ["output0"], pads=[1, 1, 1, 1])],
        "tiny",
        [helper.make_tensor_value_info("images", TensorProto.FLOAT, [2, 3, 32, 32])],
        [helper.make_tensor_value_info("output0", TensorProto.FLOAT, None)],
        [weight],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=9)
    onnx.save(model, path)
    return path


def test_onnxruntime_cpu_end_to_end(tmp_path):
    pytest.importorskip("onnxruntime")
    pytest.importorskip("cv2")
    import numpy as np

    model = _tiny_onnx(tmp_path / "tiny.onnx")
    backend = INFERENCE_BACKENDS.create("onnxruntime", device="cpu", workdir=tmp_path)
    backend.check()
    assert backend.describe() == "cpu"
    config = ModelConfig(backend.resolve_model(str(model)), 32, 2, "fp32", "onnx")
    frames = [np.zeros((120, 160, 3), dtype=np.uint8)] * 2
    rows = measure_session(
        backend,
        config,
        backend_name="onnxruntime:cpu",
        gpu_name="cpu",
        frames=frames,
        warmup=2,
        iters=5,
    )
    assert [row.mode for row in rows] == ["engine", "pipeline"]
    assert all(row.fps > 0 and row.calc_id is None for row in rows)


def test_ultralytics_predict_preprocesses_without_nms():
    # pipeline mode must feed the bare network, like the ONNX backend: no YOLO.predict,
    # so no letterbox and no NMS in the timed call
    torch = pytest.importorskip("torch")
    pytest.importorskip("cv2")
    pytest.importorskip("ultralytics")
    from types import SimpleNamespace

    import numpy as np

    from camera_bench.inference.ultralytics import UltralyticsSession

    class _CaptureBackend:
        def __init__(self):
            self.fp16 = False
            self.device = torch.device("cpu")
            self.tensors = []

        def __call__(self, tensor):
            self.tensors.append(tensor.detach().cpu().numpy())

    capture = _CaptureBackend()
    model = SimpleNamespace(predictor=None)
    model.predict = lambda frames, **kwargs: setattr(
        model, "predictor", SimpleNamespace(model=capture)
    )
    config = ModelConfig("yolo11n", 32, 2, "fp32", "torch")
    session = UltralyticsSession(model, config, "cpu")
    frame = np.zeros((40, 48, 3), dtype=np.uint8)
    frame[:, :] = (10, 20, 30)  # BGR; plain resize, BGR->RGB, /255, no letterbox
    session.predict([frame, frame])

    assert len(capture.tensors) == 1
    tensor = capture.tensors[0]
    assert tensor.shape == (2, 3, 32, 32)
    assert tensor[0, 2, 0, 0] == pytest.approx(10 / 255)  # B channel last in RGB
    assert tensor[0, 0, 0, 0] == pytest.approx(30 / 255)  # R channel first
    assert session.supports_pipeline
