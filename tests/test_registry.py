import pytest

from camera_bench.decode import DECODE_BACKENDS, DecodeBackend
from camera_bench.inference import INFERENCE_BACKENDS
from camera_bench.registry import Registry, UnknownBackendError


class Base:
    def __init__(self, value: int = 0) -> None:
        self.value = value


def test_decorator_registration_and_create():
    registry = Registry("demo", base=Base)

    @registry.register("one")
    class One(Base):
        pass

    created = registry.create("one", value=3)
    assert isinstance(created, One)
    assert created.value == 3
    assert registry.names() == ["one"]


def test_lazy_path_is_resolved_and_type_checked():
    registry = Registry("demo", base=Base)
    registry.register("bad", "collections:OrderedDict")
    with pytest.raises(TypeError):
        registry.get("bad")


def test_unknown_name_lists_known_backends():
    registry = Registry("demo", base=Base)
    registry.register("one", Base)
    with pytest.raises(UnknownBackendError, match="available: one"):
        registry.get("two")


def test_duplicate_name_is_rejected():
    registry = Registry("demo", base=Base)
    registry.register("one", Base)
    registry.register("one", Base)  # same target is fine
    with pytest.raises(ValueError):
        registry.register("one", "x:Y")


def test_builtin_backends_resolve():
    assert {"ffmpeg-cuda", "ffmpeg-cuvid", "ffmpeg-cpu"} <= set(DECODE_BACKENDS.names())
    assert {"onnxruntime", "ultralytics"} <= set(INFERENCE_BACKENDS.names())
    for name in DECODE_BACKENDS.names():
        assert issubclass(DECODE_BACKENDS.get(name), DecodeBackend)
    # Resolving must not import heavy frameworks.
    INFERENCE_BACKENDS.get("ultralytics")
    INFERENCE_BACKENDS.get("onnxruntime")
