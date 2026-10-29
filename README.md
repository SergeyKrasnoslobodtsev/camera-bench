# camera-bench

Замеры на железе для калибровки [camera-calc](https://sergeykrasnoslobodtsev.github.io/camera-calc/). Калькулятор считает битрейт, ёмкость декодера и скорость детектора по формулам. camera-bench измеряет то же самое на реальной карте и кладёт рядом прогноз калькулятора. Отношение «замер / прогноз» в каждой строке показывает, где калькулятор ошибается и во сколько раз.

| Что меряем | Команда | Константа калькулятора |
|---|---|---|
| Битрейт записей с реальных камер | `bitrate` | `bpp` 0,07 / 0,04 и коэффициент сцены |
| Сколько потоков реального времени тянет NVDEC | `prepare` + `decode` | `NVDEC_FPS` по архитектуре, опорный поток 1080p@30 |
| Скорость и память детектора | `infer` | задержка на T4 × `(640 / вход)²` × `TFLOPS / 65`, память движка |

## Установка

Нужны [uv](https://docs.astral.sh/uv/) и `ffmpeg` с поддержкой NVDEC в `PATH`. Под Windows подойдёт full-сборка с gyan.dev, под Linux — сборка с `--enable-cuda-nvcc --enable-cuvid --enable-nvdec`. Проверка: в выводе `ffmpeg -hwaccels` есть `cuda`.

```bash
uv sync --extra onnx                       # битрейт, декодер, инференс через ONNX Runtime
uv sync --extra onnx --extra ultralytics   # плюс экспорт моделей каталога из .pt в .onnx
uv sync --all-extras                       # плюс TensorRT для бэкенда ultralytics --format engine
uv run camera-bench info                   # что нашлось: GPU, ffmpeg, NVENC, пакеты, бэкенды
```

`onnxruntime-gpu[cuda,cudnn]` ставит CUDA 12 и cuDNN 9 pip-пакетами, системная CUDA не нужна. Под Windows `torch` берётся из индекса PyTorch с CUDA 12.8: с PyPI под Windows приходит сборка только для CPU.

## Порядок замеров

Пример для RTX 3090 Ti: Ampere, FP16 Tensor ≈ 80 TFLOPS с FP32-накоплением (так же записаны GeForce в каталоге калькулятора). Число NVDEC сверьте с таблицей NVIDIA «Video Encode and Decode GPU Support Matrix».

```bash
# 1. Битрейт реальных записей: файлы или папки, обход рекурсивный
uv run camera-bench bitrate D:/records/day D:/records/night

# 2. Клипы как с камеры: GOP 2 с, без B-кадров, CBR по формуле калькулятора, все разрешения справочника
uv run camera-bench prepare --source any_video.mp4

# 3. Ёмкость декодера
uv run camera-bench decode --arch Ampere --nvdec N   # N из таблицы NVIDIA

# 4. Инференс в ONNX Runtime, как в проде
uv run camera-bench infer --tflops 80 --sizes 640,960,1280 --batches 1,8,16
```

Результаты дописываются в `results/*.csv`. Прогоны на разных машинах собираются в одну таблицу, в каждой строке есть имя GPU.

### bitrate

Колонка `scene_factor` — коэффициент сцены, который нужно ввести в форму калькулятора, чтобы его битрейт совпал с записью. Стоит снять одну и ту же камеру днём, ночью и в дождь: шум матрицы и Smart-кодек меняют поток в разы. Битрейт считается по размеру видеопакетов, без звука и контейнера. Сырые `.h264`/`.h265` без меток времени ffprobe читает как 25 к/с.

### prepare

Видео с YouTube не похоже на поток камеры: там VP9/AV1, длинный GOP, B-кадры и двухпроходное кодирование. `prepare` перекодирует любой исходник так, как кодирует камера. Без `--source` используется тестовая картинка с шумом. Для замера декодера её хватает, для изучения битрейта — нет. По умолчанию кодирует NVENC, если он работает на машине, иначе libx264/libx265.

### decode

Для каждого клипа запускается 1, 2, 4, … параллельных декодов, каждый декодирует так быстро, как может. Когда общий FPS перестаёт расти больше чем на 5%, декодер насыщен, и этот FPS принимается за его ёмкость.

- `realtime_streams` — сколько потоков этого клипа декодер держит в реальном времени.
- `ref_streams` — та же ёмкость в потоках 1080p@30. В этих единицах калькулятор считает нагрузку декодера.
- `predicted_ref_streams` — прогноз калькулятора (`--arch`, `--nvdec`), `measured_to_predicted` — отношение замера к прогнозу.
- `memory_mb_per_stream` — память на поток. В неё входит CUDA-контекст каждого процесса ffmpeg, поэтому цифра завышена по сравнению с одним процессом на много потоков (DeepStream, свой сервис).

Бэкенды: `ffmpeg-cuda` (`-hwaccel cuda`), `ffmpeg-cuvid` (декодеры `*_cuvid`, при сбое падают, а не уходят молча на CPU — полезно для перекрёстной проверки), `ffmpeg-cpu`. Если NVDEC простаивает, в колонке `note` будет предупреждение о программном декодировании.

### infer

По умолчанию бэкенд `onnxruntime` с CUDA execution provider. Модели — имена Ultralytics (`yolo11n`), id калькулятора (`y11n`), `catalog` (все 15 моделей каталога) или путь к своему `.onnx`. Модели по имени один раз экспортируются в ONNX со статическим входом и батчем и кешируются в `.cache/`. Для экспорта нужен extra `ultralytics`, для прогона — только `onnx`.

Режимы:

- `engine` — граф на готовом батче, вход и выход остаются в памяти GPU (IO binding). С этим сравнивается задержка из таблицы Ultralytics, которую масштабирует калькулятор.
- `pipeline` — сырые кадры камеры (`--frame-size` или `--image`): ресайз и нормализация на CPU, копия на GPU, граф, копия результатов обратно. Декодирование боксов и NMS у разных моделей разные и в замер не входят.

Колонка `measured_to_predicted` — это фактически коэффициент реальности калькулятора для данной карты, модели, входа и батча. Калькулятор опирается на замеры TensorRT, поэтому для ONNX Runtime с CUDA EP ожидайте отношение заметно ниже единицы. Именно эту разницу и надо заложить. Прогноз не учитывает батч, поэтому строки с батчем > 1 показывают, сколько он даёт. `calc_memory_mb` калькулятор всегда считает для батча 16, сравнивайте его со строками `batch = 16`.

Настройки бэкенда передаются через `-O KEY=VALUE`:

```bash
uv run camera-bench infer -O provider=tensorrt          # TensorRT EP: граф FP32, ядра FP16
uv run camera-bench infer -O cudnn_search=HEURISTIC     # как в проде, если там не EXHAUSTIVE
uv run camera-bench infer -b ultralytics -f engine      # чистый TensorRT через Ultralytics
uv run camera-bench infer -m D:/models/my-det.onnx --sizes 960 --batches 8
```

## Устройство

```
src/camera_bench/
├── cli.py              команды typer
├── reference.py        константы и формулы калькулятора; меняются вместе с camera-calc
├── registry.py         фабрика бэкендов по имени, ленивый импорт, плагины через entry points
├── media.py            обёртки ffmpeg и ffprobe
├── gpu.py              NVML: имя карты, фоновый замер памяти и загрузки NVDEC
├── results.py          CSV и таблицы в терминале
├── bitrate.py          битрейт записей
├── clips.py            клипы как с камеры и manifest.json
├── decode/             интерфейс DecodeBackend, бэкенды ffmpeg, прогон по числу потоков
└── inference/          интерфейс InferenceBackend, бэкенды onnxruntime и ultralytics, прогон
```

### Свой бэкенд

Бэкенд — подкласс `InferenceBackend` (или `DecodeBackend`), зарегистрированный по имени. Сессия реализует `infer()` и, по желанию, `predict()`. Оба метода возвращаются только после завершения работы на устройстве.

```python
from camera_bench.inference import INFERENCE_BACKENDS, InferenceBackend, Session


class OpenVinoSession(Session):
    def infer(self) -> None: ...


@INFERENCE_BACKENDS.register("openvino")
class OpenVinoBackend(InferenceBackend):
    formats = ("onnx",)

    def load(self, config):
        return OpenVinoSession(config)
```

Из отдельного пакета бэкенд подключается без правки этого проекта, через entry point:

```toml
[project.entry-points."camera_bench.inference"]
openvino = "my_package.backends:OpenVinoBackend"
```

## Разработка

```bash
uv sync --all-extras --group dev
uv run pytest            # без GPU: CPU-декодер, ONNX Runtime на CPU, фейковый бэкенд
uv run ruff check . && uv run ruff format --check .
```

Стиль: докстринги numpy, проверка `ruff` с `convention = "numpy"`.
