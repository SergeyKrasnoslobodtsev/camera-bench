"""Field measurements that calibrate camera-calc.

The package measures three things the calculator estimates from formulas:

- stream bitrate of real camera recordings (:mod:`camera_bench.bitrate`);
- how many real-time streams NVDEC decodes (:mod:`camera_bench.decode`);
- detector throughput and memory (:mod:`camera_bench.inference`).

Calculator constants live in :mod:`camera_bench.reference`.
"""

__version__ = "0.1.0"
