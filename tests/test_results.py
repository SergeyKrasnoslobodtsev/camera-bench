import csv
from pathlib import Path

import pytest

from camera_bench.results import write_csv


def test_append_matching_fields_adds_rows(tmp_path: Path) -> None:
    path = tmp_path / "results.csv"
    write_csv([{"model": "a", "fps": 10.0}], path)
    write_csv([{"model": "b", "fps": 20.0}], path)

    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows == [{"model": "a", "fps": "10.0"}, {"model": "b", "fps": "20.0"}]


def test_append_mismatched_fields_raises_and_keeps_file(tmp_path: Path) -> None:
    path = tmp_path / "results.csv"
    write_csv([{"model": "a", "fps": 10.0}], path)
    before = path.read_text(encoding="utf-8")

    with pytest.raises(ValueError, match="model.*fps"):
        write_csv([{"model": "b", "bitrate": 5.0}], path)

    assert path.read_text(encoding="utf-8") == before


def test_append_mismatched_field_order_raises(tmp_path: Path) -> None:
    path = tmp_path / "results.csv"
    write_csv([{"model": "a", "fps": 10.0}], path)

    with pytest.raises(ValueError, match="model.*fps"):
        write_csv([{"fps": 20.0, "model": "b"}], path)
