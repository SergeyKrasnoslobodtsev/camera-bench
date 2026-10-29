"""Writing benchmark rows to CSV and to the terminal."""

from __future__ import annotations

import csv
import dataclasses
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

__all__ = ["console", "print_table", "write_csv"]

console = Console()

_PATH_COLUMNS = frozenset({"model", "file"})


def _as_dict(row: Any) -> dict[str, Any]:
    if dataclasses.is_dataclass(row) and not isinstance(row, type):
        return dataclasses.asdict(row)
    return dict(row)


def write_csv(rows: Sequence[Any], path: Path, *, append: bool = True) -> Path:
    """Write dataclass rows to a CSV file.

    Parameters
    ----------
    rows : sequence of dataclass instances or mappings
        Rows sharing the same fields.
    path : Path
        Output file. Parent directories are created.
    append : bool, default True
        Append to an existing file so several runs collect in one table. The header
        is written only when the file is new or empty.

    Returns
    -------
    Path
        The written file.
    """
    if not rows:
        return path
    records = [_as_dict(row) for row in rows]
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not append or not path.exists() or path.stat().st_size == 0
    with path.open("a" if append else "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        if new_file:
            writer.writeheader()
        writer.writerows(records)
    return path


def _format(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:,.2f}" if abs(value) < 100 else f"{value:,.0f}"
    return str(value)


def print_table(rows: Sequence[Any], columns: Sequence[str], *, title: str) -> None:
    """Print selected columns of rows as a table.

    Parameters
    ----------
    rows : sequence of dataclass instances or mappings
        Rows to show.
    columns : sequence of str
        Field names to display, in order.
    title : str
        Table caption.
    """
    table = Table(title=title, header_style="bold")
    for column in columns:
        table.add_column(column, justify="left" if column in ("model", "file", "clip") else "right")
    for row in rows:
        record = _as_dict(row)
        cells = []
        for column in columns:
            value = record.get(column)
            # Paths are kept in full in CSV; the terminal shows only the file name.
            if column in _PATH_COLUMNS and isinstance(value, str):
                value = Path(value).name
            cells.append(_format(value))
        table.add_row(*cells)
    console.print(table)
