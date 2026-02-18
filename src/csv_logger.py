"""Lightweight CSV logger for experiment metrics."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Iterable, Optional


class CSVLogger:
    """
    Append-only CSV writer that guarantees a header row and directory creation.

    Designed for long-running training loops where new metrics are appended
    incrementally (e.g., PPO updates or evaluation rollouts).
    """

    def __init__(
        self,
        filepath: str | Path,
        fieldnames: Optional[Iterable[str]] = None,
        *,
        append: bool = True,
        flush: bool = True,
    ) -> None:
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        mode = "a" if append else "w"
        self._file = path.open(mode, newline="")
        self._writer: Optional[csv.DictWriter] = None
        self._fieldnames = list(fieldnames) if fieldnames else None
        self._flush = flush
        self._header_written = path.exists() and path.stat().st_size > 0 and append

    def log(self, row: Dict[str, object]) -> None:
        """
        Append a single row to the CSV file.

        Args:
            row: Mapping of column name to value.
        """
        if self._writer is None:
            if self._fieldnames is None:
                self._fieldnames = list(row.keys())

            self._writer = csv.DictWriter(self._file, fieldnames=self._fieldnames)
            if not self._header_written:
                self._writer.writeheader()
                self._header_written = True

        self._writer.writerow(row)
        if self._flush:
            self._file.flush()

    def close(self) -> None:
        """Close the underlying file handle."""
        if not self._file.closed:
            self._file.close()

    def __enter__(self) -> "CSVLogger":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()
