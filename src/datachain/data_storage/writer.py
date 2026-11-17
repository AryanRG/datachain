"""Writer: the only way UDF output rows enter a table.

Ids are ``writer_no << ID_BITS | seq`` from 1; rows that carry their sys__id (map)
keep it. A gen/agg table's ledger ``<table>_parts`` holds a claim row per writer and
``(part_key, writer_no, lo, hi)`` per finished part, written with or after the
part's rows, so a writer's committed rows are its ids up to ``max(hi)``.
"""

import logging
import sys
import time
from collections.abc import Iterable
from contextlib import ExitStack
from typing import TYPE_CHECKING, Any, Self

import sqlalchemy as sa

from datachain.sql.types import Int64

if TYPE_CHECKING:
    from collections.abc import Callable

    import pyarrow as pa

    from datachain.data_storage.warehouse import AbstractWarehouse

logger = logging.getLogger("datachain")

ID_BITS = 40
MAX_SEQ = (1 << ID_BITS) - 1
# SQLite ids are signed 64-bit; the top writer_no is reserved for Studio
MAX_WRITER_NO = (1 << 23) - 2
MAX_ID = (1 << 63) - 1
LEDGER_COLUMNS = ("part_key", "writer_no", "lo", "hi")


def parts_table_name(table_name: str) -> str:
    return f"{table_name}_parts"


def parts_columns() -> list[sa.Column]:
    return [sa.Column(name, Int64) for name in LEDGER_COLUMNS]


class Writer:
    def __init__(
        self,
        warehouse: "AbstractWarehouse",
        table: sa.Table,
        columns: Iterable[str],
        writer_no: int = 0,
        parts: sa.Table | None = None,
        batch_size: int | None = None,
    ) -> None:
        if not 0 <= writer_no <= MAX_WRITER_NO:
            raise ValueError(f"writer_no {writer_no} is out of range")
        self.warehouse = warehouse
        self.table = table
        self.columns = list(columns)
        self.assign_ids = "sys__id" not in self.columns
        if self.assign_ids:
            self.columns.append("sys__id")
        self.writer_no = writer_no
        self.parts = parts
        self.batch_size = batch_size or warehouse.INSERT_BATCH_SIZE
        self.flush_interval = warehouse.INSERT_FLUSH_INTERVAL
        self.rows: list[tuple] = []
        self.ledger: list[tuple] = []
        self.next_id = (writer_no << ID_BITS) + 1  # id of self.rows[0]
        self.part_lo = self.next_id
        self.flush_at = time.monotonic() + self.flush_interval
        self._resources = ExitStack()
        self._insert_arrow: Callable[[pa.RecordBatch], None] | None = None

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if exc_type is None:
            self.close()
            return
        try:
            self.close(abort=True)
        except Exception:  # the error in flight says more than this one
            logger.exception("Could not flush committed rows into %s", self.table)

    def extend(self, rows: "Iterable[tuple | pa.RecordBatch]") -> None:
        buf, size = self.rows, self.batch_size
        pa = sys.modules.get("pyarrow")
        for row in rows:
            if pa is not None and isinstance(row, pa.RecordBatch):
                self.append_arrow(row)
                continue
            buf.append(row)
            if len(buf) >= size or time.monotonic() >= self.flush_at:
                self.flush()

    def append_arrow(self, batch: "pa.RecordBatch") -> None:
        """Append stored Arrow columns, assigning ids without Python row objects.
        Source constants, when present, are already columns in the batch."""
        import numpy as np
        import pyarrow as pa

        self.flush()
        if not batch.num_rows:
            return
        hi = self.next_id + batch.num_rows
        columns = list(batch.columns)
        if self.assign_ids:
            if hi - 1 - (self.writer_no << ID_BITS) > MAX_SEQ:
                raise RuntimeError(f"Writer {self.writer_no} ran out of ids")
            columns.append(pa.array(np.arange(self.next_id, hi, dtype=np.int64)))
        batch = pa.RecordBatch.from_arrays(columns, names=self.columns)
        if self._insert_arrow is None:
            self._insert_arrow = self._resources.enter_context(
                self.warehouse.arrow_writer(self.table)
            )
        self._insert_arrow(batch)
        self.next_id = hi

    def commit(self, part_key: Any) -> None:
        """The part's input is exhausted: its rows, possibly none, are final."""
        hi = self.next_id + len(self.rows) - 1
        if self.parts is not None:
            self.ledger.append((part_key, self.writer_no, self.part_lo, hi))
        self.part_lo = hi + 1
        if len(self.ledger) >= self.batch_size or time.monotonic() >= self.flush_at:
            self.flush()

    def flush(self) -> None:
        rows = self.rows
        block: list[tuple] = rows
        if rows and self.assign_ids:
            hi = self.next_id + len(rows)
            if hi - 1 - (self.writer_no << ID_BITS) > MAX_SEQ:
                raise RuntimeError(f"Writer {self.writer_no} ran out of ids")
            # a list, not an iterator: a retry after a lock error re-reads it
            block = list(map(tuple.__add__, rows, zip(range(self.next_id, hi))))
        if rows or self.ledger:
            self.warehouse.insert_block(
                self.table, self.columns, block, self.parts, self.ledger
            )
        self.next_id += len(rows)
        rows.clear()
        self.ledger = []
        self.flush_at = time.monotonic() + self.flush_interval

    def close(self, abort: bool = False) -> None:
        """Flush what is buffered. On abort, rows of the unfinished part are
        dropped: they have no ledger row, so they would be garbage."""
        if abort and self.assign_ids:
            del self.rows[max(0, self.part_lo - self.next_id) :]
        try:
            self.flush()
        finally:
            self._resources.close()


def trim_to_ledger(warehouse: "AbstractWarehouse", table: sa.Table, parts: sa.Table):
    """Delete the rows outside the ledger's committed ranges: one range per
    writer, from the end of its committed prefix to the next writer's first id."""
    committed = warehouse.db.execute(
        sa.select(parts.c.writer_no, sa.func.max(parts.c.hi))
        .group_by(parts.c.writer_no)
        .order_by(parts.c.writer_no)
    )
    gaps, lo = [], 0
    for writer_no, hi in committed:
        if (first := (writer_no << ID_BITS) + 1) > lo:
            gaps.append((lo, first - 1))
        lo = max(lo, hi + 1)
    gaps.append((lo, MAX_ID))
    warehouse.delete_id_ranges(table, gaps)


def committed_part_keys(parts: sa.Table) -> sa.Select:
    """Keys of the committed parts; claim rows have none."""
    return sa.select(parts.c.part_key).where(parts.c.part_key.is_not(None))
