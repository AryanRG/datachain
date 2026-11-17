import os
import sqlite3
import subprocess
import sys
from contextlib import closing

import pyarrow as pa
import pytest
import sqlalchemy as sa

from datachain.data_storage.sqlite_arrow import SQLiteArrowWriter
from datachain.sql.types import Boolean, Int, String
from tests.utils import skip_if_not_sqlite

pytestmark = [
    skip_if_not_sqlite,
    pytest.mark.skipif(os.name == "nt", reason="Native writer uses POSIX ownership"),
]


def test_one_worker_commits_batches_without_staging(warehouse_tmpfile, mocker):
    warehouse = warehouse_tmpfile
    table = warehouse.create_dataset_rows_table("native_direct", [sa.Column("x", Int)])
    started = mocker.spy(SQLiteArrowWriter, "__init__")
    statements = []
    warehouse.db.db.set_trace_callback(statements.append)
    try:
        with warehouse.writer(table, ["x"]) as writer:
            writer.append_arrow(pa.record_batch([list(range(2048))], names=["x"]))
            writer.append_arrow(pa.record_batch([[30]], names=["x"]))
            # Acknowledged rows are already visible from the parent's connection.
            assert warehouse.table_rows_count(table) == 2049
        assert started.call_count == 1
        process = started.call_args.args[0].process
        assert process.returncode == 0
        assert process.stdin.closed and process.stdout.closed and process.stderr.closed
        assert not any("ATTACH" in statement.upper() for statement in statements)
        assert list(warehouse.db.execute(sa.select(table.c.sys__id, table.c.x))) == [
            *[(i + 1, i) for i in range(2048)],
            (2049, 30),
        ]
        assert warehouse.db.execute_str("PRAGMA integrity_check").fetchone() == ("ok",)
    finally:
        warehouse.db.db.set_trace_callback(None)
        warehouse.db.drop_table(table)


def test_worker_rejects_a_stale_owner_before_inserting(warehouse_tmpfile):
    warehouse = warehouse_tmpfile
    table = warehouse.create_dataset_rows_table("stale_owner", [sa.Column("x", Int)])
    batch = pa.record_batch([[1], [10]], names=["sys__id", "x"])
    buffer = pa.BufferOutputStream()
    with pa.ipc.new_stream(buffer, batch.schema) as stream:
        stream.write_batch(batch)
    data = buffer.getvalue().to_pybytes()
    try:
        result = subprocess.run(  # noqa: S603
            [
                sys.executable,
                "-m",
                "datachain.data_storage.sqlite_arrow",
                warehouse.db.db_file,
                table.name,
                "-1",
            ],
            input=data,
            capture_output=True,
            timeout=30,
            check=False,
        )
        assert result.returncode == 0, result.stderr.decode()
        assert result.stdout == b""  # no acknowledgement of an uncommitted batch
        assert warehouse.table_rows_count(table) == 0
    finally:
        warehouse.db.drop_table(table)


def test_worker_survives_an_automatic_sqlite_rollback(warehouse_tmpfile):
    warehouse = warehouse_tmpfile
    table = warehouse.create_dataset_rows_table("auto_rollback", [sa.Column("x", Int)])
    warehouse.db.execute_str(
        "CREATE TRIGGER fail_batch BEFORE INSERT ON auto_rollback WHEN NEW.x = 99 "
        "BEGIN SELECT RAISE(ROLLBACK, 'forced rollback'); END"
    )
    try:
        with warehouse.writer(table, ["x"]) as writer:
            writer.append_arrow(pa.record_batch([[10] * 2048], names=["x"]))
            with pytest.raises(sqlite3.Error, match="forced rollback"):
                writer.append_arrow(pa.record_batch([[20, 99]], names=["x"]))
            writer.append_arrow(pa.record_batch([[30]], names=["x"]))
        assert list(warehouse.db.execute(sa.select(table.c.x))) == [(10,)] * 2048 + [
            (30,)
        ]
    finally:
        warehouse.db.drop_table(table)


def test_small_batches_eventually_use_native_writer(warehouse_tmpfile, mocker):
    warehouse = warehouse_tmpfile
    table = warehouse.create_dataset_rows_table("small_batches", [sa.Column("x", Int)])
    native = mocker.spy(SQLiteArrowWriter, "write")
    try:
        with warehouse.writer(table, ["x"]) as writer:
            for i in range(3):
                writer.append_arrow(pa.record_batch([[i] * 512], names=["x"]))
                assert native.call_count == i
        assert warehouse.table_rows_count(table) == 1536
    finally:
        warehouse.db.drop_table(table)


def test_dead_child_preserves_the_write_error(warehouse_tmpfile):
    with pytest.raises(sqlite3.OperationalError, match="exited before acknowledging"):
        with closing(
            SQLiteArrowWriter(warehouse_tmpfile.db.db_file, "unused")
        ) as writer:
            writer.process.kill()
            writer.process.wait()
            writer.write(pa.record_batch([[1]], names=["x"]))


def test_arrow_boolean_null_default_with_integer_array(warehouse_tmpfile, mocker):
    warehouse = warehouse_tmpfile
    mocker.patch.object(warehouse, "column_default", return_value=False)
    table = warehouse.create_dataset_rows_table(
        "bool_default", [sa.Column("flag", Boolean)]
    )
    try:
        batch = pa.record_batch(
            [pa.array([True, None, False] * 500, type=pa.bool_())], names=["flag"]
        )
        encoded = warehouse.row_codec([sa.Column("flag", Boolean())]).encode_arrow(
            batch
        )
        assert isinstance(encoded, pa.RecordBatch)
        with warehouse.writer(table, ["flag"]) as writer:
            writer.append_arrow(encoded)
        assert warehouse.db.execute_str(
            "SELECT flag FROM bool_default LIMIT 3"
        ).fetchall() == [(1,), (0,), (0,)]
        assert list(
            warehouse.dataset_rows_select(sa.select(table.c.flag).limit(3))
        ) == [
            (True,),
            (False,),
            (False,),
        ]
    finally:
        warehouse.db.drop_table(table)


def test_in_memory_uri_falls_back_to_row_writer(mocker):
    from datachain.data_storage.sqlite import SQLiteDatabaseEngine, SQLiteWarehouse

    db = sqlite3.connect("file:arrow_mem_test?mode=memory&cache=shared", uri=True)
    engine = sa.create_engine("sqlite+pysqlite:///", creator=lambda: db, future=True)
    db_engine = SQLiteDatabaseEngine(
        engine,
        sa.MetaData(),
        db,
        db_file="file:arrow_mem_test?mode=memory&cache=shared",
    )
    warehouse = SQLiteWarehouse(db=db_engine)
    started = mocker.spy(SQLiteArrowWriter, "__init__")
    table = warehouse.create_dataset_rows_table("mem_uri", [sa.Column("x", Int)])
    try:
        with warehouse.writer(table, ["x"]) as writer:
            writer.append_arrow(pa.record_batch([list(range(2048))], names=["x"]))
        assert started.call_count == 0
        assert warehouse.table_rows_count(table) == 2048
    finally:
        warehouse.close()


def test_worker_enforces_foreign_keys_and_case_sensitive_like(warehouse_tmpfile):
    warehouse = warehouse_tmpfile
    warehouse.db.execute_str("CREATE TABLE parent_fk (id INTEGER PRIMARY KEY)")
    warehouse.db.execute_str("INSERT INTO parent_fk (id) VALUES (1)")
    table = warehouse.create_dataset_rows_table(
        "child_fk", [sa.Column("parent_id", Int), sa.Column("label", String)]
    )
    warehouse.db.execute_str(
        "CREATE TRIGGER enforce_fk_and_like BEFORE INSERT ON child_fk "
        "WHEN NOT EXISTS (SELECT 1 FROM parent_fk WHERE id = NEW.parent_id) "
        "OR ('abc' LIKE 'ABC') "
        "BEGIN SELECT RAISE(ABORT, 'constraint failed: fk or like'); END"
    )
    try:
        with warehouse.writer(table, ["parent_id", "label"]) as writer:
            writer.append_arrow(
                pa.record_batch(
                    [[1] * 1024, ["ok"] * 1024], names=["parent_id", "label"]
                )
            )
            with pytest.raises(sqlite3.IntegrityError):
                writer.append_arrow(
                    pa.record_batch([[2], ["bad"]], names=["parent_id", "label"])
                )
        assert warehouse.table_rows_count(table) == 1024
    finally:
        warehouse.db.drop_table(table)
        warehouse.db.execute_str("DROP TABLE IF EXISTS parent_fk")
