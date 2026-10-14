import pickle
from collections.abc import Iterator
from datetime import datetime, timezone

import pytest
import sqlalchemy as sa
from pydantic import BaseModel

import datachain as dc
from datachain.data_storage.sqlite import SQLiteWarehouse
from datachain.lib.convert.flatten import iter_flat_columns


class Info(BaseModel):
    a: int
    b: str = "x"


class Meta(BaseModel):
    name: str
    info: Info | None = None
    tags: list[str] = []


def test_row_type_fields_are_the_flat_columns():
    meta_row = dc.row_type(Meta)

    assert meta_row._fields == ("name", "info___type_tag", "info__a", "info__b", "tags")
    assert meta_row._field_defaults == {
        "info___type_tag": 1,
        "info__a": None,
        "info__b": None,
        "tags": [],
    }
    assert dc.row_type(dc.File)._fields == tuple(
        "__".join(c.path) for c in iter_flat_columns(dc.File)
    )
    assert dc.row_type(Meta) is meta_row


def test_row_type_rows_pickle():
    row = dc.row_type(dc.File)("s3://bucket", "a.txt")
    assert pickle.loads(pickle.dumps(row)) == row  # noqa: S301


def test_gen_rows_match_gen_models(test_session):
    meta_row = dc.row_type(Meta)

    def models(n: int) -> Iterator[Meta]:
        yield Meta(name=f"m{n}", info=Info(a=n), tags=["t"])
        yield Meta(name=f"e{n}")

    def rows(n: int) -> Iterator[Meta]:
        yield meta_row(f"m{n}", 0, n, "x", ["t"])
        yield meta_row(f"e{n}")

    chain = dc.read_values(n=[1, 2], session=test_session)
    expected = chain.gen(meta=models).order_by("meta.name").to_values("meta")

    assert chain.gen(meta=rows).order_by("meta.name").to_values("meta") == expected


def test_rows_in_multiple_and_optional_outputs(test_session):
    meta_row = dc.row_type(Meta)

    def fn(n: int) -> tuple[Meta | None, float]:
        return (meta_row(f"m{n}") if n else None), n / 2

    result = (
        dc.read_values(n=[0, 1], session=test_session)
        .map(fn, output={"meta": Meta | None, "score": float})
        .order_by("score")
        .to_list("meta", "score")
    )

    assert result == [(None, 0.0), (Meta(name="m1"), 0.5)]


def test_row_of_another_model_is_rejected(test_session):
    chain = dc.read_values(n=[1], session=test_session).map(
        meta=lambda n: dc.row_type(Info)(n), output=Meta
    )

    with pytest.raises(TypeError, match="InfoRow"):
        chain.save("wrong")


def _stored(chain: dc.DataChain, session) -> list[tuple]:
    dataset = chain.dataset
    assert dataset is not None
    warehouse = session.catalog.warehouse
    table = warehouse.dataset_rows(dataset, dataset.latest_version).get_table()
    cols = [c for c in table.c if not c.name.startswith("sys__")]
    query = sa.select(*cols, *(sa.func.typeof(c) for c in cols))
    return sorted(warehouse.db.execute(query), key=repr)


def _files(n: int, as_rows: bool) -> Iterator[dc.File]:
    file_row = dc.row_type(dc.File)
    when = datetime(2026, 10, 13, 1, 2, 3, 456789, tzinfo=timezone.utc)
    for i in range(n):
        args = ("s3://bucket", f"{n}/{i}", i, "", f"e{i}", bool(i % 2), when)
        if as_rows and i != 1:
            yield file_row(*args)
        else:
            yield dc.File(**dict(zip(file_row._fields, args, strict=False)))


@pytest.mark.parametrize("insert_tuples", [True, False])
def test_gen_rows_are_stored_as_models(test_session, monkeypatch, insert_tuples):
    if not insert_tuples:
        monkeypatch.setattr(SQLiteWarehouse, "insert_columns", lambda self, t: None)
    meta_row = dc.row_type(Meta)

    def models(n: int) -> Iterator[Meta]:
        yield from (Meta(name=f"m{n}", info=Info(a=n), tags=["t"]),) * n

    def rows(n: int) -> Iterator[Meta]:
        yield from (meta_row(f"m{n}", 0, n, "x", ["t"]),) * n

    chain = dc.read_values(n=[0, 1, 3], session=test_session)
    assert _stored(chain.gen(meta=rows).save("rows"), test_session) == _stored(
        chain.gen(meta=models).save("models"), test_session
    )

    files = chain.gen(file=lambda n: _files(n, True), output=dc.File).save("files")
    assert _stored(files, test_session) == _stored(
        chain.gen(file=lambda n: _files(n, False), output=dc.File).save("objects"),
        test_session,
    )


def test_gen_rows_are_inserted_as_tuples(test_session, mocker):
    spy = mocker.spy(SQLiteWarehouse, "_insert_tuples_sql")

    dc.read_values(n=[3], session=test_session).gen(
        file=lambda n: _files(n, True), output=dc.File
    ).save("files")

    assert spy.call_count == 2  # the File object splits the batch
