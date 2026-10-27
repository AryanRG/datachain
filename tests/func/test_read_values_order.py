from typing import get_args

import pytest

import datachain as dc
from datachain.lib.convert.values_to_tuples import values_to_tuples


class OrderedValue(dc.DataModel):
    n: int


def test_values_to_tuples_uses_output_order():
    output = {"label": str, "number": int}
    original = list(output.items())

    typ, schema, rows = values_to_tuples(
        output=output, number=[1, 2], label=["one", "two"]
    )

    assert get_args(typ) == (str, int)
    assert list(schema.items()) == original
    assert rows == [("one", 1), ("two", 2)]
    assert list(output.items()) == original


@pytest.mark.parametrize("column", ["", "payload"])
@pytest.mark.parametrize("reverse_output", [False, True])
@pytest.mark.parametrize(
    ("typ", "left", "right"),
    [
        pytest.param(int, [1, 2], [101, 102], id="same-scalars"),
        pytest.param(int | None, [None, 2], [101, None], id="nullable-scalars"),
        pytest.param(list[int], [[], [1, 2]], [[101], []], id="lists"),
        pytest.param(dict[str, int], [{}, {"a": 1}], [{"b": 101}, {}], id="dicts"),
    ],
)
def test_read_values_matches_output_names(
    test_session, column, reverse_output, typ, left, right
):
    output = {"left": typ, "right": typ}
    if reverse_output:
        output = dict(reversed(output.items()))
    original = list(output.items())
    chain = dc.read_values(
        left=left, right=right, output=output, column=column, session=test_session
    )
    prefix = f"{column}." if column else ""
    names = [f"{prefix}left", f"{prefix}right"]
    expected = list(zip(left, right, strict=True))

    assert chain.to_list(*names) == expected
    chain.save("ordered-values")
    restored = dc.read_dataset("ordered-values", session=test_session)
    assert restored.to_list(*names) == expected
    assert list(output.items()) == original


@pytest.mark.parametrize("column", ["", "payload"])
def test_read_values_reordered_different_types(test_session, column):
    chain = dc.read_values(
        number=[1, 2],
        label=["one", "two"],
        output={"label": str, "number": int},
        column=column,
        session=test_session,
    )
    prefix = f"{column}." if column else ""

    assert chain.to_list(f"{prefix}number", f"{prefix}label") == [
        (1, "one"),
        (2, "two"),
    ]


@pytest.mark.parametrize("reverse_output", [False, True])
def test_read_values_model_output_order(test_session, reverse_output):
    left = [OrderedValue(n=1), OrderedValue(n=2)]
    right = [OrderedValue(n=101), OrderedValue(n=102)]
    output = {"left": OrderedValue, "right": OrderedValue}
    if reverse_output:
        output = dict(reversed(output.items()))
    chain = dc.read_values(left=left, right=right, output=output, session=test_session)

    assert chain.to_list("left", "right") == list(zip(left, right, strict=True))
