from __future__ import annotations

from typing import Annotated

import numpy as np
import pytest
from numpy.typing import NDArray

from mcdc.code_factory.numba_layers_generator import (
    _accessor_1d_all,
    _accessor_1d_last,
)
from mcdc.object_.base import MCDCBase
from mcdc.object_.util import check_type, parse_dimension_expression


class DimensionedObject(MCDCBase):
    label = "dimensioned_object"

    G: int
    energy: Annotated[NDArray[np.float64], ("G+1",)]

    def __init__(self, G, energy):
        self.G = G
        self.energy = energy


@pytest.mark.parametrize(
    "expression, expected",
    [
        ("G", ("G", 0)),
        ("G+1", ("G", 1)),
        ("N - 2", ("N", -2)),
    ],
)
def test_parse_dimension_expression(expression, expected):
    assert parse_dimension_expression(expression) == expected


@pytest.mark.parametrize("expression", ["1+G", "G*2", "G+1+2", "G + value"])
def test_parse_dimension_expression_rejects_unsupported_syntax(expression):
    with pytest.raises(ValueError, match="Invalid dimension expression"):
        parse_dimension_expression(expression)


def test_stringified_annotation_resolves_dimension_offset():
    dimensioned = DimensionedObject(2, np.zeros(3))

    assert dimensioned.energy.shape == (3,)


def test_structured_annotation_resolves_dimension_offset():
    dimensioned = DimensionedObject.__new__(DimensionedObject)
    dimensioned.G = 2
    hint = Annotated[NDArray[np.float64], ("G+1",)]

    assert check_type(np.zeros(3), hint, DimensionedObject, dimensioned)
    assert not check_type(np.zeros(2), hint, DimensionedObject, dimensioned)


def test_stringified_annotation_rejects_incorrect_offset_shape(capsys):
    with pytest.raises(SystemExit):
        DimensionedObject(2, np.zeros(2))

    assert "energy must be" in capsys.readouterr().out


def test_generated_accessor_resolves_dimension_offset():
    all_source = _accessor_1d_all("mgxs", "energy", "G+1")
    last_source = _accessor_1d_last("mgxs", "energy", "N - 2")

    assert 'size = mgxs["G"] + 1' in all_source
    assert 'size = mgxs["N"] - 2' in last_source
