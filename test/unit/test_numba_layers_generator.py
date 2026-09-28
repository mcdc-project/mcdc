from typing import Annotated

import numba as nb
import numpy as np
import pytest
from numpy.typing import NDArray

from mcdc.code_factory.numba_layers_generator import (
    AccessorTarget,
    _accessor_1d_all,
    _accessor_1d_element,
    _accessor_1d_last,
    _accessor_2d_element,
    _accessor_3d_element,
    _accessor_4d_element,
    set_object,
    set_structure,
    validate_unique_class_labels,
    validate_accessor_targets,
)
from mcdc.mcdc_get import cell as get_cell
from mcdc.object_.base import MCDCBase
from mcdc.object_.surface import Surface


class EmbeddedLeaf(MCDCBase):
    label = "child"

    active: bool
    values: NDArray[np.float64]

    def __init__(self):
        self.active = True
        self.values = np.array([2.0, 3.0])


class EmbeddedOwner(MCDCBase):
    label = "embedded_owner_test"

    child: EmbeddedLeaf

    def __init__(self):
        self.child = EmbeddedLeaf()


class DuplicateLabelA(MCDCBase):
    label = "duplicate_label"


class DuplicateLabelB(MCDCBase):
    label = "duplicate_label"


def test_set_structure_tracks_logical_accessor_types():
    annotations = {
        "example": {
            "integer_values": NDArray[np.int64],
            "float_values": NDArray[np.float64],
            "integer_grid": Annotated[NDArray[np.int64], ("Nx", "Ny", "Nz")],
            "surfaces": list[Surface],
        }
    }
    structures = {"example": []}
    accessor_targets = {"example": []}

    set_structure("example", structures, accessor_targets, annotations)

    assert accessor_targets["example"] == [
        ("integer_values", ("integer_values_length",), True),
        ("float_values", ("float_values_length",), False),
        ("integer_grid", ("Nx", "Ny", "Nz"), True),
        ("surface_IDs", ("N_surface",), True),
    ]


def test_embedded_mcdc_base_structures_and_records_are_recursive():
    annotations = {
        EmbeddedOwner.label: {"child": EmbeddedLeaf},
        EmbeddedLeaf.label: {
            "active": bool,
            "values": NDArray[np.float64],
        },
    }
    structures = {label: [] for label in annotations}
    accessor_targets = {label: [] for label in annotations}
    records = {label: {} for label in annotations}
    order = []

    set_structure(
        EmbeddedOwner.label,
        structures,
        accessor_targets,
        annotations,
        order=order,
    )

    owner = EmbeddedOwner()
    data = {"size": 0}
    set_object(owner, annotations, structures, records, data)
    data = {"size": 0, "array": np.zeros(data["size"])}
    set_object(owner, annotations, structures, records, data, set_data=True)

    assert order == [EmbeddedLeaf.label, EmbeddedOwner.label]
    assert structures[EmbeddedOwner.label][0][0] == "child"
    assert records[EmbeddedOwner.label]["child"]["active"]
    np.testing.assert_array_equal(data["array"], owner.child.values)


def test_runtime_class_labels_must_be_unique(capsys):
    with pytest.raises(SystemExit):
        validate_unique_class_labels([DuplicateLabelA, DuplicateLabelB])

    assert "Duplicate MC/DC class label 'duplicate_label'" in capsys.readouterr().out


def test_scalar_integer_getters_cast_values_from_data():
    assert "return int64(data[offset + index])" in _accessor_1d_element(
        "example", "values", cast_to_int=True
    )
    assert "return int64(data[end - 1])" in _accessor_1d_last(
        "example", "values", "values_length", cast_to_int=True
    )
    assert "return int64(data[offset + index_1 * stride + index_2])" in (
        _accessor_2d_element("example", "values", "Ny", cast_to_int=True)
    )
    assert "return int64(data[offset + index_1 * stride_2 * stride_3" in (
        _accessor_3d_element("example", "values", "Ny", "Nz", cast_to_int=True)
    )


def test_float_and_bulk_getters_remain_zero_copy_views():
    assert "return data[offset + index]" in _accessor_1d_element("example", "values")
    assert "return array_result(data[start:end])" in _accessor_1d_all(
        "example", "values", "values_length", nb.types.float64
    )


def test_mixed_literal_and_named_strides_are_generated():
    getter_3d = _accessor_3d_element("example", "values", 3, "Nz")
    getter_4d = _accessor_4d_element("example", "values", 2, "Nz", 4)

    assert "stride_2 = 3" in getter_3d
    assert 'stride_3 = example["Nz"]' in getter_3d
    assert "stride_2 = 2" in getter_4d
    assert 'stride_3 = example["Nz"]' in getter_4d
    assert "stride_4 = 4" in getter_4d


def test_unsupported_accessor_rank_is_rejected_before_generation():
    targets = {
        "example": [AccessorTarget("values", ("N1", "N2", "N3", "N4", "N5"), False)]
    }

    with pytest.raises(ValueError, match="one through four dimensions"):
        validate_accessor_targets(targets)


def test_generated_integer_getter_returns_int_and_bulk_getter_returns_view():
    cell = np.zeros(
        1,
        dtype=[("surface_IDs_offset", np.int64), ("N_surface", np.int64)],
    )[0]
    cell["N_surface"] = 2
    data = np.array([3.0, 7.0])

    surface_ID = get_cell.surface_IDs(1, cell, data)
    surface_IDs = get_cell.surface_IDs_all(cell, data)

    assert isinstance(surface_ID, (int, np.integer))
    assert surface_ID == 7
    assert surface_IDs.dtype == np.float64
    assert np.shares_memory(surface_IDs, data)
