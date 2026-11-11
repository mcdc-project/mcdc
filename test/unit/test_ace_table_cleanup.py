"""Outgoing histogram cleanup preserves support and aligned conditional data."""

from pathlib import Path
import sys
import warnings

import h5py
import numpy as np
import pytest

from mcdc.object_.hdf5_reader import read_distribution

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))


def source_table(
    kind,
    incident=7.4,
    *,
    energy=None,
    pdf=None,
    cdf=None,
    interpolation=1,
    discrete=0,
    invalid_cosine=None,
):
    ACEtk = pytest.importorskip("ACEtk")
    energy = [1.4, 0.0001, 0.2, 1.6, 1.8, 2.0] if energy is None else energy
    pdf = [1.0e-15] * 3 + [2.5] * 3 if pdf is None else pdf
    if cdf is None:
        cdf = np.r_[0.0, np.cumsum(np.diff(energy) * np.asarray(pdf)[:-1])]
    if kind == 4:
        return ACEtk.continuous.TabulatedEnergyDistribution(
            incident, interpolation, energy, pdf, list(cdf), discrete
        )
    if kind == 44:
        return ACEtk.continuous.TabulatedKalbachMannDistribution(
            incident,
            interpolation,
            energy,
            pdf,
            list(cdf),
            np.linspace(0.0, 0.5, len(energy)).tolist(),
            np.arange(1.0, len(energy) + 1).tolist(),
            discrete,
        )
    angles = []
    for i, (outgoing, density, cumulative) in enumerate(zip(energy, pdf, cdf)):
        right = 0.1 * (i + 1)
        cosines = [right, -1.0] if i == invalid_cosine else [-1.0, right]
        angles.append(
            ACEtk.continuous.TabulatedAngularDistributionWithProbability(
                outgoing,
                density,
                cumulative,
                2,
                cosines,
                [1.0 / (right + 1.0)] * 2,
                [0.0, 1.0],
            )
        )
    return ACEtk.continuous.TabulatedEnergyAngleDistribution(
        incident, interpolation, angles
    )


def write_distribution(group, kind, tables):
    import ACEtk
    from data_library_generator.common.ace_converter import write_energy_distribution

    cls = {
        4: ACEtk.continuous.OutgoingEnergyDistributionData,
        44: ACEtk.continuous.KalbachMannDistributionData,
        61: ACEtk.continuous.EnergyAngleDistributionData,
    }[kind]
    write_energy_distribution(group, cls([], [], tables))


@pytest.mark.parametrize("kind", [4, 44, 61])
def test_negligible_prefix_preserves_support_and_conditional_data(tmp_path, kind):
    normal = source_table(kind, 5.0, energy=[0.0, 1.0, 2.0], pdf=[0.5] * 3)
    malformed = source_table(kind)
    original = np.asarray(malformed.xss_array).copy()
    with h5py.File(tmp_path / "distribution.h5", "w") as file:
        group = file.create_group("reactions/reaction-22/energy-distribution")
        with pytest.warns(RuntimeWarning) as caught:
            write_distribution(group, kind, [normal, malformed])
        assert len(caught) == 1
        message = str(caught[0].message)
        for text in (
            group.name,
            f"ACE LAW {kind}",
            "table 1",
            "7.4 MeV",
            "discarded",
            "below-support",
            "probability bound",
        ):
            assert text in message
        outgoing = group["tables"] if kind == 4 else group["tables/outgoing-energy"]
        np.testing.assert_array_equal(outgoing["offsets"], [0, 3, 7])
        np.testing.assert_allclose(
            outgoing["value"], np.array([0, 1, 2, 1.4, 1.6, 1.8, 2]) * 1e6
        )
        np.testing.assert_allclose(outgoing["pdf"][3:], [1e-21, 2.5e-6, 2.5e-6, 2.5e-6])
        if kind == 44:
            for name, expected in (
                ("precompound-factor", [0.0, 0.3, 0.4, 0.5]),
                ("angular-slope", [1.0, 4.0, 5.0, 6.0]),
            ):
                parameter = group[f"tables/{name}/1"]
                np.testing.assert_allclose(
                    parameter["grid"], [1.4e6, 1.6e6, 1.8e6, 2e6]
                )
                np.testing.assert_allclose(parameter["value"], expected)
        elif kind == 61:
            angular = group["tables/cosine-distributions"]
            np.testing.assert_array_equal(angular["offsets"], np.arange(0, 15, 2))
            np.testing.assert_allclose(angular["value"][7::2], [0.1, 0.4, 0.5, 0.6])
        result = read_distribution(group)
    # Both original support endpoints determine incident-energy support scaling.
    values = result.value if kind == 4 else result.energy_out
    np.testing.assert_allclose(values[3:], [1.4e6, 1.6e6, 1.8e6, 2e6])
    if kind == 44:
        np.testing.assert_allclose(result.precompound_tables[1].y, [0, 0.3, 0.4, 0.5])
        np.testing.assert_allclose(result.angular_slope_tables[1].y, [1, 4, 5, 6])
    elif kind == 61:
        np.testing.assert_allclose(result.cosine[7::2], [0.1, 0.4, 0.5, 0.6])
    np.testing.assert_array_equal(malformed.xss_array, original)


@pytest.mark.parametrize("kind", [4, 44, 61])
def test_ordered_tables_are_unchanged_and_silent(tmp_path, kind):
    tables = [
        source_table(kind, incident, energy=[0.0, 1.0, 2.0], pdf=[0.5] * 3)
        for incident in (5.0, 7.4)
    ]
    with h5py.File(tmp_path / "normal.h5", "w") as group:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            write_distribution(group, kind, tables)
        assert not caught
        outgoing = group["tables"] if kind == 4 else group["tables/outgoing-energy"]
        np.testing.assert_array_equal(outgoing["value"][3:], [0, 1e6, 2e6])


@pytest.mark.parametrize("kind", [4, 44, 61])
@pytest.mark.parametrize(
    "changes",
    [
        {"interpolation": 2},
        {"pdf": [1e-3] * 3 + [2.5] * 3},
        {"cdf": [0.0, -1e-5, -1e-5, 0.0, 0.5, 1.0]},
        # Signed prefix masses cancel, but their absolute probability is large.
        {"pdf": [1.0, 1.0, 1.2 / 1.4, 2.5, 2.5, 2.5], "cdf": [0, 0, 0, 0, 0.5, 1]},
        {"energy": [1.4, 0.0001, 0.2, 1.6, 1.5, 2.0]},
    ],
    ids=[
        "linear",
        "significant-pdf",
        "significant-cdf",
        "cancellation",
        "later-inversion",
    ],
)
def test_non_negligible_or_unsupported_inversions_still_fail(tmp_path, kind, changes):
    normal = source_table(kind, 5.0, energy=[0.0, 1.0, 2.0], pdf=[0.5] * 3)
    malformed = source_table(kind, **changes)
    with h5py.File(tmp_path / "invalid.h5", "w") as group:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with pytest.raises(
                ValueError, match="PDF coordinates must be nondecreasing"
            ):
                write_distribution(group, kind, [normal, malformed])
        assert not caught


@pytest.mark.parametrize("kind", [4, 44])
def test_discrete_tables_are_excluded_from_cleanup(tmp_path, kind):
    normal = source_table(kind, 5.0, energy=[0.0, 1.0, 2.0], pdf=[0.5] * 3)
    malformed = source_table(
        kind,
        energy=[3.0, 1.4, 0.0001, 0.2, 1.6, 1.8, 2.0],
        pdf=[0.2] + [1e-15] * 3 + [2.0] * 3,
        cdf=[0.2] * 5 + [0.6, 1.0],
        discrete=1,
    )
    with h5py.File(tmp_path / "discrete.h5", "w") as group:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with pytest.raises(
                ValueError, match="PDF coordinates must be nondecreasing"
            ):
                write_distribution(group, kind, [normal, malformed])
        assert not caught


def test_cosine_error_after_cleanup_uses_original_source_coordinates(tmp_path):
    tables = [source_table(61, 5.0), source_table(61, invalid_cosine=4)]
    with h5py.File(tmp_path / "invalid-cosine.h5", "w") as group:
        with pytest.warns(RuntimeWarning) as caught:
            with pytest.raises(ValueError) as error:
                write_distribution(group, 61, tables)
    assert len(caught) == 2
    message = str(error.value)
    assert "ACE LAW 61" in message
    assert "cosine table 4 (E_in = 7.4 MeV, E_out = 1.8 MeV)" in message
    assert "PDF coordinates must be nondecreasing" in message


def test_histogram_cleanup_is_independent_of_coordinate_units():
    from data_library_generator.common.ace_converter import histogram_support_rows

    energy = np.array([1.4, 0.0001, 0.2, 1.6, 1.8, 2.0])
    pdf = np.array([1e-15] * 3 + [2.5] * 3)
    cdf = np.r_[0.0, np.cumsum(np.diff(energy) * pdf[:-1])]
    rows, bound = histogram_support_rows(energy, pdf, cdf)
    converted_rows, converted_bound = histogram_support_rows(
        energy * 1e6, pdf / 1e6, cdf
    )
    np.testing.assert_array_equal(rows, [0, 3, 4, 5])
    np.testing.assert_array_equal(converted_rows, rows)
    assert converted_bound == pytest.approx(bound, rel=1e-14)
    assert 0 < bound <= 1e-12
