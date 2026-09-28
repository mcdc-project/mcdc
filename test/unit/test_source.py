import numpy as np
import pytest

import mcdc
import mcdc.numba_types as type_
from mcdc.transport.source import source_particle


@pytest.mark.parametrize("coordinate", ["x", "y", "z"])
def test_position_and_box_bounds_are_mutually_exclusive(coordinate, capsys):
    with pytest.raises(SystemExit):
        mcdc.Source(position=[0.0, 0.0, 0.0], **{coordinate: [-1.0, 1.0]})

    assert "Cannot specify position together with x, y, or z" in capsys.readouterr().out


@pytest.mark.parametrize("coordinate", ["x", "y", "z"])
def test_scalar_spatial_coordinate(coordinate):
    source = mcdc.Source(**{coordinate: 2.5})

    assert not source.point_source
    assert getattr(source, f"uniform_{coordinate}")
    np.testing.assert_array_equal(getattr(source, coordinate), [2.5, 2.5])


@pytest.mark.parametrize("coordinate", ["x", "y", "z"])
def test_piecewise_linear_spatial_distribution(coordinate):
    source = mcdc.Source(
        **{
            coordinate: (
                [0.0, 5.0, 10.0],
                [0.2, 1.0, 0.4],
            )
        }
    )

    assert not source.point_source
    assert not getattr(source, f"uniform_{coordinate}")
    np.testing.assert_array_equal(getattr(source, coordinate), [0.0, 10.0])
    np.testing.assert_array_equal(
        getattr(source, f"{coordinate}_pdf").pdf.x,
        [0.0, 5.0, 10.0],
    )

    for other_coordinate in {"x", "y", "z"} - {coordinate}:
        assert getattr(source, f"uniform_{other_coordinate}")
        np.testing.assert_array_equal(
            getattr(source, other_coordinate),
            [0.0, 0.0],
        )


@pytest.mark.parametrize(
    "x, expected_message",
    [
        ([0.0, 1.0, 2.0], "Source x must be a scalar"),
        ([1.0, 0.0], "Source x bounds must satisfy min <= max"),
        (np.nan, "Source x coordinate must be finite"),
        (([0.0, 0.0], [1.0, 1.0]), "must be strictly increasing"),
        (([0.0, 1.0], [-1.0, 1.0]), "PDF must be nonnegative"),
    ],
)
def test_invalid_spatial_distribution(x, expected_message, capsys):
    with pytest.raises(SystemExit):
        mcdc.Source(x=x)

    assert expected_message in capsys.readouterr().out


def test_transport_source_samples_piecewise_linear_coordinate(prepare_simulation):
    source = mcdc.Source(
        x=([0.0, 1.0], [0.0, 2.0]),
        y=2.0,
        z=3.0,
        direction=[1.0, 0.0, 0.0],
    )
    simulation_container, data = prepare_simulation(sources=[source])
    simulation = simulation_container[0]
    packed_source = simulation["sources"][0]

    assert not packed_source["uniform_x"]
    assert packed_source["uniform_y"]
    assert packed_source["uniform_z"]
    assert packed_source["x_pdf_ID"] == source.x_pdf.ID

    sampled_x = []
    for seed in range(1, 17):
        particle_container = np.zeros(1, dtype=type_.particle)
        source_particle(
            particle_container,
            np.uint64(seed),
            simulation,
            data,
        )
        sampled_x.append(particle_container[0]["x"])
        assert particle_container[0]["y"] == 2.0
        assert particle_container[0]["z"] == 3.0

    assert np.all(np.asarray(sampled_x) >= 0.0)
    assert np.all(np.asarray(sampled_x) <= 1.0)
    assert np.ptp(sampled_x) > 0.0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"isotropic": True, "direction": [0.0, 0.0, 1.0]},
        {"isotropic": True, "white_direction": [0.0, 0.0, 1.0]},
        {
            "direction": [0.0, 0.0, 1.0],
            "white_direction": [0.0, 0.0, 1.0],
        },
    ],
)
def test_direction_representations_are_mutually_exclusive(kwargs, capsys):
    with pytest.raises(SystemExit):
        mcdc.Source(**kwargs)

    assert "Cannot specify more than one" in capsys.readouterr().out


@pytest.mark.parametrize(
    "kwargs",
    [
        {"polar_cosine": [0.8, 1.0]},
        {"azimuthal": [0.0, np.pi]},
        {
            "white_direction": [0.0, 0.0, 1.0],
            "polar_cosine": [0.8, 1.0],
        },
    ],
)
def test_angular_bounds_require_direction(kwargs, capsys):
    with pytest.raises(SystemExit):
        mcdc.Source(**kwargs)

    assert "polar_cosine and azimuthal require direction" in capsys.readouterr().out


def test_direction_accepts_angular_bounds():
    source = mcdc.Source(
        direction=[0.0, 0.0, 1.0],
        polar_cosine=[0.8, 1.0],
        azimuthal=[0.0, np.pi],
    )

    assert not source.isotropic_direction
    assert not source.mono_direction
    np.testing.assert_array_equal(source.polar_cosine, [0.8, 1.0])
    np.testing.assert_array_equal(source.azimuthal, [0.0, np.pi])


@pytest.mark.parametrize(
    "energy",
    [
        10_000,
        10_000.0,
        np.int64(10_000),
        np.float64(10_000.0),
    ],
)
def test_scalar_energy(energy):
    source = mcdc.Source(energy=energy)

    assert source.mono_energetic
    assert isinstance(source.energy, float)
    assert source.energy == 10_000.0


@pytest.mark.parametrize(
    "energy",
    [
        [[9_999.0, 10_001.0], [0.5, 0.5]],
        ([9_999.0, 10_001.0], [0.5, 0.5]),
        np.array([[9_999.0, 10_001.0], [0.5, 0.5]]),
    ],
)
def test_energy_distribution(energy):
    source = mcdc.Source(energy=energy)

    assert not source.mono_energetic
    assert not source.discrete_energy
    np.testing.assert_array_equal(source.energy_pdf.pdf.x, [9_999.0, 10_001.0])


@pytest.mark.parametrize(
    "discrete_energy",
    [
        [[9_999.0, 10_001.0], [0.25, 0.75]],
        ([9_999.0, 10_001.0], [0.25, 0.75]),
        np.array([[9_999.0, 10_001.0], [0.25, 0.75]]),
    ],
)
def test_discrete_energy_distribution(discrete_energy):
    source = mcdc.Source(discrete_energy=discrete_energy)

    assert not source.mono_energetic
    assert source.discrete_energy
    np.testing.assert_array_equal(source.energy_pmf.value, [9_999.0, 10_001.0])
    assert "Energy: PMF" in repr(source)


def test_energy_and_discrete_energy_are_mutually_exclusive(capsys):
    with pytest.raises(SystemExit):
        mcdc.Source(
            energy=10_000.0,
            discrete_energy=[[9_999.0, 10_001.0], [0.25, 0.75]],
        )

    assert "Cannot specify both energy and discrete_energy" in capsys.readouterr().out


def test_transport_source_sets_mono_energy(prepare_simulation):
    source = mcdc.Source(
        position=[0.0, 0.0, 0.0],
        direction=[0.0, 0.0, 1.0],
        energy=10_000.0,
    )
    simulation_container, data = prepare_simulation(sources=[source])
    particle_container = np.zeros(1, dtype=type_.particle)

    source_particle(
        particle_container,
        np.uint64(1),
        simulation_container[0],
        data,
    )

    assert particle_container[0]["E"] == 10_000.0


def test_transport_source_samples_discrete_energy(prepare_simulation):
    source = mcdc.Source(
        position=[0.0, 0.0, 0.0],
        direction=[0.0, 0.0, 1.0],
        discrete_energy=([10_001.0], [1.0]),
    )
    simulation_container, data = prepare_simulation(sources=[source])
    particle_container = np.zeros(1, dtype=type_.particle)

    source_particle(
        particle_container,
        np.uint64(1),
        simulation_container[0],
        data,
    )

    assert particle_container[0]["E"] == 10_001.0


@pytest.mark.parametrize("time", [2, 2.0, np.int64(2), np.float64(2.0)])
def test_scalar_time(time):
    source = mcdc.Source(time=time)

    assert source.discrete_time
    assert source.time == 2.0


@pytest.mark.parametrize(
    "time",
    [
        [1.0, 2.0],
        (1.0, 2.0),
        np.array([1.0, 2.0]),
    ],
)
def test_time_interval(time):
    source = mcdc.Source(time=time)

    assert not source.discrete_time
    np.testing.assert_array_equal(source.time_range, [1.0, 2.0])


@pytest.mark.parametrize(
    "kwargs, expected_message",
    [
        ({"energy": [1.0, 2.0, 3.0]}, "Energy distribution must have shape (2, N)"),
        (
            {"discrete_energy": [1.0, 2.0, 3.0]},
            "Discrete energy distribution must have shape (2, N)",
        ),
        ({"time": [1.0, 2.0, 3.0]}, "Source time interval must have shape (2,)"),
    ],
)
def test_invalid_distribution_shape(kwargs, expected_message, capsys):
    with pytest.raises(SystemExit):
        mcdc.Source(**kwargs)

    assert expected_message in capsys.readouterr().out
