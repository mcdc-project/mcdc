import numpy as np
import pytest

import mcdc
import mcdc.numba_types as type_
import mcdc.transport.rng as rng
from mcdc.transport.source import source_particle
from mcdc.transport.util import calculate_angles, find_bin_with_rules


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
        (np.nan, "Source x value must be finite"),
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


def test_transport_source_selects_from_simulation_cdf(prepare_simulation):
    probabilities = [1.0, 2.0, 1.0]
    sources = [
        mcdc.Source(
            position=[float(i), 0.0, 0.0],
            direction=[1.0, 0.0, 0.0],
            probability=probability,
        )
        for i, probability in enumerate(probabilities)
    ]
    simulation_container, data = prepare_simulation(sources=sources)
    simulation = simulation_container[0]
    expected_cdf = np.array([0.0, 0.25, 0.75, 1.0])

    for seed in range(1, 17):
        expected_particle_container = np.zeros(1, dtype=type_.particle)
        expected_particle_container[0]["rng_seed"] = np.uint64(seed)
        xi = rng.lcg(expected_particle_container)
        expected_source_idx = find_bin_with_rules(xi, expected_cdf, 0.0, False)

        particle_container = np.zeros(1, dtype=type_.particle)
        source_particle(
            particle_container,
            np.uint64(seed),
            simulation,
            data,
        )

        assert particle_container[0]["x"] == float(expected_source_idx)


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
    assert source.uniform_polar_cosine
    assert source.uniform_azimuthal
    np.testing.assert_array_equal(source.polar_cosine, [0.8, 1.0])
    np.testing.assert_array_equal(source.azimuthal, [0.0, np.pi])


@pytest.mark.parametrize(
    "name, value",
    [
        ("polar_cosine", 0.5),
        ("azimuthal", 0.25),
    ],
)
def test_scalar_angular_distribution(name, value):
    source = mcdc.Source(
        direction=[0.0, 0.0, 1.0],
        **{name: value},
    )

    assert getattr(source, f"uniform_{name}")
    np.testing.assert_array_equal(getattr(source, name), [value, value])


@pytest.mark.parametrize(
    "name, grid",
    [
        ("polar_cosine", [-1.0, 0.0, 1.0]),
        ("azimuthal", [0.0, np.pi, 2.0 * np.pi]),
    ],
)
def test_piecewise_linear_angular_distribution(name, grid):
    source = mcdc.Source(
        direction=[0.0, 0.0, 1.0],
        **{name: (grid, [0.2, 1.0, 0.4])},
    )

    assert not getattr(source, f"uniform_{name}")
    np.testing.assert_array_equal(getattr(source, name), [grid[0], grid[-1]])
    np.testing.assert_array_equal(getattr(source, f"{name}_pdf").pdf.x, grid)


@pytest.mark.parametrize(
    "polar_cosine",
    [
        -1.1,
        [-1.1, 1.0],
        ([-1.1, 0.0, 1.0], [0.2, 1.0, 0.4]),
    ],
)
def test_polar_cosine_domain(polar_cosine, capsys):
    with pytest.raises(SystemExit):
        mcdc.Source(
            direction=[0.0, 0.0, 1.0],
            polar_cosine=polar_cosine,
        )

    assert "must be within [-1, 1]" in capsys.readouterr().out


def test_transport_source_samples_piecewise_linear_angles(prepare_simulation):
    source = mcdc.Source(
        position=[0.0, 0.0, 0.0],
        direction=[0.0, 0.0, 1.0],
        polar_cosine=([0.5, 1.0], [0.0, 2.0]),
        azimuthal=([0.0, 0.5 * np.pi], [2.0, 0.0]),
    )
    simulation_container, data = prepare_simulation(sources=[source])
    simulation = simulation_container[0]
    packed_source = simulation["sources"][0]

    assert not packed_source["uniform_polar_cosine"]
    assert not packed_source["uniform_azimuthal"]
    assert packed_source["polar_cosine_pdf_ID"] == source.polar_cosine_pdf.ID
    assert packed_source["azimuthal_pdf_ID"] == source.azimuthal_pdf.ID

    sampled_mu = []
    sampled_azimuthal = []
    for seed in range(1, 17):
        particle_container = np.zeros(1, dtype=type_.particle)
        source_particle(
            particle_container,
            np.uint64(seed),
            simulation,
            data,
        )
        mu, azi = calculate_angles(
            particle_container,
            np.array([0.0, 0.0, 1.0]),
        )
        sampled_mu.append(mu)
        sampled_azimuthal.append(azi)

    assert np.all(np.asarray(sampled_mu) >= 0.5)
    assert np.all(np.asarray(sampled_mu) <= 1.0)
    assert np.all(np.asarray(sampled_azimuthal) >= 0.0)
    assert np.all(np.asarray(sampled_azimuthal) <= 0.5 * np.pi)
    assert np.ptp(sampled_mu) > 0.0
    assert np.ptp(sampled_azimuthal) > 0.0


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


@pytest.mark.parametrize(
    "energy_kwargs",
    [
        {
            "energy": 10_000.0,
            "discrete_energy": [[9_999.0, 10_001.0], [0.25, 0.75]],
        },
        {
            "energy": 10_000.0,
            "energy_at_polar_cosine": [9_999.0, 10_001.0],
        },
        {
            "discrete_energy": [[9_999.0, 10_001.0], [0.25, 0.75]],
            "energy_at_polar_cosine": [9_999.0, 10_001.0],
        },
    ],
)
def test_energy_representations_are_mutually_exclusive(energy_kwargs, capsys):
    with pytest.raises(SystemExit):
        mcdc.Source(
            direction=[0.0, 0.0, 1.0],
            polar_cosine=([0.0, 1.0], [1.0, 1.0]),
            **energy_kwargs,
        )

    assert "Cannot specify more than one" in capsys.readouterr().out


def test_energy_at_polar_cosine():
    source = mcdc.Source(
        direction=[0.0, 0.0, 1.0],
        polar_cosine=([-1.0, 0.0, 1.0], [0.2, 1.0, 0.4]),
        energy_at_polar_cosine=[1.0e6, 2.0e6, 4.0e6],
    )

    assert source.energy_at_polar_cosine_active
    assert not source.mono_energetic
    assert not source.discrete_energy
    np.testing.assert_array_equal(
        source.energy_at_polar_cosine,
        [1.0e6, 2.0e6, 4.0e6],
    )
    assert "Energy: Function of polar cosine" in repr(source)


def test_distributed_energy_at_polar_cosine():
    conditional_energy = np.array(
        [
            [[1.0e6, 2.0e6, 3.0e6], [0.0, 1.0, 0.0]],
            [[2.0e6, 4.0e6, 6.0e6], [1.0, 2.0, 1.0]],
            [[5.0e6, 7.0e6, 9.0e6], [0.5, 1.0, 0.5]],
        ]
    )
    source = mcdc.Source(
        direction=[0.0, 0.0, 1.0],
        polar_cosine=([-1.0, 0.0, 1.0], [0.2, 1.0, 0.4]),
        energy_at_polar_cosine=conditional_energy,
    )

    assert source.energy_at_polar_cosine_active
    assert source.energy_at_polar_cosine_is_distribution
    assert not source.mono_energetic
    assert not source.discrete_energy
    assert len(source.energy_at_polar_cosine_distribution.tables) == 3
    for index, table in enumerate(source.energy_at_polar_cosine_distribution.tables):
        np.testing.assert_array_equal(table.pdf.x, conditional_energy[index, 0])
    assert "Energy: PDF conditional on polar cosine" in repr(source)


@pytest.mark.parametrize(
    "kwargs, expected_message",
    [
        (
            {
                "polar_cosine": ([0.0, 1.0], [1.0, 1.0]),
                "energy_at_polar_cosine": [1.0e6, 2.0e6],
            },
            "require direction",
        ),
        (
            {
                "direction": [0.0, 0.0, 1.0],
                "polar_cosine": [0.0, 1.0],
                "energy_at_polar_cosine": [1.0e6, 2.0e6],
            },
            "requires a tabulated polar_cosine PDF",
        ),
        (
            {
                "direction": [0.0, 0.0, 1.0],
                "polar_cosine": ([0.0, 0.5, 1.0], [1.0, 1.0, 1.0]),
                "energy_at_polar_cosine": [1.0e6, 2.0e6],
            },
            "N_mu must equal",
        ),
        (
            {
                "direction": [0.0, 0.0, 1.0],
                "polar_cosine": ([0.0, 1.0], [1.0, 1.0]),
                "energy_at_polar_cosine": [[1.0e6, 2.0e6]],
            },
            "must have shape (N_mu,) or (N_mu, 2, N_energy)",
        ),
        (
            {
                "direction": [0.0, 0.0, 1.0],
                "polar_cosine": ([0.0, 1.0], [1.0, 1.0]),
                "energy_at_polar_cosine": np.ones((2, 3)),
            },
            "must have shape (N_mu,) or (N_mu, 2, N_energy)",
        ),
        (
            {
                "direction": [0.0, 0.0, 1.0],
                "polar_cosine": ([0.0, 1.0], [1.0, 1.0]),
                "energy_at_polar_cosine": np.ones((2, 2, 1)),
            },
            "at least two energy points",
        ),
        (
            {
                "direction": [0.0, 0.0, 1.0],
                "polar_cosine": ([0.0, 1.0], [1.0, 1.0]),
                "energy_at_polar_cosine": [
                    [[1.0, 2.0], [1.0, 1.0]],
                    [[3.0, 2.0], [1.0, 1.0]],
                ],
            },
            "energy grid must be strictly increasing",
        ),
        (
            {
                "direction": [0.0, 0.0, 1.0],
                "polar_cosine": ([0.0, 1.0], [1.0, 1.0]),
                "energy_at_polar_cosine": [1.0e6, np.nan],
            },
            "values must be finite",
        ),
        (
            {
                "direction": [0.0, 0.0, 1.0],
                "polar_cosine": ([0.0, 1.0], [1.0, 1.0]),
                "energy_at_polar_cosine": [1.0e6, -1.0],
            },
            "values must be nonnegative",
        ),
    ],
)
def test_invalid_energy_at_polar_cosine(kwargs, expected_message, capsys):
    with pytest.raises(SystemExit):
        mcdc.Source(**kwargs)

    assert expected_message in capsys.readouterr().out


def test_transport_source_interpolates_energy_at_polar_cosine(prepare_simulation):
    cosine_grid = np.array([-1.0, -0.25, 0.5, 1.0])
    energy_grid = np.array([1.0e6, 2.0e6, 4.0e6, 8.0e6])
    source = mcdc.Source(
        position=[0.0, 0.0, 0.0],
        direction=[0.0, 0.0, 1.0],
        polar_cosine=(cosine_grid, [0.2, 1.0, 0.4, 0.8]),
        energy_at_polar_cosine=energy_grid,
    )
    simulation_container, data = prepare_simulation(sources=[source])
    simulation = simulation_container[0]

    assert simulation["sources"][0]["energy_at_polar_cosine_active"]
    assert simulation["sources"][0]["energy_at_polar_cosine_length"] == len(energy_grid)

    for seed in range(1, 33):
        particle_container = np.zeros(1, dtype=type_.particle)
        source_particle(
            particle_container,
            np.uint64(seed),
            simulation,
            data,
        )
        mu, _ = calculate_angles(
            particle_container,
            np.array([0.0, 0.0, 1.0]),
        )
        expected_energy = np.interp(mu, cosine_grid, energy_grid)
        np.testing.assert_allclose(
            particle_container[0]["E"],
            expected_energy,
            rtol=1.0e-13,
        )


def test_transport_source_samples_distributed_energy_at_polar_cosine(
    prepare_simulation,
):
    cosine_grid = np.array([-1.0, 0.0, 1.0])
    energy_bounds = np.array(
        [
            [1.0e6, 2.0e6],
            [3.0e6, 5.0e6],
            [7.0e6, 10.0e6],
        ]
    )
    conditional_energy = np.stack(
        [
            energy_bounds,
            np.ones_like(energy_bounds),
        ],
        axis=1,
    )
    source = mcdc.Source(
        position=[0.0, 0.0, 0.0],
        direction=[0.0, 0.0, 1.0],
        polar_cosine=(cosine_grid, [1.0, 1.0, 1.0]),
        energy_at_polar_cosine=conditional_energy,
    )
    simulation_container, data = prepare_simulation(sources=[source])
    simulation = simulation_container[0]

    assert simulation["sources"][0]["energy_at_polar_cosine_active"]
    assert simulation["sources"][0]["energy_at_polar_cosine_is_distribution"]

    for seed in range(1, 65):
        particle_container = np.zeros(1, dtype=type_.particle)
        source_particle(
            particle_container,
            np.uint64(seed),
            simulation,
            data,
        )
        mu, _ = calculate_angles(
            particle_container,
            np.array([0.0, 0.0, 1.0]),
        )
        minimum = np.interp(mu, cosine_grid, energy_bounds[:, 0])
        maximum = np.interp(mu, cosine_grid, energy_bounds[:, 1])
        assert minimum <= particle_container[0]["E"] <= maximum


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
    assert source.uniform_time
    np.testing.assert_array_equal(source.time_range, [1.0, 2.0])


@pytest.mark.parametrize(
    "time",
    [
        [[1.0, 2.0, 3.0], [0.2, 1.0, 0.4]],
        ([1.0, 2.0, 3.0], [0.2, 1.0, 0.4]),
        np.array([[1.0, 2.0, 3.0], [0.2, 1.0, 0.4]]),
    ],
)
def test_time_distribution(time):
    source = mcdc.Source(time=time)

    assert not source.discrete_time
    assert not source.uniform_time
    np.testing.assert_array_equal(source.time_range, [1.0, 3.0])
    np.testing.assert_array_equal(source.time_pdf.pdf.x, [1.0, 2.0, 3.0])
    assert "Time: PDF" in repr(source)


def test_transport_source_samples_piecewise_linear_time(prepare_simulation):
    source = mcdc.Source(
        position=[0.0, 0.0, 0.0],
        direction=[0.0, 0.0, 1.0],
        time=([1.0, 2.0], [0.0, 2.0]),
    )
    simulation_container, data = prepare_simulation(sources=[source])
    simulation = simulation_container[0]
    packed_source = simulation["sources"][0]

    assert not packed_source["uniform_time"]
    assert packed_source["time_pdf_ID"] == source.time_pdf.ID

    sampled_time = []
    for seed in range(1, 17):
        particle_container = np.zeros(1, dtype=type_.particle)
        source_particle(
            particle_container,
            np.uint64(seed),
            simulation,
            data,
        )
        sampled_time.append(particle_container[0]["t"])

    assert np.all(np.asarray(sampled_time) >= 1.0)
    assert np.all(np.asarray(sampled_time) <= 2.0)
    assert np.ptp(sampled_time) > 0.0


@pytest.mark.parametrize(
    "kwargs, expected_message",
    [
        ({"energy": [1.0, 2.0, 3.0]}, "Energy distribution must have shape (2, N)"),
        (
            {"discrete_energy": [1.0, 2.0, 3.0]},
            "Discrete energy distribution must have shape (2, N)",
        ),
        ({"time": [1.0, 2.0, 3.0]}, "Source time must be a scalar"),
    ],
)
def test_invalid_distribution_shape(kwargs, expected_message, capsys):
    with pytest.raises(SystemExit):
        mcdc.Source(**kwargs)

    assert expected_message in capsys.readouterr().out
