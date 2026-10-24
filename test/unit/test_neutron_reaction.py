"""Neutron reader and packed transport checks using self-contained mcdc-data files."""

import h5py
import numpy as np
import pytest

import mcdc
from mcdc.constant import NEUTRON_REACTION_TOTAL, NEUTRON_REACTION_FISSION
from mcdc.object_.hdf5_reader import check_format, read_data, numbered_groups
from mcdc.transport.data import evaluate_data_clamped
from mcdc.transport.physics.neutron import native


def interpolation(group, laws=("linear",), boundaries=(3,)):
    rule = group.create_group("interpolation")
    rule["laws"] = np.array(laws, dtype=h5py.string_dtype())
    rule["boundaries"] = boundaries


def constant(group, value):
    group.attrs["type"] = "polynomial"
    group["coefficients"] = [value]


def basic_reaction(
    parent,
    name,
    category,
    values=(1.0, 1.0, 1.0),
    offset=0,
    laws=("linear",),
    boundaries=(3,),
):
    group = parent.create_group(name)
    group.attrs["category"] = category
    group["reference-frame"] = "LAB"
    group["Q-value"] = 2.0e6
    xs = group.create_group("xs")
    xs["offset"], xs["value"] = offset, values
    interpolation(xs, laws, boundaries)
    group.create_group("cross-products")
    return group


def add_fission(parent, name, yield_):
    group = basic_reaction(parent, name, "fission")
    group.create_group("prompt-angular-cosine-distribution").attrs["type"] = "isotropic"
    spectrum = group.create_group("prompt-energy-distribution")
    spectrum.attrs["type"] = "level-scattering"
    spectrum["C1"], spectrum["C2"] = 0.0, 0.5
    constant(group.create_group("prompt-multiplicity"), yield_)
    constant(group.create_group("delayed-multiplicity"), 0.0)
    delayed = group.create_group("delayed-neutrons")
    delayed["fractions"], delayed["decay-rates"] = np.zeros(0), np.zeros(0)
    return group


@pytest.fixture
def library(tmp_path, monkeypatch):
    """Provide identity and projectile files without requiring downloaded ACE data."""
    monkeypatch.setenv("MCDC_LIB", str(tmp_path))
    for name in ("H", "H2", "H2-293.6K"):
        with h5py.File(tmp_path / f"{name}.h5", "w") as file:
            file.attrs["format"], file.attrs["format-version"] = "mcdc-data", "0.1.0"
            if name == "H":
                file["atomic-number"] = 1
            elif name == "H2":
                file["mass-number"], file["atomic-weight-ratio"] = 2, 1.9968
                file["fissionable"], file["excitation-level"] = True, 0
            else:
                file["temperature"], file["kT"] = 293.6, 0.0253
                neutron = file.create_group("projectiles/neutron")
                neutron["xs-energy-grid"] = [1.0, 10.0, 100.0]
                neutron.create_group("reactions")
    return tmp_path


@pytest.fixture
def hydrogen_library(library):
    """Provide two hydrogen isotopes and two temperatures for H2."""
    import shutil

    shutil.copyfile(library / "H2.h5", library / "H3.h5")
    with h5py.File(library / "H3.h5", "a") as file:
        file["mass-number"][()] = 3
        file["atomic-weight-ratio"][()] = 2.990
        file["fissionable"][()] = False
    for name, temperature in [("H3-293.6K", 293.6), ("H2-600.0K", 600.0)]:
        shutil.copyfile(library / "H2-293.6K.h5", library / f"{name}.h5")
        with h5py.File(library / f"{name}.h5", "a") as file:
            file["temperature"][()] = temperature
            file["kT"][()] = temperature * 8.61733326e-5
    return library


def test_shared_element_identity_and_material_densities(hydrogen_library, monkeypatch):
    """Isotopes and temperatures share identity while materials retain their own recipes."""
    import mcdc.mcdc_get as mcdc_get
    from mcdc.main import prepare
    from mcdc import numba_types
    from mcdc.transport.physics.proton.condensed_interactions import (
        calculate_total_stopping_power,
    )

    reads = []
    open_file = h5py.File

    def count_element_reads(path, *args, **kwargs):
        if str(path) == str(hydrogen_library / "H.h5"):
            reads.append(path)
        return open_file(path, *args, **kwargs)

    monkeypatch.setattr(h5py, "File", count_element_reads)
    cold = mcdc.Material(
        nuclide_composition={"H2": 0.02, "H3": 0.03}, temperature=293.6
    )
    hot = mcdc.Material(nuclide_composition={"H2": 0.07}, temperature=600.0)
    model = mcdc.Simulation()
    model.set_model([mcdc.Cell(fill=cold), mcdc.Cell(fill=hot)])
    model.set_sources([mcdc.Source(energy=5.0)])

    # Recompilation must preserve sharing and read element identity once per snapshot.
    for expected_reads in (1, 2):
        model.compile()
        assert len(model.elements) == 1
        assert len(model.nuclides) == 3
        element = model.elements[0]
        assert element.atomic_number == 1
        assert all(nuclide.element is element for nuclide in model.nuclides)
        assert cold.elements == hot.elements == [element]
        np.testing.assert_allclose(cold.element_densities, [0.05])
        np.testing.assert_allclose(hot.element_densities, [0.07])
        assert len(reads) == expected_reads
        assert not model.electron_reactions
        assert not model.photon_reactions

    container, data = prepare(model)
    simulation = container[0]
    assert "atomic_number" not in simulation["nuclides"].dtype.names
    assert all(record["element_ID"] == element.ID for record in simulation["nuclides"])
    np.testing.assert_allclose(
        mcdc_get.material.element_densities_all(simulation["materials"][cold.ID], data),
        [0.05],
    )
    np.testing.assert_allclose(
        mcdc_get.material.element_densities_all(simulation["materials"][hot.ID], data),
        [0.07],
    )

    # Proton material-property evaluation also obtains Z through the packed element reference.
    particle = np.zeros(1, dtype=numba_types.particle)
    particle[0]["material_ID"], particle[0]["E"] = cold.ID, 1.0e6
    _, Z, _, _ = calculate_total_stopping_power(particle, simulation, data)
    assert Z == 1.0


def test_elemental_input_expands_to_nuclides_with_shared_element(
    hydrogen_library, monkeypatch, prepare_simulation
):
    from mcdc.object_.material import ISOTOPIC_ABUNDANCE

    monkeypatch.setitem(ISOTOPIC_ABUNDANCE, "H", {"H2": 1.0, "H3": 3.0})
    material = mcdc.Material(element_composition={"H": 0.08})
    container, _ = prepare_simulation(
        cells=[mcdc.Cell(fill=material)], sources=[mcdc.Source(energy=5.0)]
    )
    assert container[0]["N_element"] == 1
    assert material.elements[0].atomic_number == 1
    assert all(nuclide.element is material.elements[0] for nuclide in material.nuclides)
    densities = {
        nuclide.name: density
        for nuclide, density in material.nuclide_composition.items()
    }
    assert densities == pytest.approx({"H2": 0.02, "H3": 0.06})
    np.testing.assert_allclose(material.element_densities, [0.08])


def prepare_library(prepare_simulation):
    material = mcdc.Material(nuclide_composition={"H2": 0.02})
    return prepare_simulation(
        cells=[mcdc.Cell(fill=material)], sources=[mcdc.Source(energy=5.0)]
    )


def test_material_temperature_comes_from_available_files(library):
    import shutil

    shutil.copyfile(library / "H2-293.6K.h5", library / "H2-456.7K.h5")
    material = mcdc.Material(nuclide_composition={"H2": 0.02}, temperature=450.0)
    assert material.nuclides[0].temperature == 456.7


def test_split_identity_and_nonlinear_xs(library, prepare_simulation):
    with h5py.File(library / "H2-293.6K.h5", "a") as file:
        reactions = file["projectiles/neutron/reactions"]
        basic_reaction(
            reactions, "absorption", "capture", (1.0, 10.0, 100.0), laws=("log",)
        )
        basic_reaction(
            reactions, "threshold", "capture", (2.0, 4.0), 1, boundaries=(2,)
        )
    container, data = prepare_library(prepare_simulation)
    simulation = container[0]
    nuclide = simulation["nuclides"][0]
    assert simulation["elements"][nuclide["element_ID"]]["atomic_number"] == 1
    assert nuclide["mass_number"] == 2
    assert nuclide["atomic_weight_ratio"] == pytest.approx(1.9968)
    assert not nuclide["neutron_xs_is_linear"]
    for energy, expected in [
        (0.1, 1.0),
        (1.0, 1.0),
        (5.0, 5.0),
        (10.0, 12.0),
        (100.0, 104.0),
        (200.0, 104.0),
    ]:
        assert native.total_micro_xs(
            NEUTRON_REACTION_TOTAL, energy, nuclide, simulation, data
        ) == pytest.approx(expected)
    assert simulation["neutron_reactions"][0]["q_value"] == 2.0e6


@pytest.mark.parametrize("law", ["linear", "histogram"])
def test_shared_grid_jumps_and_endpoints(library, prepare_simulation, law):
    """Neutron tables use the same right-hand jump convention as photon tables."""
    grid = np.array([1.0, 10.0, 10.0, 100.0, 100.0])
    values = np.array([1.0, 3.0, 7.0, 11.0, 13.0])
    with h5py.File(library / "H2-293.6K.h5", "a") as file:
        neutron = file["projectiles/neutron"]
        del neutron["xs-energy-grid"]
        neutron["xs-energy-grid"] = grid
        basic_reaction(
            neutron["reactions"],
            "capture",
            "capture",
            values,
            laws=(law,),
            boundaries=(5,),
        )
    container, data = prepare_library(prepare_simulation)
    simulation = container[0]
    nuclide = simulation["nuclides"][0]
    assert bool(nuclide["neutron_xs_is_linear"]) == (law == "linear")
    for energy, expected in [
        (0.1, 1.0),
        (1.0, 1.0),
        (np.nextafter(10.0, -np.inf), 3.0 if law == "linear" else 1.0),
        (10.0, 7.0),
        (np.nextafter(100.0, -np.inf), 11.0 if law == "linear" else 7.0),
        (100.0, 13.0),
        (200.0, 13.0),
    ]:
        assert native.total_micro_xs(
            NEUTRON_REACTION_TOTAL, energy, nuclide, simulation, data
        ) == pytest.approx(expected)


def test_fission_production_is_reaction_local(library, prepare_simulation):
    with h5py.File(library / "H2-293.6K.h5", "a") as file:
        reactions = file["projectiles/neutron/reactions"]
        add_fission(reactions, "first", 2.0)
        add_fission(reactions, "second", 4.0)
    container, data = prepare_library(prepare_simulation)
    simulation = container[0]
    from mcdc import numba_types

    particle = np.zeros(1, dtype=numba_types.particle)
    particle[0]["material_ID"], particle[0]["E"] = 0, 5.0
    production = native.neutron_production_xs(
        NEUTRON_REACTION_FISSION, particle, simulation, data
    )
    assert production == pytest.approx(0.02 * (2.0 + 4.0))
    yields = [
        native.neutron_fission_prompt_multiplicity(5.0, reaction, simulation, data)
        for reaction in simulation["neutron_fission_reactions"]
    ]
    assert yields == [2.0, 4.0]


@pytest.mark.parametrize("prompt_yield", [0.0, 0.8])
def test_delayed_neutron_groups_spectra_and_directions(
    library, prepare_simulation, prompt_yield
):
    """Exercise actual fission emission, including branch probabilities and birth time."""
    from mcdc import numba_types

    with h5py.File(library / "H2-293.6K.h5", "a") as file:
        group = add_fission(
            file["projectiles/neutron/reactions"], "fission", prompt_yield
        )
        group["delayed-multiplicity/coefficients"][0] = 1.0 - prompt_yield
        delayed = group["delayed-neutrons"]
        del delayed["fractions"], delayed["decay-rates"]
        delayed["fractions"], delayed["decay-rates"] = [0.25, 0.75], [1.0, 2.0]
        for index, fraction in enumerate([0.2, 0.8]):
            spectrum = delayed.create_group(f"energy-distribution-{index}")
            spectrum.attrs["type"] = "level-scattering"
            spectrum["C1"], spectrum["C2"] = 0.0, fraction
    container, data = prepare_library(prepare_simulation)
    simulation = container[0]
    energies, directions, times = [], [], []
    for seed in range(1, 501):
        particle = np.zeros(1, dtype=numba_types.particle)
        particle[0]["E"], particle[0]["w"] = 10.0, 1.0
        particle[0]["ux"], particle[0]["alive"] = 1.0, True
        particle[0]["rng_seed"] = seed
        interaction = np.zeros(1, dtype=numba_types.interaction_data)
        native.sample_fission(
            simulation["neutron_reactions"][0],
            particle,
            interaction,
            simulation["nuclides"][0],
            simulation,
            data,
        )
        assert particle[0]["alive"]
        energies.append(particle[0]["E"])
        directions.append([particle[0][axis] for axis in ("ux", "uy", "uz")])
        times.append(particle[0]["t"])
    assert set(energies) == ({2.0, 5.0, 8.0} if prompt_yield else {2.0, 8.0})
    assert np.mean(np.array(energies) == 2.0) == pytest.approx(
        0.25 * (1.0 - prompt_yield), abs=0.06
    )
    np.testing.assert_allclose(np.linalg.norm(directions, axis=1), 1.0, atol=1.0e-12)
    np.testing.assert_allclose(np.mean(directions, axis=0), 0.0, atol=0.1)
    delayed_mask = np.array(energies) != 5.0
    assert np.all(np.array(times)[delayed_mask] > 0.0)
    assert np.all(np.array(times)[~delayed_mask] == 0.0)


def test_multiple_spectra_are_alternatives_for_each_neutron(
    library, prepare_simulation
):
    """Even when multiplicity equals the spectrum count, probabilities select spectra."""
    from mcdc import numba_types

    with h5py.File(library / "H2-293.6K.h5", "a") as file:
        group = basic_reaction(
            file["projectiles/neutron/reactions"], "n2n", "inelastic-scattering"
        )
        group["multiplicity"] = 2
        group["spectrum-probability-grid"] = [1.0, 100.0]
        group["spectrum-probability"] = [[1.0, 0.0]]
        group.create_group("angular-cosine-distribution").attrs["type"] = "isotropic"
        for index, fraction in enumerate([0.2, 0.8]):
            spectrum = group.create_group(f"energy-spectrum-{index}")
            spectrum.attrs["type"] = "level-scattering"
            spectrum["C1"], spectrum["C2"] = 0.0, fraction
    container, data = prepare_library(prepare_simulation)
    simulation = container[0]
    particle = np.zeros(1, dtype=numba_types.particle)
    particle[0]["E"], particle[0]["w"] = 10.0, 1.0
    particle[0]["ux"], particle[0]["rng_seed"] = 1.0, 1234567
    interaction = np.zeros(1, dtype=numba_types.interaction_data)
    native.sample_inelastic_scattering(
        simulation["neutron_reactions"][0],
        particle,
        interaction,
        simulation["nuclides"][0],
        simulation,
        data,
    )
    assert particle[0]["alive"]
    assert particle[0]["E"] == 2.0
    assert simulation["bank_active"]["size"] == 1
    assert simulation["bank_active"]["particle_data"][0]["E"] == 2.0


def test_capture_q_value_is_already_ev(library, prepare_simulation):
    from mcdc import numba_types

    with h5py.File(library / "H2-293.6K.h5", "a") as file:
        basic_reaction(file["projectiles/neutron/reactions"], "capture", "capture")
    container, data = prepare_library(prepare_simulation)
    simulation = container[0]
    particle = np.zeros(1, dtype=numba_types.particle)
    particle[0]["E"], particle[0]["w"] = 5.0, 2.0
    interaction = np.zeros(1, dtype=numba_types.interaction_data)
    native.capture(
        simulation["neutron_reactions"][0],
        particle,
        interaction,
        simulation["nuclides"][0],
        simulation,
        data,
    )
    assert interaction[0]["energy_deposition"] == 2.0 * (5.0 + 2.0e6)


@pytest.mark.parametrize("category", ["inelastic-scattering", "fission"])
def test_nbody_reaction_energy_scale_and_com_transformation(
    library, prepare_simulation, category
):
    """Reaction context sets LAW 66's energy scale before the COM-to-LAB transform."""
    from mcdc import numba_types

    with h5py.File(library / "H2-293.6K.h5", "a") as file:
        reactions = file["projectiles/neutron/reactions"]
        if category == "fission":
            group = add_fission(reactions, "nbody", 1.0)
            del group["prompt-angular-cosine-distribution"]
            del group["prompt-energy-distribution"]
            spectrum = group.create_group("prompt-energy-distribution")
        else:
            group = basic_reaction(reactions, "nbody", category)
            group["multiplicity"] = 1
            group["spectrum-probability-grid"] = [1.0, 100.0]
            group["spectrum-probability"] = [[1.0]]
            spectrum = group.create_group("energy-spectrum-0")
        group["reference-frame"][()] = "COM"
        group["Q-value"][()] = -2.0
        spectrum.attrs["type"] = "n-body"
        spectrum["number-particles"], spectrum["total-mass-ratio"] = 3, 3.0
        spectrum["value"], spectrum["pdf"] = [0.0, 1.0], [1.0, 1.0]
        interpolation(spectrum, boundaries=(2,))
    container, data = prepare_library(prepare_simulation)
    simulation = container[0]
    nuclide = simulation["nuclides"][0]
    A = nuclide["atomic_weight_ratio"]
    fractions, cosines = [], []
    for energy in (10.0, 20.0):
        particle = np.zeros(1, dtype=numba_types.particle)
        particle[0]["E"], particle[0]["w"] = energy, 1.0
        particle[0]["ux"], particle[0]["rng_seed"] = 1.0, 1234567
        interaction = np.zeros(1, dtype=numba_types.interaction_data)
        sampler = (
            native.sample_fission
            if category == "fission"
            else native.sample_inelastic_scattering
        )
        sampler(
            simulation["neutron_reactions"][0],
            particle,
            interaction,
            nuclide,
            simulation,
            data,
        )
        outgoing = particle[0]["E"]
        # Recover the COM velocity by subtracting the incident COM translation.
        translation = np.sqrt(energy) / (A + 1)
        forward_com = np.sqrt(outgoing) * particle[0]["ux"] - translation
        energy_com = (
            outgoing
            + translation**2
            - 2 * np.sqrt(outgoing) * particle[0]["ux"] * translation
        )
        maximum = (2.0 / 3.0) * (A / (A + 1) * energy - 2.0)
        assert 0 < energy_com < maximum
        fractions.append(energy_com / maximum)
        cosines.append(forward_com / np.sqrt(energy_com))
        assert outgoing + interaction[0]["energy_deposition"] == pytest.approx(
            energy - 2.0
        )
    # Identical random streams sample identical normalized energy and COM angle.
    assert fractions[0] == pytest.approx(fractions[1])
    assert cosines[0] == pytest.approx(cosines[1])


@pytest.mark.parametrize("isotropic", [True, False])
def test_elastic_angular_representations(library, prepare_simulation, isotropic):
    from mcdc import numba_types

    with h5py.File(library / "H2-293.6K.h5", "a") as file:
        group = basic_reaction(
            file["projectiles/neutron/reactions"], "elastic", "elastic-scattering"
        )
        group["reference-frame"][()] = "COM"
        group["Q-value"][()] = 0.0
        angle = group.create_group("angular-cosine-distribution")
        angle.attrs["type"] = "isotropic" if isotropic else "table"
        if not isotropic:
            angle["value"], angle["pdf"] = [-1.0, 1.0], [0.5, 0.5]
            interpolation(angle, boundaries=(2,))
    container, data = prepare_library(prepare_simulation)
    simulation = container[0]
    particle = np.zeros(1, dtype=numba_types.particle)
    particle[0]["E"], particle[0]["w"] = 10.0, 1.0
    particle[0]["ux"], particle[0]["rng_seed"] = 1.0, 1234567
    interaction = np.zeros(1, dtype=numba_types.interaction_data)
    native.sample_elastic_scattering(
        simulation["neutron_reactions"][0],
        particle,
        interaction,
        simulation["nuclides"][0],
        simulation,
        data,
    )
    assert 0.0 < particle[0]["E"] <= 10.0
    direction = [particle[0][axis] for axis in ("ux", "uy", "uz")]
    assert np.linalg.norm(direction) == pytest.approx(1.0)
    assert interaction[0]["energy_deposition"] == pytest.approx(10.0 - particle[0]["E"])


def test_target_thermal_velocity_uses_temperature():
    from mcdc import numba_types
    from mcdc.constant import NEUTRON_MASS, LIGHT_SPEED, BOLTZMANN_K

    # At zero incident speed the collision-weighted Maxwellian has <(beta V)^2> = 2.
    cold = np.zeros(1, dtype=numba_types.particle)
    hot = np.zeros(1, dtype=numba_types.particle)
    for particle in (cold, hot):
        particle[0]["ux"], particle[0]["rng_seed"] = 1.0, 1234567
    squares = []
    beta_squared = 2.0 * NEUTRON_MASS / (2.0 * LIGHT_SPEED**2 * BOLTZMANN_K * 0.1)
    for _ in range(1000):
        v_cold = np.array(native.sample_nucleus_velocity(2.0, 0.1, cold))
        v_hot = np.array(native.sample_nucleus_velocity(2.0, 1200.0, hot))
        np.testing.assert_allclose(v_hot, v_cold * np.sqrt(12000.0), rtol=1.0e-12)
        squares.append(np.dot(v_cold, v_cold) * beta_squared)
    assert np.mean(squares) == pytest.approx(2.0, abs=0.15)


@pytest.mark.parametrize("kind", ["constant", "polynomial", "table"])
def test_data_representations_keep_ev_units(tmp_path, kind, prepare_simulation):
    with h5py.File(tmp_path / "data.h5", "w") as group:
        group.attrs["type"] = kind
        if kind == "constant":
            group.attrs["type"] = "polynomial"
            group["coefficients"] = [3.0]
            expected = 3.0
        elif kind == "polynomial":
            group["coefficients"] = [2.0, 0.5]
            expected = 4.5
        else:
            group["grid"], group["value"] = [1.0, 10.0, 100.0], [2.0, 4.0, 8.0]
            interpolation(group, ("histogram", "log"), (2, 3))
            expected = 2.0
        result = read_data(group)
    container, data = prepare_simulation(objects=[result])
    simulation = container[0]
    assert evaluate_data_clamped(
        5.0, simulation["data"][result.ID], simulation, data
    ) == pytest.approx(expected)


def test_numeric_indices_and_format_validation(tmp_path):
    with h5py.File(tmp_path / "data.h5", "w") as file:
        for index in range(12):
            file.create_group(f"energy-spectrum-{index}")
        assert [group.name for group in numbered_groups(file, "energy-spectrum-")] == [
            f"/energy-spectrum-{i}" for i in range(12)
        ]
        with pytest.raises(ValueError, match="expected mcdc-data"):
            check_format(file)
        file.attrs["format"], file.attrs["format-version"] = "mcdc-data", "0.1.0"
        check_format(file)
        file.attrs["format-version"] = "9.0.0"
        with pytest.raises(ValueError, match="9.0.0"):
            check_format(file)
