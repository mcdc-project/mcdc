import h5py
import numpy as np
import pytest

import mcdc
import mcdc.numba_types as type_
from mcdc.constant import PARTICLE_PROTON
import mcdc.transport.physics.proton.native as proton_native
from mcdc.object_.proton_reaction import ProtonReactionInelasticScattering
from mcdc.object_.distribution import DistributionMixture
from mcdc.transport.data import evaluate_data
from mcdc.transport.distribution import sample_distribution_with_scale


def _write_tabulated_distribution(group, *, energy, value, pdf):
    group.attrs["type"] = "tabulated"
    group["energy"] = np.asarray(energy, dtype=np.float64)
    group["offset"] = np.asarray([0] * len(energy), dtype=np.int64)
    group["value"] = np.asarray(value, dtype=np.float64)
    group["pdf"] = np.asarray(pdf, dtype=np.float64)


def _write_angular_distribution(group):
    group.attrs["type"] = "tabulated"
    group["incident_energies"] = np.array([1.0], dtype=np.float64)
    subgroup = group.create_group("E_in_1")
    subgroup.attrs["type"] = "tabulated"
    subgroup["cosines"] = np.array([-1.0, 1.0], dtype=np.float64)
    subgroup["pdf"] = np.array([0.5, 0.5], dtype=np.float64)


def _write_proton_library(path):
    with h5py.File(path / "H.h5", "w") as file:
        file["atomic_number"] = np.int64(1)
        file["atomic_weight_ratio"] = np.float64(1.0)

    def write_nuclide_library(nuclide_name, mass):
        with h5py.File(path / f"{nuclide_name}-293.6K.h5", "w") as file:
            file["atomic_number"] = np.int64(1)
            file["mass_number"] = np.int64(mass)
            file["atomic_weight_ratio"] = np.float64(1.0)
            file["fissionable"] = False
            file["excitation_level"] = np.int64(0)
            file["radiation_length"] = np.float64(1.0)

            neutron_reactions = file.create_group("neutron_reactions")
            neutron_reactions["xs_energy_grid"] = np.array(
                [0.1, 1.0, 10.0], dtype=np.float64
            )
            neutron_reactions.create_group("elastic_scattering")
            neutron_reactions.create_group("capture")
            neutron_reactions.create_group("inelastic_scattering")

            reactions = file.create_group("proton_reactions")
            reactions["xs_energy_grid"] = np.array([0.1, 1.0, 10.0], dtype=np.float64)

            def add_reaction(group_name, mt, xs, q_value):
                group = reactions.create_group(group_name)
                reaction = group.create_group(f"MT-{mt}")
                reaction.attrs["MT"] = np.int64(mt)
                reaction["xs"] = np.asarray(xs, dtype=np.float64)
                reaction["xs"].attrs["offset"] = 0
                reaction["reference_frame"] = np.bytes_("LAB")
                reaction["Q-value"] = np.float64(q_value)
                _write_angular_distribution(
                    reaction.create_group("angular_cosine_distribution")
                )
                return reaction

            add_reaction("elastic_scattering", 2, [1.0, 1.0, 1.0], 0.0)
            add_reaction("capture", 103, [0.5, 0.5, 0.5], 1.0)
            inelastic = add_reaction("inelastic_scattering", 4, [0.5, 0.5, 0.5], 0.5)
            inelastic["multiplicity"] = np.int64(1)
            inelastic["spectrum_probability_grid"] = np.array(
                [0.0, 1.0], dtype=np.float64
            )
            inelastic["spectrum_probability"] = np.array([[1.0]], dtype=np.float64)
            _write_tabulated_distribution(
                inelastic.create_group("energy_spectrum-1"),
                energy=[1.0],
                value=[0.0, 1.0],
                pdf=[1.0, 1.0],
            )

            secondary_group = inelastic.create_group("secondary_products")
            neutron = secondary_group.create_group("neutron")
            neutron.attrs["ZAP"] = np.int64(1)
            neutron.attrs["multiplicity"] = np.int64(1)
            neutron.attrs["reference_frame"] = np.bytes_("LAB")
            _write_tabulated_distribution(
                neutron.create_group("energy_spectrum"),
                energy=[1.0],
                value=[0.0, 1.0],
                pdf=[1.0, 1.0],
            )
            _write_angular_distribution(
                neutron.create_group("angular_cosine_distribution")
            )

    for nuclide_name, mass in [("H1", 1), ("H2", 2)]:
        write_nuclide_library(nuclide_name, mass)


@pytest.fixture
def inelastic_model(tmp_path, monkeypatch, prepare_simulation, neutron_library):
    _write_proton_library(tmp_path)
    monkeypatch.setenv("MCDC_LIB", str(tmp_path))
    material = mcdc.Material(number_density={"H": 1.0})
    cell = mcdc.Cell(fill=material)

    def configure(simulation):
        simulation.settings.proton_transport.active = True
        simulation.settings.neutron_transport.active = True

    simulation_container, data = prepare_simulation(cells=[cell], configure=configure)
    return simulation_container[0], data


def _make_particle(E, *, w=1.0):
    particles = np.zeros(1, dtype=type_.particle)
    particle = particles[0]
    particle["cell_ID"] = 0
    particle["material_ID"] = 0
    particle["surface_ID"] = -1
    particle["alive"] = True
    particle["fresh"] = True
    particle["event"] = 0
    particle["x"] = 0.0
    particle["y"] = 0.0
    particle["z"] = 0.0
    particle["t"] = 0.0
    particle["ux"] = 0.0
    particle["uy"] = 0.0
    particle["uz"] = 1.0
    particle["E"] = E
    particle["w"] = w
    particle["particle_type"] = PARTICLE_PROTON
    particle["rng_seed"] = np.uint64(1)
    return particles


def _make_interaction_data(particle):
    interaction = np.zeros(1, dtype=type_.interaction_data)
    incident = np.zeros(1, dtype=type_.particle_data)[0]
    for name in ("x", "y", "z", "t", "ux", "uy", "uz", "E", "w", "particle_type"):
        incident[name] = particle[name]
    interaction[0]["incident_particle"] = incident
    return interaction


def test_inelastic_scattering_conserves_energy(inelastic_model):
    simulation, data = inelastic_model
    particle_container = _make_particle(3.0e6)
    interaction = _make_interaction_data(particle_container[0])
    simulation["bank_active"]["size"][0] = 0

    reaction = simulation["proton_reactions"][1]
    proton_native.inelastic_scattering(
        reaction,
        particle_container,
        interaction,
        simulation["nuclides"][0],
        simulation,
        data,
    )

    bank = simulation["bank_active"]
    bank_energy = sum(
        bank["particle_data"][i]["E"] * bank["particle_data"][i]["w"]
        for i in range(bank["size"][0])
    )

    assert particle_container[0]["E"] >= 0.0
    assert (
        interaction[0]["energy_deposition"]
        + particle_container[0]["E"] * particle_container[0]["w"]
        + bank_energy
    ) == pytest.approx(3.0e6 + reaction["q_value"] * 1.0e6, rel=1e-12)


def test_energy_dependent_proton_yield(tmp_path, prepare_simulation):
    _write_proton_library(tmp_path)
    with h5py.File(tmp_path / "H1-293.6K.h5", "a") as file:
        group = file["proton_reactions/inelastic_scattering/MT-4"]
        del group["multiplicity"]
        yield_group = group.create_group("proton_yield")
        yield_group["energy"], yield_group["value"] = [1.0, 5.0], [0.0, 2.0]
        reaction = ProtonReactionInelasticScattering.from_h5_group(group, None)
    container, data = prepare_simulation(objects=[reaction])
    simulation = container[0]
    packed = simulation["proton_inelastic_scattering_reactions"][reaction.sub_ID]
    mean = evaluate_data(
        2e6, simulation["data"][packed["multiplicity_data_ID"]], simulation, data
    )
    assert mean == 0.5
    assert "use_all_spectra" not in packed.dtype.names
    assert "spectrum_ID" in packed.dtype.names


def _set_alternative_spectra(group, grid, probabilities):
    """Two deterministic energy laws make component selection observable."""
    for name in (
        "spectrum_probability_grid",
        "spectrum_probability",
        "energy_spectrum-1",
    ):
        del group[name]
    group["spectrum_probability_grid"] = grid
    group["spectrum_probability"] = probabilities
    for index, coefficient in enumerate((0.1, 0.2), start=1):
        spectrum = group.create_group(f"energy_spectrum-{index}")
        spectrum.attrs["type"] = "level-scattering"
        spectrum["C1"], spectrum["C2"] = 0.0, coefficient


def test_multiplicity_does_not_assign_one_particle_per_spectrum(
    tmp_path, monkeypatch, prepare_simulation, neutron_library
):
    _write_proton_library(tmp_path)
    with h5py.File(tmp_path / "H1-293.6K.h5", "a") as file:
        group = file["proton_reactions/inelastic_scattering/MT-4"]
        group["multiplicity"][()] = 2
        _set_alternative_spectra(group, [0.0, 10.0], [[0.0, 1.0]])
    monkeypatch.setenv("MCDC_LIB", str(tmp_path))
    material = mcdc.Material(number_density={"H1": 1.0})

    def configure(simulation):
        simulation.settings.proton_transport.active = True
        simulation.settings.neutron_transport.active = True

    container, data = prepare_simulation(
        cells=[mcdc.Cell(fill=material)], configure=configure
    )
    simulation = container[0]
    particle = _make_particle(3e6)
    interaction = _make_interaction_data(particle[0])
    simulation["bank_active"]["size"][0] = 0
    proton_native.inelastic_scattering(
        simulation["proton_reactions"][1],
        particle,
        interaction,
        simulation["nuclides"][0],
        simulation,
        data,
    )
    bank = simulation["bank_active"]
    energies = [particle[0]["E"]]
    energies.extend(
        product["E"]
        for product in bank["particle_data"][: bank["size"][0]]
        if product["particle_type"] == PARTICLE_PROTON
    )
    assert particle[0]["alive"]
    np.testing.assert_allclose(energies, [0.6e6, 0.6e6])


@pytest.mark.parametrize(
    "energy,expected",
    [(0.5e6, 0.1e6), (1.5e6, 0.3e6), (2e6, 0.4e6), (2.5e6, 0.25e6), (4e6, 0.4e6)],
)
def test_proton_mixture_uses_probability_bins_and_endpoint_values(
    tmp_path, prepare_simulation, energy, expected
):
    _write_proton_library(tmp_path)
    with h5py.File(tmp_path / "H1-293.6K.h5", "a") as file:
        group = file["proton_reactions/inelastic_scattering/MT-4"]
        _set_alternative_spectra(group, [1.0, 2.0, 3.0], [[0.0, 1.0], [1.0, 0.0]])
        reaction = ProtonReactionInelasticScattering.from_h5_group(group, None)
    assert isinstance(reaction.spectrum, DistributionMixture)
    container, data = prepare_simulation(objects=[reaction])
    simulation = container[0]
    packed = simulation["proton_inelastic_scattering_reactions"][reaction.sub_ID]
    sampled = sample_distribution_with_scale(
        energy,
        simulation["distributions"][packed["spectrum_ID"]],
        _make_particle(energy),
        simulation,
        data,
    )
    assert sampled == pytest.approx(expected)


def test_proton_spectrum_columns_follow_numeric_suffixes(tmp_path, prepare_simulation):
    _write_proton_library(tmp_path)
    with h5py.File(tmp_path / "H1-293.6K.h5", "a") as file:
        group = file["proton_reactions/inelastic_scattering/MT-4"]
        del group["energy_spectrum-1"]
        del group["spectrum_probability"]
        probabilities = np.zeros((1, 11))
        probabilities[0, 9] = 1.0
        group["spectrum_probability"] = probabilities
        for index in range(1, 12):
            spectrum = group.create_group(f"energy_spectrum-{index}")
            spectrum.attrs["type"] = "level-scattering"
            spectrum["C1"], spectrum["C2"] = 0.0, 0.01 * index
        reaction = ProtonReactionInelasticScattering.from_h5_group(group, None)
    container, data = prepare_simulation(objects=[reaction])
    simulation = container[0]
    result = sample_distribution_with_scale(
        1e6,
        simulation["distributions"][reaction.spectrum.ID],
        _make_particle(1e6),
        simulation,
        data,
    )
    assert result == pytest.approx(0.1e6)
