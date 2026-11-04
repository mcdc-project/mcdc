import h5py
import numpy as np
import pytest

import mcdc
import mcdc.numba_types as type_
from mcdc.constant import PARTICLE_PROTON
import mcdc.transport.physics.proton.native as proton_native


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

            elastic = add_reaction("elastic_scattering", 2, [1.0, 1.0, 1.0], 0.0)
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
            elastic["mu_table_ID"] = np.int64(0)

    for nuclide_name, mass in [("H1", 1), ("H2", 2)]:
        write_nuclide_library(nuclide_name, mass)


@pytest.fixture
def elastic_model(tmp_path, monkeypatch, prepare_simulation, neutron_library):
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


def test_elastic_scattering_conserves_energy(elastic_model):
    simulation, data = elastic_model
    particle_container = _make_particle(1.5e6)
    interaction = _make_interaction_data(particle_container[0])

    reaction = simulation["proton_reactions"][0]
    proton_native.elastic_scattering(
        reaction,
        particle_container,
        interaction,
        simulation["nuclides"][0],
        simulation,
        data,
    )

    assert interaction[0]["energy_deposition"] >= 0.0
    assert (
        interaction[0]["energy_deposition"]
        + particle_container[0]["E"] * particle_container[0]["w"]
    ) == pytest.approx(1.5e6, rel=1e-12)
