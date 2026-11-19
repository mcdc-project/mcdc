import pytest
import numpy as np

import mcdc

from mcdc.object_.settings import Settings


@pytest.mark.parametrize("chunks", [1, 8])
def test_tally_reduction_chunks_are_packed(prepare_simulation, chunks):
    def configure(simulation):
        simulation.settings.tally_reduction_chunks = chunks

    container, _ = prepare_simulation(configure=configure)
    assert container[0]["settings"]["tally_reduction_chunks"] == chunks


@pytest.mark.parametrize("chunks", [0, -1, 1.5, 2.0, True, np.bool_(True), "8"])
def test_reject_invalid_tally_reduction_chunks(chunks, capsys):
    simulation = mcdc.Simulation()
    simulation.set_model([mcdc.Cell()])
    with pytest.raises(SystemExit):
        simulation.settings.tally_reduction_chunks = chunks
        simulation.compile()
    assert "tally_reduction_chunks must be" in capsys.readouterr().out


def test_transport_settings_are_independent():
    first, second = Settings(), Settings()
    assert not first.neutron_transport.active
    assert not first.electron_transport.active
    assert first.electron_transport.prioritize_low_energy
    first.electron_transport.prioritize_low_energy = False
    assert not first.neutron_transport.prioritize_low_energy
    assert second.electron_transport.prioritize_low_energy


def test_transport_settings_are_packed(prepare_simulation):
    def configure(simulation):
        simulation.settings.electron_transport.active = True

    container, _ = prepare_simulation(configure=configure)
    settings = container[0]["settings"]
    assert not settings["neutron_transport"]["active"]
    assert not settings["neutron_transport"]["prioritize_low_energy"]
    assert settings["electron_transport"]["active"]
    assert settings["electron_transport"]["prioritize_low_energy"]


@pytest.mark.parametrize(
    "species,explicit_neutron,explicit_electron,expected_neutron,expected_electron",
    [
        ([], False, False, False, False),
        (["neutron"], False, False, True, False),
        (["electron"], False, False, False, True),
        (["neutron", "electron"], False, False, True, True),
        (["electron"], True, False, True, True),
        (["neutron"], False, True, True, True),
    ],
)
def test_finalization_activates_source_species(
    prepare_simulation,
    species,
    explicit_neutron,
    explicit_electron,
    expected_neutron,
    expected_electron,
):
    def configure(simulation):
        simulation.settings.neutron_transport.active = explicit_neutron
        simulation.settings.electron_transport.active = explicit_electron

    sources = [mcdc.Source(particle_type=particle) for particle in species]
    container, _ = prepare_simulation(sources=sources, configure=configure)
    settings = container[0]["settings"]
    assert settings["neutron_transport"]["active"] == expected_neutron
    assert settings["electron_transport"]["active"] == expected_electron


@pytest.mark.parametrize("active", [False, True])
def test_reject_neutron_energy_priority(active, capsys):
    simulation = mcdc.Simulation()
    simulation.set_model([mcdc.Cell()])
    simulation.settings.neutron_transport.active = active
    simulation.settings.neutron_transport.prioritize_low_energy = True

    with pytest.raises(SystemExit):
        simulation.compile()

    assert (
        "prioritize_low_energy is currently supported only for electron transport."
        in capsys.readouterr().out
    )
