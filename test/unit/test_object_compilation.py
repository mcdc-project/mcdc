import numpy as np
import pytest

import mcdc

from mcdc.object_.base import MCDCBase, MCDCObject
from mcdc.object_.data import DataPolynomial
from mcdc.object_.transport_model_data import NeutronMultigroupData
from mcdc.object_.technique import Technique
from mcdc.object_.nuclide import Nuclide
from mcdc.object_.universe import Universe


class ObjectOwner(Universe):
    child: DataPolynomial
    children: list[DataPolynomial]
    ignored: DataPolynomial

    non_numba = ["ignored"]

    def __init__(self, child, children, ignored):
        super().__init__()
        self.child = child
        self.children = children
        self.ignored = ignored


class EmbeddedConfiguration(MCDCBase):
    label = "embedded_configuration"

    def __init__(self):
        self.member = None


def test_simulation_reserves_zero_group_mg_as_id_zero():
    simulation = mcdc.Simulation()
    simulation.set_model([mcdc.Cell()])

    assert isinstance(simulation.technique, Technique)

    simulation.compile()

    assert len(simulation.neutron_multigroup_data) == 1
    assert isinstance(simulation.neutron_multigroup_data[0], NeutronMultigroupData)
    assert simulation.neutron_multigroup_data[0].ID == 0
    assert simulation.neutron_multigroup_data[0].G == 0
    assert simulation.neutron_multigroup_data[0].compile_ID == simulation.compile_ID


def test_simulation_derives_hybrid_for_local_multigroup_grids():
    material_a = mcdc.Material.multigroup(capture=[0.1], energy_grid=[1.0, 2.0])
    material_b = mcdc.Material.multigroup(capture=[0.2], energy_grid=[2.0, 3.0])
    simulation = mcdc.Simulation()
    simulation.set_model([mcdc.Cell(fill=material_a), mcdc.Cell(fill=material_b)])

    simulation.compile()

    assert simulation.technique.neutron_multigroup.hybrid


@pytest.mark.parametrize("explicit_grid", [False, True])
def test_simulation_derives_standard_multigroup_for_shared_grid(
    explicit_grid, prepare_simulation
):
    kwargs = {"energy_grid": [1.0, 2.0]} if explicit_grid else {}
    material_a = mcdc.Material.multigroup(capture=[0.1], **kwargs)
    material_b = mcdc.Material.multigroup(capture=[0.2], **kwargs)

    simulation_container, _ = prepare_simulation(
        cells=[mcdc.Cell(fill=material_a), mcdc.Cell(fill=material_b)]
    )

    simulation = simulation_container[0]
    assert not simulation["technique"]["neutron_multigroup"]["hybrid"]


def test_hybrid_multigroup_requires_explicit_energy_grids(capsys):
    material_default = mcdc.Material.multigroup(capture=[0.1])
    material_explicit = mcdc.Material.multigroup(capture=[0.2], energy_grid=[2.0, 3.0])
    simulation = mcdc.Simulation()
    simulation.set_model(
        [mcdc.Cell(fill=material_default), mcdc.Cell(fill=material_explicit)]
    )

    with pytest.raises(SystemExit):
        simulation.compile()

    assert "requires an explicit energy_grid" in capsys.readouterr().out


@pytest.mark.parametrize(
    "source",
    [
        mcdc.Source(energy=1),
        mcdc.Source(energy=1.0),
        mcdc.Source(discrete_energy=([0, 1], [0.25, 0.75])),
    ],
)
def test_standard_multigroup_accepts_integer_group_coordinates(source):
    material = mcdc.Material.multigroup(capture=[0.1, 0.2])
    simulation = mcdc.Simulation()
    simulation.set_model([mcdc.Cell(fill=material)])
    simulation.set_sources([source])

    simulation.compile()

    assert not simulation.technique.neutron_multigroup.hybrid


@pytest.mark.parametrize(
    "source, expected_message",
    [
        (
            mcdc.Source(energy=0.5),
            "must be finite, integer-valued group coordinates",
        ),
        (
            mcdc.Source(discrete_energy=([0, 1.5], [0.25, 0.75])),
            "must be finite, integer-valued group coordinates",
        ),
        (
            mcdc.Source(energy=np.nan),
            "must be finite, integer-valued group coordinates",
        ),
        (
            mcdc.Source(energy=-1),
            "must satisfy 0 <= energy < G",
        ),
        (
            mcdc.Source(energy=2),
            "must satisfy 0 <= energy < G",
        ),
        (
            mcdc.Source(energy=([0.0, 1.0], [1.0, 1.0])),
            "requires neutron sources to use a scalar energy or discrete_energy",
        ),
    ],
)
def test_standard_multigroup_rejects_invalid_source_energy(
    source,
    expected_message,
    capsys,
):
    material = mcdc.Material.multigroup(capture=[0.1, 0.2])
    simulation = mcdc.Simulation()
    simulation.set_model([mcdc.Cell(fill=material)])
    simulation.set_sources([source])

    with pytest.raises(SystemExit):
        simulation.compile()

    assert expected_message in capsys.readouterr().out


def test_standard_neutron_multigroup_does_not_validate_electron_source_energy():
    material = mcdc.Material.multigroup(capture=[0.1, 0.2])
    source = mcdc.Source(
        particle_type="electron",
        energy=([1.0e3, 2.0e3], [1.0, 1.0]),
    )
    simulation = mcdc.Simulation()
    simulation.set_model([mcdc.Cell(fill=material)])
    simulation.set_sources([source])

    simulation.compile()

    assert not simulation.technique.neutron_multigroup.hybrid


def test_native_only_simulation_remains_hybrid(monkeypatch):
    # Isolate mode finalization from native data-library loading.
    def compile_nuclide(nuclide, simulation):
        nuclide.fissionable = False
        return MCDCObject._compile_into_simulation(nuclide, simulation)

    monkeypatch.setattr(Nuclide, "_compile_into_simulation", compile_nuclide)
    monkeypatch.setattr(Nuclide, "set_neutron_data", lambda self, simulation: None)
    material = mcdc.Material(nuclide_composition={"H1": 0.1})
    simulation = mcdc.Simulation()
    simulation.set_model([mcdc.Cell(fill=material)])

    simulation.compile()

    assert simulation.technique.neutron_multigroup.hybrid


def test_local_multigroup_grids_pack_hybrid(prepare_simulation):
    material_a = mcdc.Material.multigroup(capture=[0.1], energy_grid=[1.0, 2.0])
    material_b = mcdc.Material.multigroup(capture=[0.2], energy_grid=[2.0, 3.0])

    simulation_container, _ = prepare_simulation(
        cells=[mcdc.Cell(fill=material_a), mcdc.Cell(fill=material_b)]
    )

    simulation = simulation_container[0]
    assert simulation["technique"]["neutron_multigroup"]["hybrid"]


def test_mcdc_object_compiles_object_members_and_lists():
    child = DataPolynomial(np.array([1.0]))
    children = [
        DataPolynomial(np.array([2.0])),
        DataPolynomial(np.array([3.0])),
    ]
    ignored = DataPolynomial(np.array([4.0]))
    simulation = mcdc.Simulation()
    simulation.root_universe = ObjectOwner(child, children, ignored)
    simulation.root_universe.cells = [mcdc.Cell()]

    simulation.compile()

    assert simulation.data[1:] == [child, *children]
    assert ignored.compile_ID == 0


def test_simulation_compiles_objects_owned_by_embedded_configuration():
    mesh = mcdc.MeshUniform()
    weight_windows = np.ones((1, 1, 1, 1, 3))
    simulation = mcdc.Simulation()
    simulation.set_model([mcdc.Cell()])
    simulation.technique.weight_windows(weight_windows, mesh=mesh)

    simulation.compile()

    assert simulation.meshes == [mesh]
    assert simulation.technique.compile_ID == simulation.compile_ID
    assert simulation.technique.weight_windows.compile_ID == simulation.compile_ID


def test_embedded_compile_id_prevents_cycles_and_supports_recompilation():
    configuration_a = EmbeddedConfiguration()
    configuration_b = EmbeddedConfiguration()
    configuration_a.member = configuration_b
    configuration_b.member = configuration_a
    simulation = mcdc.Simulation()
    simulation.set_model([mcdc.Cell()])
    simulation.configuration = configuration_a

    simulation.compile()
    first_compile_ID = simulation.compile_ID

    assert configuration_a.compile_ID == first_compile_ID
    assert configuration_b.compile_ID == first_compile_ID

    simulation.compile()

    assert simulation.compile_ID != first_compile_ID
    assert configuration_a.compile_ID == simulation.compile_ID
    assert configuration_b.compile_ID == simulation.compile_ID


def test_material_canonicalizes_composition_before_member_compilation(monkeypatch):
    # Replace library loading with the minimal state needed for this unit test
    def compile_nuclide(nuclide, simulation):
        nuclide.fissionable = nuclide.name == "U235"
        return MCDCObject._compile_into_simulation(nuclide, simulation)

    monkeypatch.setattr(Nuclide, "_compile_into_simulation", compile_nuclide)

    material_a = mcdc.Material(nuclide_composition={"U235": 1.0})
    material_b = mcdc.Material(nuclide_composition={"U235": 2.0})
    simulation = mcdc.Simulation()
    simulation.set_model([mcdc.Cell()])
    simulation.compile()

    material_a._compile_into_simulation(simulation)
    material_b._compile_into_simulation(simulation)

    assert len(simulation.nuclides) == 1
    assert material_a.nuclides == simulation.nuclides
    assert material_b.nuclides == simulation.nuclides
    assert material_a.fissionable
    assert material_b.fissionable


def test_simulation_compilation_finalizes_model_wide_state():
    source_a = mcdc.Source(probability=1.0)
    source_b = mcdc.Source(probability=3.0)
    simulation = mcdc.Simulation()
    simulation.set_model([mcdc.Cell()])
    simulation.set_sources([source_a, source_b])
    simulation.settings.neutron_eigenvalue_mode = True
    simulation.settings.N_inactive = 2
    simulation.settings.N_cycle = 4
    simulation.settings.k_init = 1.25

    simulation.compile()

    assert np.allclose([source_a.probability, source_b.probability], [0.25, 0.75])
    assert simulation.k_eff == 1.25
    assert not simulation.cycle_active
    assert simulation.k_cycle.shape == (4,)
    assert simulation.gyration_radius.shape == (4,)


def test_simulation_compilation_sets_particle_bank_capacities():
    simulation = mcdc.Simulation()
    simulation.set_model([mcdc.Cell()])
    simulation.settings.N_particle = 100
    simulation.settings.N_census = 2
    simulation.settings.N_batch = 2
    simulation.settings.active_bank_buffer = 11
    simulation.settings.census_bank_buffer_ratio = 2.0
    simulation.settings.source_bank_buffer_ratio = 3.0
    simulation.settings.future_bank_buffer_ratio = 4.0

    simulation.compile()

    N_work = int(np.ceil(100 / simulation.mpi_size))
    assert simulation.bank_active.size[0] == 11
    assert simulation.bank_census.size[0] == 2 * N_work
    assert simulation.bank_source.size[0] == 3 * N_work
    assert simulation.bank_future.size[0] == 4 * N_work


def test_simulation_rejects_an_empty_root_universe(capsys):
    simulation = mcdc.Simulation()

    with pytest.raises(SystemExit):
        simulation.compile()

    assert "root universe is empty" in capsys.readouterr().out
