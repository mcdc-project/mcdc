import pytest
from types import SimpleNamespace

import mcdc
from mcdc.constant import FILL_MATERIAL
from mcdc.object_.base import MCDCObject, MCDCPolymorphic
from mcdc.object_.material import Material, set_elements_from_nuclides
from mcdc.object_.nuclide import Nuclide
from mcdc.object_.element import Element
from mcdc.object_.util import element_symbol_from_nuclide_name


def test_material_is_the_only_public_non_polymorphic_material_type():
    assert mcdc.Material is Material
    assert issubclass(Material, MCDCObject)
    assert not issubclass(Material, MCDCPolymorphic)
    assert not hasattr(mcdc, "MaterialMG")


def test_multigroup_factory_returns_a_material_and_forwards_transport_data():
    material = Material.multigroup(
        name="Delayed fuel",
        fission=[0.2, 0.3],
        nu_d=[[0.1, 0.2], [0.3, 0.4]],
        chi_d=[[1.0, 3.0], [3.0, 1.0]],
        decay_rate=[0.01, 0.02],
        energy_grid=[1.0e-5, 1.0, 20.0e6],
    )

    assert type(material) is Material
    assert material.name == "Delayed fuel"
    assert material.has_neutron_multigroup
    assert material.neutron_multigroup.G == 2
    assert material.neutron_multigroup.J == 2


def test_material_accepts_native_multigroup_and_hybrid_data():
    native = Material(nuclide_composition={"H1": 0.1})
    neutron_multigroup = mcdc.NeutronMultigroupData(capture=[0.2])
    multigroup = Material(neutron_multigroup=neutron_multigroup)
    hybrid = Material(
        element_composition={"H": 0.1},
        neutron_multigroup=mcdc.NeutronMultigroupData(
            scatter=[[0.3]],
            energy_grid=[1.0e-5, 20.0e6],
        ),
    )

    assert list(nuclide.name for nuclide in native.nuclides) == ["H1"]
    assert not native.has_neutron_multigroup
    assert native.neutron_multigroup.G == 0
    assert multigroup.has_neutron_multigroup
    assert multigroup.neutron_multigroup is neutron_multigroup
    assert hybrid.has_neutron_multigroup
    assert hybrid.elements[0].name == "H"
    assert hybrid.neutron_multigroup.G == 1


@pytest.mark.parametrize(
    "kwargs, expected_message",
    [
        ({}, "Material requires"),
        (
            {
                "nuclide_composition": {"H1": 0.1},
                "element_composition": {"H": 0.1},
            },
            "Cannot specify both",
        ),
        (
            {"neutron_multigroup": object()},
            "neutron_multigroup must be a NeutronMultigroupData object",
        ),
        (
            {"neutron_multigroup": mcdc.NeutronMultigroupData()},
            "must define at least one energy group",
        ),
        (
            {
                "nuclide_composition": {"H1": 0.1},
                "neutron_multigroup": mcdc.NeutronMultigroupData(capture=[0.2]),
            },
            "requires an explicit neutron multigroup energy_grid",
        ),
        (
            {
                "element_composition": {"H": 0.1},
                "neutron_multigroup": mcdc.NeutronMultigroupData(capture=[0.2]),
            },
            "requires an explicit neutron multigroup energy_grid",
        ),
    ],
)
def test_material_rejects_invalid_representations(kwargs, expected_message, capsys):
    with pytest.raises(SystemExit):
        Material(**kwargs)

    assert expected_message in capsys.readouterr().out


def test_absent_mg_points_to_the_reserved_simulation_object(monkeypatch):
    # Isolate object registration from native data-library loading
    def compile_nuclide(nuclide, simulation):
        nuclide.fissionable = False
        nuclide.element.atomic_number = 1
        MCDCObject._compile_into_simulation(nuclide.element, simulation)
        return MCDCObject._compile_into_simulation(nuclide, simulation)

    monkeypatch.setattr(Nuclide, "_compile_into_simulation", compile_nuclide)

    simulation = mcdc.Simulation()
    simulation.set_model([mcdc.Cell()])
    simulation.compile()
    material = Material(nuclide_composition={"H1": 0.1})

    material._compile_into_simulation(simulation)

    assert material.neutron_multigroup is simulation.neutron_multigroup_data[0]
    assert material.neutron_multigroup.ID == 0
    assert not material.has_neutron_multigroup
    assert simulation.materials == [material]
    assert len(simulation.neutron_multigroup_data) == 1


def test_shared_mg_is_registered_once_for_multiple_materials():
    neutron_multigroup = mcdc.NeutronMultigroupData(capture=[0.1])
    material_a = Material(neutron_multigroup=neutron_multigroup)
    material_b = Material(neutron_multigroup=neutron_multigroup)
    simulation = mcdc.Simulation()
    simulation.set_model([mcdc.Cell()])
    simulation.compile()

    material_a._compile_into_simulation(simulation)
    material_b._compile_into_simulation(simulation)

    assert simulation.materials == [material_a, material_b]
    assert len(simulation.neutron_multigroup_data) == 2
    assert simulation.neutron_multigroup_data[1] is neutron_multigroup
    assert neutron_multigroup.ID == 1


def test_material_mg_reference_is_packed(prepare_simulation):
    neutron_multigroup = mcdc.NeutronMultigroupData(capture=[0.1, 0.2])
    material = Material(neutron_multigroup=neutron_multigroup)
    cell = mcdc.Cell(fill=material)

    simulation_container, _ = prepare_simulation(cells=[cell])
    simulation = simulation_container[0]
    packed = simulation["materials"][0]
    packed_cell = simulation["cells"][0]

    assert simulation["N_material"] == 1
    assert simulation["N_neutron_multigroup_data"] == 2
    assert packed["ID"] == 0
    assert packed["has_neutron_multigroup"]
    assert packed["neutron_multigroup_ID"] == 1
    assert packed_cell["fill_type"] == FILL_MATERIAL
    assert packed_cell["fill_ID"] == 0


def test_multigroup_factory_rejects_zero_group_data(capsys):
    with pytest.raises(SystemExit):
        Material.multigroup()

    assert "must define at least one energy group" in capsys.readouterr().out


def test_element_symbol_from_nuclide_name():
    assert element_symbol_from_nuclide_name("C12") == "C"
    assert element_symbol_from_nuclide_name("Cr52") == "Cr"
    assert element_symbol_from_nuclide_name("Am242m1") == "Am"
    with pytest.raises(ValueError):
        element_symbol_from_nuclide_name("c12")


def test_elements_from_nuclides_do_not_merge_prefix_symbols(monkeypatch):
    monkeypatch.setattr(Element, "_compile_into_simulation", lambda self, sim: True)
    material = Material(
        nuclide_composition={"C12": 1.0, "Cr52": 10.0, "H1": 100.0, "He4": 1000.0}
    )
    set_elements_from_nuclides(material, SimpleNamespace(elements=[]))

    densities = {e.name: d for e, d in material.element_composition.items()}
    assert densities == {"C": 1.0, "Cr": 10.0, "H": 100.0, "He": 1000.0}
