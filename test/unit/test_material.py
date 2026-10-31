import doctest
import h5py
import pytest
from types import SimpleNamespace

import mcdc
from mcdc.constant import FILL_MATERIAL
from mcdc.object_.base import MCDCObject, MCDCPolymorphic
from mcdc.object_.material import (
    Material,
    set_elements_from_nuclides,
    set_nuclides_from_elements,
)
from mcdc.object_.nuclide import Nuclide
from mcdc.object_.element import Element
from mcdc.object_.util import (
    element_symbol_from_nuclide_name,
    ISOTOPIC_ABUNDANCE,
    is_nuclide_name,
)
from mcdc.mcdc_get import material as get_material


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
    native = Material(number_density={"H1": 0.1})
    neutron_multigroup = mcdc.NeutronMultigroupData(capture=[0.2])
    multigroup = Material(neutron_multigroup=neutron_multigroup)
    hybrid = Material(
        number_density={"H": 0.1},
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
                "number_density": {"H1": 0.1},
                "mass_density": {"H1": 0.1},
            },
            "Cannot specify more than one of",
        ),
        (
            {
                "number_density": {"H1": 0.1},
                "mass_fractions": {"H1": 0.1},
            },
            "Cannot specify more than one of",
        ),
        (
            {
                "mass_fractions": {"H1": 0.1},
                "mass_density": {"H1": 0.1},
            },
            "Cannot specify more than one of",
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
                "number_density": {"H1": 0.1},
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
    material = Material(number_density={"H1": 0.1})

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
        number_density={"C12": 1.0, "Cr52": 10.0, "H1": 100.0, "He4": 1000.0}
    )
    set_elements_from_nuclides(material, SimpleNamespace(elements=[]))

    densities = {e.name: d for e, d in material.element_composition.items()}
    assert densities == {"C": 1.0, "Cr": 10.0, "H": 100.0, "He": 1000.0}


# Mixed nuclide and element compositions
# --------------------------------------


@pytest.fixture
def native_simulation(monkeypatch):
    # Isolate object registration from native data-library loading
    def compile_nuclide(nuclide, simulation):
        nuclide.fissionable = False
        return MCDCObject._compile_into_simulation(nuclide, simulation)

    def compile_element(element, simulation):
        return MCDCObject._compile_into_simulation(element, simulation)

    monkeypatch.setattr(Nuclide, "_compile_into_simulation", compile_nuclide)
    monkeypatch.setattr(Element, "_compile_into_simulation", compile_element)

    simulation = mcdc.Simulation()
    simulation.set_model([mcdc.Cell()])
    simulation.compile()
    return simulation


def natural(element, density):
    # Natural isotope densities of an element, in table order
    norm = sum(ISOTOPIC_ABUNDANCE[element].values())
    return {
        name: density * abundance / norm
        for name, abundance in ISOTOPIC_ABUNDANCE[element].items()
    }


def nuclide_view(material):
    return {n.name: d for n, d in material.nuclide_composition.items()}


def element_view(material):
    return {e.name: d for e, d in material.element_composition.items()}


def test_mixed_composition_nuclide_view_expands_only_the_elements(native_simulation):
    material = Material(number_density={"U235": 0.1, "O": 0.2})

    set_nuclides_from_elements(material, native_simulation)

    expected = {"U235": 0.1, **natural("O", 0.2)}
    assert nuclide_view(material) == pytest.approx(expected)
    assert list(nuclide_view(material)) == list(expected)
    assert material.nuclide_densities == pytest.approx(list(expected.values()))
    assert [n.name for n in material.nuclides] == list(expected)


def test_mixed_composition_element_view_collapses_only_the_nuclides(native_simulation):
    material = Material(number_density={"U235": 0.1, "U238": 0.3, "O": 0.2})

    set_elements_from_nuclides(material, native_simulation)

    assert element_view(material) == pytest.approx({"U": 0.4, "O": 0.2})
    assert material.element_densities == pytest.approx([0.4, 0.2])
    assert [e.name for e in material.elements] == ["U", "O"]


def test_views_are_derived_from_the_input_not_from_each_other(native_simulation):
    material = Material(number_density={"U235": 0.1, "O": 0.2})

    set_nuclides_from_elements(material, native_simulation)
    set_elements_from_nuclides(material, native_simulation)

    # Collapsing must not re-count the isotopes expanded from O
    assert element_view(material) == pytest.approx({"U": 0.1, "O": 0.2})
    assert nuclide_view(material) == pytest.approx({"U235": 0.1, **natural("O", 0.2)})


def test_derivations_are_idempotent(native_simulation):
    material = Material(number_density={"U235": 0.1, "O": 0.2})

    set_nuclides_from_elements(material, native_simulation)
    set_elements_from_nuclides(material, native_simulation)
    nuclides, elements = nuclide_view(material), element_view(material)
    set_nuclides_from_elements(material, native_simulation)
    set_elements_from_nuclides(material, native_simulation)

    assert nuclide_view(material) == nuclides
    assert element_view(material) == elements


def test_repeated_nuclides_are_summed(native_simulation):
    material = Material(number_density={"O16": 1.0, "O": 2.0})

    set_nuclides_from_elements(material, native_simulation)

    expected = natural("O", 2.0)
    expected["O16"] += 1.0
    assert nuclide_view(material) == pytest.approx(expected)
    assert len(material.nuclides) == len(expected)


def test_element_only_composition_expands_to_natural_isotopes(native_simulation):
    material = Material(number_density={"H": 0.1, "O": 0.2})

    set_nuclides_from_elements(material, native_simulation)

    expected = {**natural("H", 0.1), **natural("O", 0.2)}
    assert list(nuclide_view(material)) == list(expected)
    assert nuclide_view(material) == pytest.approx(expected)
    # The user's elements are left as given
    assert [e.name for e in material.elements] == ["H", "O"]


def test_single_kind_compositions_are_left_alone_by_the_other_derivation(
    native_simulation,
):
    nuclides = Material(number_density={"H1": 0.1})
    elements = Material(number_density={"H": 0.1})
    nuclide_before = nuclide_view(nuclides)
    element_before = element_view(elements)

    set_nuclides_from_elements(nuclides, native_simulation)
    set_elements_from_nuclides(elements, native_simulation)

    assert nuclide_view(nuclides) == nuclide_before
    assert element_view(elements) == element_before


def test_multigroup_only_material_is_left_alone_by_both_derivations(native_simulation):
    material = Material.multigroup(capture=[1.0])

    set_nuclides_from_elements(material, native_simulation)
    set_elements_from_nuclides(material, native_simulation)

    assert material.nuclides == []
    assert material.elements == []


def test_element_without_abundance_data_is_rejected(native_simulation, capsys):
    material = Material(number_density={"Xx": 0.1})

    with pytest.raises(SystemExit):
        set_nuclides_from_elements(material, native_simulation)

    assert (
        "No natural isotopic abundance data for element Xx" in capsys.readouterr().out
    )


# Mass density and mass fractions
# -------------------------------

AWR = {  # neutron-mass atomic weight ratios, as stored in the data library
    "H1": 0.999167,
    "H2": 1.99626,
    "O16": 15.8575,
    "O17": 16.8531,
    "O18": 17.8445,
    "U235": 233.025,
    "U238": 236.006,
}
N_A_BARN_CM = 0.602214076
AMU = 1.00866491595


@pytest.fixture
def atomic_weight_library(tmp_path, monkeypatch):
    """Provide target masses independently of the reaction-loading mocks."""
    monkeypatch.setenv("MCDC_LIB", str(tmp_path))
    for name, ratio in AWR.items():
        with h5py.File(tmp_path / f"neutron-{name}-293.6K.h5", "w") as file:
            file.attrs["format"] = "mcdc-data"
            file.attrs["format-version"] = "0.1.0"
            file["atomic-weight-ratio"] = ratio
    return tmp_path


@pytest.fixture
def weighted_simulation(monkeypatch, atomic_weight_library):
    # Provide atomic weights without reading the data library
    def compile_nuclide(nuclide, simulation):
        nuclide.fissionable = False
        nuclide.atomic_weight_ratio = AWR[nuclide.name]
        return MCDCObject._compile_into_simulation(nuclide, simulation)

    def compile_element(element, simulation):
        return MCDCObject._compile_into_simulation(element, simulation)

    monkeypatch.setattr(Nuclide, "_compile_into_simulation", compile_nuclide)
    monkeypatch.setattr(Element, "_compile_into_simulation", compile_element)

    simulation = mcdc.Simulation()
    simulation.set_model([mcdc.Cell()])
    simulation.compile()
    return simulation


def atoms(rho, awr):
    return rho * N_A_BARN_CM / (awr * AMU)


def element_weight(element):
    abundances = ISOTOPIC_ABUNDANCE[element]
    weighted = sum(a * AWR[isotope] * AMU for isotope, a in abundances.items())
    return weighted / sum(abundances.values())


def test_mass_density_is_stored_with_placeholder_atomic_densities():
    material = Material(mass_density={"U235": 0.2, "O": 1.0})

    assert material.mass_density == {"U235": 0.2, "O": 1.0}
    assert [n.name for n in material.nuclides] == ["U235"]
    assert [e.name for e in material.elements] == ["O"]
    assert material.number_density == {"U235": 0.0, "O": 0.0}


def test_materials_without_mass_inputs_have_no_mass_density():
    assert Material(number_density={"H1": 0.1}).mass_density == {}


def test_nuclide_mass_density_is_converted_at_compile(weighted_simulation):
    material = Material(mass_density={"U235": 0.195, "U238": 8.7})

    material._compile_into_simulation(weighted_simulation)

    expected = [atoms(0.195, AWR["U235"]), atoms(8.7, AWR["U238"])]
    assert material.nuclide_densities == pytest.approx(expected)
    assert list(nuclide_view(material).values()) == pytest.approx(expected)
    assert list(material.number_density.values()) == pytest.approx(expected)
    assert list(material.number_density) == ["U235", "U238"]


def test_element_mass_density_uses_the_abundance_weighted_weight(weighted_simulation):
    material = Material(mass_density={"O": 1.195})

    material._compile_into_simulation(weighted_simulation)

    expected = 1.195 * N_A_BARN_CM / element_weight("O")
    assert material.element_densities == pytest.approx([expected])
    assert material.number_density["O"] == pytest.approx(expected)
    assert weighted_simulation.nuclides == []
    assert not hasattr(material.elements[0], "atomic_weight_ratio")


def test_converted_element_expands_to_the_same_total_density(weighted_simulation):
    material = Material(mass_density={"O": 1.195})
    material._compile_into_simulation(weighted_simulation)

    set_nuclides_from_elements(material, weighted_simulation)

    # The isotopes carry the element's converted atomic density
    assert sum(material.nuclide_densities) == pytest.approx(
        material.number_density["O"]
    )
    assert list(nuclide_view(material)) == list(natural("O", 1.0))


def test_mixed_mass_density_converts_each_name_by_its_kind(weighted_simulation):
    material = Material(mass_density={"U235": 0.2, "O": 1.0, "H1": 0.1})

    material._compile_into_simulation(weighted_simulation)

    assert material.number_density == pytest.approx(
        {
            "U235": atoms(0.2, AWR["U235"]),
            "O": 1.0 * N_A_BARN_CM / element_weight("O"),
            "H1": atoms(0.1, AWR["H1"]),
        }
    )


def test_mass_fractions_scale_the_density(weighted_simulation):
    by_fraction = Material(
        density=10.0, mass_fractions={"U235": 0.02, "U238": 0.86, "O16": 0.12}
    )
    by_density = Material(mass_density={"U235": 0.2, "U238": 8.6, "O16": 1.2})

    by_fraction._compile_into_simulation(weighted_simulation)
    by_density._compile_into_simulation(weighted_simulation)

    assert by_fraction.mass_density == pytest.approx(by_density.mass_density)
    assert by_fraction.nuclide_densities == pytest.approx(by_density.nuclide_densities)


def test_mass_fractions_are_normalized_and_warn_when_far_from_one(capsys):
    material = Material(density=10.0, mass_fractions={"H1": 0.5, "O16": 1.5})

    assert material.mass_density == pytest.approx({"H1": 2.5, "O16": 7.5})
    assert "normalized" in capsys.readouterr().out


def test_mass_fractions_close_to_one_are_normalized_without_a_warning(capsys):
    # Rounded fractions such as 0.0193 + 0.862 + 0.1184 = 0.9997 are routine
    material = Material(
        density=10.09, mass_fractions={"U235": 0.0193, "U238": 0.862, "O16": 0.1184}
    )

    assert sum(material.mass_density.values()) == pytest.approx(10.09)
    assert "normalized" not in capsys.readouterr().out


@pytest.mark.parametrize(
    "kwargs, expected_message",
    [
        (
            {"mass_density": {"H1": 1.0}, "number_density": {"H1": 0.1}},
            "Cannot specify more than one of",
        ),
        (
            {"mass_fractions": {"H1": 1.0}, "number_density": {"H1": 0.1}},
            "Cannot specify more than one of",
        ),
        (
            {"mass_density": {"H1": 1.0}, "mass_fractions": {"H1": 1.0}},
            "Cannot specify more than one of",
        ),
        ({"mass_fractions": {"H1": 1.0}}, "must be specified together"),
        ({"density": 1.0}, "must be specified together"),
        ({"density": 0.0, "mass_fractions": {"H1": 1.0}}, "density must be positive"),
        ({"mass_density": {"H1": -1.0}}, "must be non-negative"),
        (
            {"density": 1.0, "mass_fractions": {"H1": -0.5, "O16": 1.5}},
            "must be non-negative",
        ),
        (
            {"density": 1.0, "mass_fractions": {"H1": 0.0}},
            "must have a positive sum",
        ),
        (
            {
                "mass_density": {"H1": 1.0},
                "neutron_multigroup": mcdc.NeutronMultigroupData(capture=[0.2]),
            },
            "requires an explicit neutron multigroup energy_grid",
        ),
    ],
)
def test_mass_inputs_are_validated(kwargs, expected_message, capsys):
    with pytest.raises(SystemExit):
        Material(**kwargs)

    assert expected_message in capsys.readouterr().out


@pytest.fixture
def compile_without_data(monkeypatch, atomic_weight_library):
    # Full simulation compilation without reading data-library files
    def compile_nuclide(nuclide, simulation):
        nuclide.fissionable = False
        nuclide.mass_number = 1
        nuclide.excitation_level = 0
        nuclide.radiation_length = 1.0
        nuclide.atomic_weight_ratio = AWR[nuclide.name]
        return MCDCObject._compile_into_simulation(nuclide, simulation)

    def compile_element(element, simulation):
        element.atomic_number = 1
        return MCDCObject._compile_into_simulation(element, simulation)

    monkeypatch.setattr(Nuclide, "_compile_into_simulation", compile_nuclide)
    monkeypatch.setattr(Nuclide, "set_neutron_data", lambda self, simulation: None)
    monkeypatch.setattr(Element, "_compile_into_simulation", compile_element)
    monkeypatch.setattr(Element, "set_electron_data", lambda self, simulation: None)


def test_mass_inputs_survive_neutron_simulation_compilation(compile_without_data):
    material = Material(mass_density={"U235": 0.2, "O": 1.0})
    simulation = mcdc.Simulation()
    simulation.set_model([mcdc.Cell(fill=material)])
    simulation.set_sources([mcdc.Source(particle_type="neutron", energy=1.0)])

    simulation.compile()

    # The element is expanded into isotopes carrying its converted density
    oxygen = 1.0 * N_A_BARN_CM / element_weight("O")
    expected = {"U235": atoms(0.2, AWR["U235"]), **natural("O", oxygen)}
    assert nuclide_view(material) == pytest.approx(expected)
    assert list(material.number_density) == ["U235", "O"]


def test_mass_inputs_survive_electron_simulation_compilation(compile_without_data):
    material = Material(mass_density={"U235": 0.2, "U238": 8.6, "H1": 0.1})
    simulation = mcdc.Simulation()
    simulation.set_model([mcdc.Cell(fill=material)])
    simulation.set_sources([mcdc.Source(particle_type="electron", energy=1.0)])

    simulation.compile()

    # Nuclides collapse into elements using their converted atomic densities
    assert element_view(material) == pytest.approx(
        {
            "U": atoms(0.2, AWR["U235"]) + atoms(8.6, AWR["U238"]),
            "H": atoms(0.1, AWR["H1"]),
        }
    )


# Additional coverage
# -------------------


@pytest.mark.parametrize(
    "name, expected",
    [
        ("U235", True),
        ("Cr52", True),
        ("H1", True),
        ("Am242m1", True),
        ("U", False),
        ("Cr", False),
        ("u235", False),
        ("235U", False),
        ("U235x", False),
        ("U-235", False),
        ("", False),
    ],
)
def test_is_nuclide_name(name, expected):
    assert is_nuclide_name(name) is expected


def test_mass_inputs_are_packed_as_atomic_densities(
    compile_without_data, prepare_simulation
):
    material = Material(mass_density={"U235": 0.2, "H1": 0.1})
    source = mcdc.Source(particle_type="neutron", energy=1.0)

    simulation_container, data = prepare_simulation(
        cells=[mcdc.Cell(fill=material)], sources=[source]
    )
    packed = simulation_container[0]["materials"][0]

    # Transport must see atoms/(barn cm), not the g/cm^3 the user supplied
    assert get_material.nuclide_densities_all(packed, data) == pytest.approx(
        [atoms(0.2, AWR["U235"]), atoms(0.1, AWR["H1"])]
    )


def test_mass_inputs_with_neutron_and_electron_transport_together(
    compile_without_data,
):
    material = Material(mass_density={"U235": 0.2, "O": 1.0})
    simulation = mcdc.Simulation()
    simulation.set_model([mcdc.Cell(fill=material)])
    simulation.set_sources(
        [
            mcdc.Source(particle_type="neutron", energy=1.0),
            mcdc.Source(particle_type="electron", energy=1.0),
        ]
    )

    simulation.compile()

    oxygen = 1.0 * N_A_BARN_CM / element_weight("O")
    uranium = atoms(0.2, AWR["U235"])
    # Each view is built from the converted input, so neither re-counts the other
    assert nuclide_view(material) == pytest.approx(
        {"U235": uranium, **natural("O", oxygen)}
    )
    assert element_view(material) == pytest.approx({"U": uranium, "O": oxygen})


def test_mass_inputs_compile_identically_into_separate_simulations(
    weighted_simulation, compile_without_data
):
    material = Material(mass_density={"U235": 0.2, "O": 1.0})
    material._compile_into_simulation(weighted_simulation)
    first = dict(material.number_density)

    other = mcdc.Simulation()
    other.set_model([mcdc.Cell()])
    other.compile()
    material._compile_into_simulation(other)

    assert material.number_density == pytest.approx(first)
    assert material.mass_density == {"U235": 0.2, "O": 1.0}


def test_mass_inputs_combine_with_hybrid_multigroup_data(weighted_simulation):
    material = Material(
        mass_density={"H1": 0.1},
        neutron_multigroup=mcdc.NeutronMultigroupData(
            capture=[0.2], energy_grid=[1.0e-5, 20.0e6]
        ),
    )

    material._compile_into_simulation(weighted_simulation)

    assert material.has_neutron_multigroup
    assert material.nuclide_densities == pytest.approx([atoms(0.1, AWR["H1"])])


def test_material_docstring_examples_are_valid_python():
    examples = doctest.DocTestParser().get_examples(Material.__doc__)

    assert len(examples) >= 5
    for example in examples:
        compile(example.source, "<Material docstring>", "exec")
