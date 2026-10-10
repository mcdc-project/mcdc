import numpy as np
import pytest

import mcdc
from mcdc.constant import (
    PARTICLE_ANY,
    PARTICLE_NEUTRON,
    PARTICLE_ELECTRON,
    PARTICLE_PROTON,
)
from mcdc.object_.util import decode_particle_type, parse_particle_type
from mcdc.transport.util import particle_name


@pytest.mark.parametrize(
    "name,code",
    [
        ("neutron", PARTICLE_NEUTRON),
        ("electron", PARTICLE_ELECTRON),
        ("proton", PARTICLE_PROTON),
    ],
)
def test_particle_conversion_across_apis(name, code):
    assert parse_particle_type(name) == code
    assert decode_particle_type(code) == name.capitalize()
    assert particle_name(code) == name
    assert mcdc.Source(particle_type=name).particle_type == code
    tally = mcdc.Tally(particle_type=name)
    assert tally.particle_type == code
    assert f"Particle: {name.capitalize()}" in tally._phasespace_filter_text()
    simulation = mcdc.Simulation()
    simulation.technique.weight_windows(np.array([0.5, 1.0, 2.0]), particle_type=name)
    window = getattr(simulation.technique, f"{name}_weight_windows")
    assert window.ptype == code
    assert window.active


def test_wildcard_and_unknown_names():
    assert mcdc.Tally().particle_type == PARTICLE_ANY
    assert decode_particle_type(PARTICLE_ANY) == "Any"
    assert particle_name(PARTICLE_ANY) == "any particle"
    assert decode_particle_type(-1, "Unspecified") == "Unspecified"
    assert particle_name(-1) == "unknown particle type -1"


@pytest.mark.parametrize("name", ["any", "photon", "Neutron", None])
def test_concrete_species_validation(name):
    with pytest.raises(ValueError):
        parse_particle_type(name)
    with pytest.raises(SystemExit):
        mcdc.Source(particle_type=name)
    if name is not None:
        with pytest.raises(SystemExit):
            mcdc.Tally(particle_type=name)
