import numpy as np

from dataclasses import dataclass, field
from typing import Annotated
from numpy import int64, uint64
from numpy.typing import NDArray

####

from mcdc.constant import PARTICLE_NEUTRON
from mcdc.object_.base import MCDCBase


@dataclass
class ParticleData(MCDCBase):
    """Serializable phase-space state stored in particle banks."""

    # MC/DC framework metadata
    label = "particle_data"

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    t: float = 0.0
    ux: float = 0.0
    uy: float = 0.0
    uz: float = 0.0
    E: float = 0.0
    w: float = 0.0
    particle_type: int = PARTICLE_NEUTRON
    rng_seed: uint64 = uint64(1)


@dataclass
class InteractionData(MCDCBase):
    """Scoring data shared by discrete collisions and condensed interactions.

    Each treatment records its incoming particle state before changing it and
    accumulates weighted energy deposition in eV. Condensed interactions use
    the step endpoint with energy and direction from before their application.
    """

    # MC/DC framework metadata
    label = "interaction_data"

    energy_deposition: float = 0.0
    incident_particle: ParticleData = field(default_factory=ParticleData)


@dataclass
class Particle(ParticleData):
    """Active transport particle with geometry and event-tracking state."""

    # MC/DC framework metadata
    label = "particle"

    cell_ID: int = -1
    material_ID: int = -1
    surface_ID: int = -1
    unstructured_mesh_ID: int = -1
    alive: bool = False
    fresh: bool = False
    event: int = -1


class ParticleBank(MCDCBase):
    """Particle storage metadata used by the compiled runtime.

    Parameters
    ----------
    tag : str
        Bank role, such as ``"active"``, ``"source"``, ``"census"``, or
        ``"future"``.
    """

    # MC/DC framework metadata
    label = "particle_bank"
    non_numba = ["particles"]

    particles: list[ParticleData] = []  # Non-numba
    size: Annotated[NDArray[int64], (1,)]
    tag: str = ""

    def __init__(self, tag):
        self.tag = tag
        self.size = np.zeros(1, dtype=int64)
