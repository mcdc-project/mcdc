import numpy as np

from dataclasses import dataclass, field
from typing import Annotated
from numpy import int64, uint
from numpy.typing import NDArray

####

from mcdc.constant import PARTICLE_NEUTRON
from mcdc.object_.base import ObjectBase, ObjectSingleton


@dataclass
class ParticleData(ObjectBase):
    label: str = "particle_data"
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    t: float = 0.0
    ux: float = 0.0
    uy: float = 0.0
    uz: float = 0.0
    g: int = -1
    E: float = 0.0
    w: float = 0.0
    particle_type: int = PARTICLE_NEUTRON
    rng_seed: uint = uint(1)


@dataclass
class Particle(ParticleData):
    label: str = "particle"
    cell_ID: int = -1
    material_ID: int = -1
    surface_ID: int = -1
    alive: bool = False
    fresh: bool = False
    event: int = -1


class ParticleBank(ObjectSingleton):
    label: str = "particle_bank"
    non_numba: list[str] = ["particles"]
    particles: list[ParticleData] = []
    size: Annotated[NDArray[int64], (1,)]
    tag: str = ""

    def __init__(self, tag):
        super().__init__()
        self.tag = tag
        self.size = np.zeros(1, dtype=int64)
