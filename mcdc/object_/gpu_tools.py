from dataclasses import dataclass
from numpy import uint64

####

from mcdc.object_.base import ObjectSingleton


@dataclass
class GPUMeta(ObjectSingleton):
    # Annotations for Numba mode
    label: str = "gpu_meta"
    #
    state_pointer: uint64 = uint64(0)
    program_pointer: uint64 = uint64(0)
    simulation_pointer: uint64 = uint64(0)
    data_pointer: uint64 = uint64(0)
