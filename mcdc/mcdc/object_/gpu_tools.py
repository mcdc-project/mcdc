from dataclasses import dataclass
from numpy import uintp

####

from mcdc.object_.base import ObjectSingleton


@dataclass
class GPUMeta(ObjectSingleton):
    # Annotations for Numba mode
    label: str = "gpu_meta"
    #
    state_pointer: uintp = uintp(0)
    program_pointer: uintp = uintp(0)
    simulation_pointer: uintp = uintp(0)
    data_pointer: uintp = uintp(0)

    # Note that the uintp is manually overriden in code_factory.
