import numpy as np

from dataclasses import dataclass, field
from numpy.typing import NDArray

import mcdc.objects

from mcdc.constant import GYRATION_RADIUS_ALL


@dataclass
class Settings:
    # Basic
    N_particle: int = 0
    N_batch: int = 1
    rng_seed: int = 1

    # Simulation mode
    multigroup_mode: bool = False
    eigenvalue_mode: bool = False

    # k-eigenvalue
    N_inactive: int = 0
    N_active: int = 0
    k_init: float = 1.0
    use_gyration_radius: bool = False
    gyration_radius_type: int = GYRATION_RADIUS_ALL

    # Particle source
    use_source_file: bool = False
    source_file_name: str = ""

    # Misc.
    time_boundary: float = np.inf
    output_name: str = "output"
    use_progress_bar: bool = True

    # Time census
    N_census: int = 1
    census_time: NDArray[np.float64] = field(default_factory=lambda: np.array([np.inf]))
    use_census_based_tally: bool = False
    census_tally_frequency: int = 0

    # Initial condition source
    use_IC_file: bool = False
    IC_file_name: str = ""
    N_precursor: int = 0

    # Particle bank-related
    save_particle: bool = False
    active_bank_buffer: int = 100
    census_bank_buffer_ratio: float = 1.0
    source_bank_buffer_ratio: float = 1.0
    future_bank_buffer_ratio: float = 0.5

    def __post_init__(self):
        # Recasting
        self.N_particle = int(self.N_particle)

        # Following ObjectBase
        self.type = "settings"
        self.derived_class = False
        self.numba_ID = -1

        # Register the settings
        mcdc.objects.settings = self
