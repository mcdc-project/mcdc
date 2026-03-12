import h5py
from h5py._hl.dataset import sel
import numpy as np

from dataclasses import dataclass, field
from numpy.typing import NDArray

####

from mcdc.constant import *
from mcdc.object_.base import ObjectSingleton
from mcdc.object_.util import is_sorted
from mcdc.print_ import print_error

# ======================================================================================
# Settings
# ======================================================================================


@dataclass
class Settings(ObjectSingleton):
    # Annotations for Numba mode
    label: str = "settings"

    # Basic
    N_particle: int = 0
    N_batch: int = 1
    rng_seed: int = 1

    # k-eigenvalue
    N_inactive: int = 0
    N_active: int = 0
    N_cycle: int = 0
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

    # Particle bank-related
    save_particle: bool = False
    active_bank_buffer: int = 100
    census_bank_buffer_ratio: float = 2.0
    source_bank_buffer_ratio: float = 2.0
    future_bank_buffer_ratio: float = 1.5

    # Multi-particle options
    neutron_transport: bool = True
    electron_transport: bool = False
    proton_transport: bool = False

    # Neutron transport modes
    neutron_multigroup_mode: bool = False
    neutron_eigenvalue_mode: bool = False

    # GPU mode
    gpu_strategy: int = GPU_STRATEGY_SIMPLE_ASYNC

    def __post_init__(self):
        super().__init__()

    def set_time_census(self, time, tally_frequency=None):
        # Make sure that the time grid points are sorted
        if not is_sorted(time):
            print_error("Time census: Time grid points have to be sorted.")

        # Make sure that the starting point is larger than zero
        if time[0] <= 0.0:
            print_error("Time census: First census time should be larger than zero.")

        # Add the default, final census-at-infinity
        time = np.append(time, np.inf)

        # Set the time census parameters
        self.census_time = time
        self.N_census = len(self.census_time)

        # Set the census-based tallying
        if tally_frequency is not None and tally_frequency > 0:
            # Flag to reset all tallies' time grids (done in main.py)
            self.use_census_based_tally = True
            self.census_tally_frequency = tally_frequency

    def set_eigenmode(
        self,
        N_inactive=0,
        N_active=0,
        k_init=1.0,
        gyration_radius=None,
        save_particle=False,
    ):
        # Update setting self
        self.N_inactive = N_inactive
        self.N_active = N_active
        self.N_cycle = self.N_inactive + self.N_active
        self.neutron_eigenvalue_mode = True
        self.k_init = k_init
        self.save_particle = save_particle

        # Gyration radius setup
        if gyration_radius is not None:
            self.use_gyration_radius = True
            if gyration_radius == "all":
                self.gyration_radius_type = GYRATION_RADIUS_ALL
            elif gyration_radius == "infinite-x":
                self.gyration_radius_type = GYRATION_RADIUS_INFINITE_X
            elif gyration_radius == "infinite-y":
                self.gyration_radius_type = GYRATION_RADIUS_INFINITE_Y
            elif gyration_radius == "infinite-z":
                self.gyration_radius_type = GYRATION_RADIUS_INFINITE_Z
            elif gyration_radius == "only-x":
                self.gyration_radius_type = GYRATION_RADIUS_ONLY_X
            elif gyration_radius == "only-y":
                self.gyration_radius_type = GYRATION_RADIUS_ONLY_Y
            elif gyration_radius == "only-z":
                self.gyration_radius_type = GYRATION_RADIUS_ONLY_Z
            else:
                print_error("Unknown gyration radius type")

        # Allocate cycle-wise quantities
        from mcdc.object_.simulation import simulation

        simulation.k_cycle = np.zeros(self.N_cycle)
        simulation.gyration_radius = np.zeros(self.N_cycle)

    def set_source_file(self, source_file_name):
        self.use_source_file = True
        self.source_file_name = source_file_name

        # Set number of particles
        with h5py.File(source_file_name, "r") as f:
            self.N_particle = int(f["particles_size"][()])

    def set_transported_particles(self, transported_particles):
        # Reset the flags
        self.neutron_transport = False
        self.electron_transport = False
        self.proton_transport = False

        # Set flags
        for particle in transported_particles:
            if particle == "neutron":
                self.neutron_transport = True
            elif particle == "electron":
                self.electron_transport = True
            elif particle == "proton":
                self.proton_transport = True
            else:
                print_error(r"Unsupported particle types: {particle}")
