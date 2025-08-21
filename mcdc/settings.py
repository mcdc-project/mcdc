import h5py
import numpy as np

from dataclasses import dataclass, field
from numpy.typing import NDArray

####

import mcdc.objects as objects

from mcdc.constant import *
from mcdc.material import MaterialMG
from mcdc.objects import ObjectSingleton
from mcdc.prints import print_error


@dataclass
class Settings(ObjectSingleton):
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
    save_input_deck: bool = True

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

    # Portability
    target_gpu: bool = False

    def __post_init__(self):
        super().__init__('settings')

        # Recasting
        self.N_particle = int(self.N_particle)
        self.active_bank_buffer = int(self.active_bank_buffer)

        # Set multgroup mode
        self.multigroup_mode = isinstance(objects.materials[0], MaterialMG)

        # Register the settings
        objects.settings = self

    def set_time_census(self, time, tally_frequency=None):
        """
        Set time census

        Parameters
        ----------
        time : array_like[float]
            The time-census boundaries.
        tally_frecuency : integer, optional
            Number of uniform tally time mesh bins in census-based tallying.
            This overrides manual tally time mesh definitions.
        """

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
            # Reset all tallies' time grids:
            self.use_census_based_tally = True
            self.census_tally_frequency = tally_frequency

    def set_eigenmode(
        self, N_inactive=0, N_active=0, k_init=1.0, gyration_radius=None, save_particle=False
    ):
        """
        Set eigenvalue-related settings

        Parameters
        ----------
        N_inactive : int
            Number of cycles not included when averaging the k-eigenvalue (default 0).
        N_active : int
            Number of cycles to include for statistics of the k-eigenvalue (default 0).
        k_init : float
            Initial k value to iterate on (default 1.0).
        gyration_radius : float, optional
            Specify a gyration radius (default None).
        save_particle : bool
            Whether final particle bank outputs (default False).
        """

        # Update setting self
        self.N_inactive = N_inactive
        self.N_active = N_active
        self.N_cycle = self.N_inactive + self.N_active
        self.eigenvalue_mode = True
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

    def set_source_file(self, source_file_name):
        self.use_source_file = True
        self.source_file_name = source_file_name

        # Set number of particles
        with h5py.File(source_file_name, "r") as f:
            self.N_particle = f["particles_size"][()]

def is_sorted(a):
    return np.all(a[:-1] <= a[1:])
