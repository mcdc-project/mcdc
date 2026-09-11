import h5py
import numpy as np

from dataclasses import dataclass, field
from numpy.typing import ArrayLike, NDArray

####

from mcdc.constant import *
from mcdc.object_.base import MCDCBase
from mcdc.object_.util import is_sorted
from mcdc.print_ import print_error

# ======================================================================================
# Settings
# ======================================================================================


@dataclass
class Settings(MCDCBase):
    """Execution and transport settings owned by a simulation."""

    # MC/DC framework metadata
    label = "settings"

    # Basic
    #: Number of particle histories simulated per batch or eigenvalue cycle.
    #: The default is ``0``.
    N_particle: int = 0
    #: Number of statistically independent fixed-source batches. The default
    #: is ``1``.
    N_batch: int = 1
    #: Seed used to initialize the pseudorandom-number generator. The default
    #: is ``1``.
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
    #: Time in seconds at which particle transport terminates. The default is
    #: infinity.
    time_boundary: float = np.inf
    #: Base name used for the HDF5 output file. The default is ``"output"``.
    output_name: str = "output"
    #: Whether to display transport progress. The default is ``True``.
    use_progress_bar: bool = True

    # Time census
    N_census: int = 1
    census_time: NDArray[np.float64] = field(default_factory=lambda: np.array([np.inf]))
    use_census_based_tally: bool = False
    census_tally_frequency: int = 0

    # Particle bank-related
    save_particle: bool = False
    #: Additional particle capacity allocated for the active bank. The default
    #: is ``100``.
    active_bank_buffer: int = 100
    #: Capacity multiplier used when allocating the census bank. The default
    #: is ``2.0``.
    census_bank_buffer_ratio: float = 2.0
    #: Capacity multiplier used when allocating the source bank. The default
    #: is ``2.0``.
    source_bank_buffer_ratio: float = 2.0
    #: Capacity multiplier used when allocating the future bank. The default
    #: is ``1.5``.
    future_bank_buffer_ratio: float = 1.5

    # Multi-particle options
    neutron_transport: bool = True
    electron_transport: bool = False
    proton_transport: bool = False

    # Neutron transport modes
    neutron_eigenvalue_mode: bool = False

    # Target info
    target: int = CPU

    # GPU mode
    gpu_strategy: int = GPU_STRATEGY_ASYNC
    gpu_async_type: int = GPU_ASYNC_SIMPLE
    gpu_storage: int = GPU_STORAGE_SEPARATE

    def set_time_census(
        self, time: ArrayLike, tally_frequency: int | None = None
    ) -> None:
        """Configure census times for time-dependent transport.

        Parameters
        ----------
        time : array_like of float
            Positive, nondecreasing census times in seconds. An infinite final
            census is appended automatically.
        tally_frequency : int, optional
            Number of tally intervals per census period. A positive value
            enables census-based tally output.

        Examples
        --------
        Configure explicit census times:

        >>> import mcdc
        >>> simulation = mcdc.Simulation()
        >>> simulation.settings.set_time_census(
        ...     time=[1.0e-6, 2.0e-6, 5.0e-6],
        ... )

        Enable census-based tallies with ten intervals per census period:

        >>> simulation.settings.set_time_census(
        ...     time=[1.0e-6, 2.0e-6, 5.0e-6],
        ...     tally_frequency=10,
        ... )
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
            # Flag to reset all tally time grids during simulation compilation
            self.use_census_based_tally = True
            self.census_tally_frequency = tally_frequency

    def set_eigenmode(
        self,
        N_inactive: int = 0,
        N_active: int = 0,
        k_init: float = 1.0,
        gyration_radius: str | None = None,
        save_particle: bool = False,
    ) -> None:
        """Enable neutron k-eigenvalue mode.

        Parameters
        ----------
        N_inactive : int, optional
            Number of inactive cycles.
        N_active : int, optional
            Number of active cycles used for statistics.
        k_init : float, optional
            Initial multiplication-factor estimate.
        gyration_radius : str, optional
            Gyration-radius mode: ``"all"``, ``"infinite-x"``,
            ``"infinite-y"``, ``"infinite-z"``, ``"only-x"``, ``"only-y"``,
            or ``"only-z"``.
        save_particle : bool, optional
            Whether to save source-bank particles.

        Examples
        --------
        Configure a standard eigenvalue calculation:

        >>> import mcdc
        >>> simulation = mcdc.Simulation()
        >>> simulation.settings.set_eigenmode(
        ...     N_inactive=20,
        ...     N_active=100,
        ...     k_init=1.0,
        ... )

        Score the source gyration radius and save source particles:

        >>> simulation.settings.set_eigenmode(
        ...     N_inactive=20,
        ...     N_active=100,
        ...     gyration_radius="all",
        ...     save_particle=True,
        ... )
        """
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

    def set_source_file(self, source_file_name: str) -> None:
        """Use particles from an HDF5 source file.

        The particle count is read from the file's ``particles_size`` dataset.

        Parameters
        ----------
        source_file_name : str or path-like
            Source-particle HDF5 file.

        Examples
        --------
        Initialize a simulation from a previously written particle source:

        >>> import mcdc
        >>> simulation = mcdc.Simulation()
        >>> simulation.settings.set_source_file("source.h5")
        """
        self.use_source_file = True
        self.source_file_name = source_file_name

        # Set number of particles
        with h5py.File(source_file_name, "r") as f:
            self.N_particle = int(f["particles_size"][()])

    def set_transported_particles(self, transported_particles: list[str]) -> None:
        """Select the particle species enabled during transport.

        Parameters
        ----------
        transported_particles : list of {"neutron", "electron", "proton"}
            Particle species to enable. Species not listed are disabled.

        Examples
        --------
        Transport neutrons only:

        >>> import mcdc
        >>> simulation = mcdc.Simulation()
        >>> simulation.settings.set_transported_particles(["neutron"])

        Enable coupled neutron and electron transport:

        >>> simulation.settings.set_transported_particles(
        ...     ["neutron", "electron"],
        ... )
        """
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
