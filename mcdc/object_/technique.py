import numpy as np

from mcdc.constant import (
    INF,
    PI,
    PARTICLE_TYPE_BY_NAME,
    WW_GENERATOR_MAGIC,
)
from mcdc.object_.base import MCDCBase, MCDCPolymorphic
from mcdc.object_.mesh import MeshBase, MeshUniform
from mcdc.object_.tally import Tally, TallyTracklength 
from mcdc.print_ import print_error
from numpy.typing import NDArray
from typing import Annotated

# ======================================================================================
# Implicit capture
# ======================================================================================


class ImplicitCapture(MCDCBase):
    """Simulation-owned implicit-capture configuration."""

    # MC/DC framework metadata
    label = "implicit_capture"

    active: bool

    def __init__(self) -> None:
        self.active = False

    def __call__(self, active: bool = True) -> None:
        """Configure implicit capture.

        Parameters
        ----------
        active : bool, optional
            Whether implicit capture is enabled.

        Examples
        --------
        Enable implicit capture:

        >>> import mcdc
        >>> simulation = mcdc.Simulation()
        >>> simulation.technique.implicit_capture()

        Disable implicit capture:

        >>> simulation.technique.implicit_capture(active=False)
        """
        self.active = active


# ======================================================================================
# Weighted emission
# ======================================================================================


class WeightedEmission(MCDCBase):
    """Simulation-owned weighted-emission configuration."""

    # MC/DC framework metadata
    label = "weighted_emission"

    active: bool
    weight_target: float

    def __init__(self) -> None:
        self.active = False
        self.weight_target = 0.0

    def __call__(self, active: bool = True, weight_target: float = 1.0) -> None:
        """Configure weighted emission.

        Parameters
        ----------
        active : bool, optional
            Whether the technique is active.
        weight_target : float, optional
            Target statistical weight for emitted particles.

        Examples
        --------
        Enable weighted emission with unit target weight:

        >>> import mcdc
        >>> simulation = mcdc.Simulation()
        >>> simulation.technique.weighted_emission(weight_target=1.0)

        Select a different target weight:

        >>> simulation.technique.weighted_emission(weight_target=0.5)

        Disable weighted emission:

        >>> simulation.technique.weighted_emission(active=False)
        """
        self.active = active
        self.weight_target = weight_target


# ======================================================================================
# Weight roulette
# ======================================================================================


class GlobalWeightRoulette(MCDCBase):
    """Simulation-owned global weight-roulette configuration."""

    # MC/DC framework metadata
    label = "global_weight_roulette"

    active: bool
    weight_threshold: float
    weight_target: float

    def __init__(self) -> None:
        self.active = False
        self.weight_threshold = 0.0
        self.weight_target = 1.0

    def __call__(
        self, weight_threshold: float = 0.0, weight_target: float = 1.0
    ) -> None:
        """Enable roulette below a global weight threshold.

        Parameters
        ----------
        weight_threshold : float, optional
            Particle weight below which roulette is applied.
        weight_target : float, optional
            Statistical weight assigned to particles that survive roulette.
            Must be greater than or equal to ``weight_threshold``.

        Examples
        --------
        Apply roulette below a particle weight of 0.25 and raise surviving
        particles to unit weight:

        >>> import mcdc
        >>> simulation = mcdc.Simulation()
        >>> simulation.technique.global_weight_roulette(
        ...     weight_threshold=0.25,
        ...     weight_target=1.0,
        ... )

        Use a lower target weight:

        >>> simulation.technique.global_weight_roulette(
        ...     weight_threshold=0.1,
        ...     weight_target=0.5,
        ... )
        """
        if weight_threshold > weight_target:
            print_error(
                "For weight roulette, weight threshold has to be smaller than the target"
            )
        self.active = True
        self.weight_threshold = weight_threshold
        self.weight_target = weight_target


# ======================================================================================
# Weight Windows
# ======================================================================================


class WeightWindows(MCDCBase):
    """Simulation-owned particle weight-window configuration."""

    # MC/DC framework metadata
    label = "weight_windows"

    active: bool

    ptype: int

    # time
    time_bounds: NDArray[np.float64]
    Nt: int
    # energy
    energy_bounds: NDArray[np.float64]
    Ne: int
    # angle
    mu_bounds: NDArray[np.float64]
    Nmu: int
    azi_bounds: NDArray[np.float64]
    Na: int
    # space
    mesh: MeshBase
    Nx: int
    Ny: int
    Nz: int

    # reference vector for calculating mu and azimuthal angle
    polar_reference: Annotated[NDArray[np.float64], (3,)]

    # arrays of ww params
    N_WW_parameters: int = 3
    weights: Annotated[
        NDArray[np.float64],
        ("Nt", "Ne", "Nmu", "Na", "Nx", "Ny", "Nz", "N_WW_parameters"),
    ]

    def __init__(self, ptype) -> None:
        self.active = False
        self.ptype = ptype
        self.time_bounds = np.array([0.0, INF])
        self.Nt = 1
        self.azi_bounds = np.array([-PI, PI])
        self.Na = 1
        self.mu_bounds = np.array([-1.0, 1.0])
        self.Nmu = 1
        self.energy_bounds = np.array([-0.5, INF])
        self.Ne = 1
        self.mesh_ID = -1  # skirt around having to create a MeshBase instance
        self.Nx, self.Ny, self.Nz = 1, 1, 1
        self.Nt = 1
        self.polar_reference = np.array([0.0, 0.0, 1.0])
        shape = (
            self.Nt,
            self.Ne,
            self.Nmu,
            self.Na,
            self.Nx,
            self.Ny,
            self.Nz,
            self.N_WW_parameters,
        )
        self.weights = np.zeros(shape)

    def __call__(
        self,
        weight_windows: NDArray[np.float64],
        *,
        mesh: MeshBase | None = None,
        energy: NDArray[np.float64] | None = None,
        mu: NDArray[np.float64] | None = None,
        azimuthal: NDArray[np.float64] | None = None,
        time: NDArray[np.float64] | None = None,
    ) -> None:
        # fill in defaults
        if mesh is None:
            mesh = MeshUniform()
        if energy is None:
            # usable for both groups and max energy
            energy = np.array([-0.5, INF])
        if mu is None:
            mu = np.array([-1.0, 1.0])
        if azimuthal is None:
            azimuthal = np.array([-PI, PI])
        if time is None:
            time = np.array([0, INF])

        # get mesh size
        match mesh.label:
            case "uniform_mesh":
                nx, ny, nz = mesh.Nx, mesh.Ny, mesh.Nz
            case "structured_mesh":
                nx, ny, nz = (
                    mesh.x.shape[0] - 1,
                    mesh.y.shape[0] - 1,
                    mesh.z.shape[0] - 1,
                )
            case _:
                print_error(
                    f"{type(mesh).__name__} is not supported for weight windows"
                )
        # validate energy and get size
        self._validate_bin_boundaries(energy, "Energy")
        ne = energy.shape[0] - 1
        # validate mu and get size
        self._validate_bin_boundaries(mu, "Mu")
        nmu = mu.shape[0] - 1
        # validate azimuthal and get size
        self._validate_bin_boundaries(azimuthal, "Azimuthal")
        na = azimuthal.shape[0] - 1
        # validate time and get size
        self._validate_bin_boundaries(time, "Time")
        nt = time.shape[0] - 1

        # check correct shape
        expected_shape = (nt, ne, nmu, na, nx, ny, nz, self.N_WW_parameters)
        ww_shape = weight_windows.shape
        if ww_shape != expected_shape:
            try:
                weight_windows = weight_windows.reshape(expected_shape)
            except:
                print_error(
                    f"Could not reshape weight window array from {ww_shape} to {expected_shape}!"
                )

        self.active = True
        self.time_bounds = time
        self.Nt = nt
        self.energy_bounds = energy
        self.Ne = ne
        self.mu_bounds = mu
        self.Nmu = nmu
        self.azi_bounds = azimuthal
        self.Na = na
        self.mesh = mesh
        self.Nx, self.Ny, self.Nz = (nx, ny, nz)
        self.weights = weight_windows

        # check weight windows are valid
        if (self.weights[..., 0] <= 0.0).any():
            print_error(
                "Lower bound weights must be strictly positive to avoid invalid roulette behavior"
            )
        if (self.weights[..., 0] > self.weights[..., 1]).any():
            print_error(
                "Lower bound weight can not be greater than the target weight for any weight window"
            )
        if (self.weights[..., 1] > self.weights[..., 2]).any():
            print_error(
                "Target weight can not be greater than the upper bound weight for any weight window"
            )

    def set_polar_reference(self, px, py, pz):
        polar_reference = np.array([px, py, pz])
        self.polar_reference = polar_reference / np.linalg.norm(polar_reference)

    @staticmethod
    def _validate_bin_boundaries(array: NDArray[np.float64], name: str):
        if not (np.diff(array) > 0).all():
            print_error(f"{name} bounds must be strictly increasing")
        if len(array.shape) != 1:
            print_error(
                f"Invalid shape for {name} bounds; expected 1D got {len(array.shape)}D"
            )

    def _get_shape(self):
        return (self.Nt, self.Ne, self.Nmu, self.Na, self.Nx, self.Ny, self.Nz, self.N_WW_parameters)


# ======================================================================================
# Base Weight Window Generator
# ======================================================================================


class WeightWindowGenerator(MCDCPolymorphic):
    """Polymorphic base class for all weight window generators"""
    label = "weight_window_generator"

    sub_type = -1

    active: bool

    # type of particle this generator applies to
    ptype: int

    def __init__(self, ptype: int):
        self.active = False
        self.ptype = ptype

    def _compile_into_simulation(self, simulation) -> bool:
        # Already compiled?
        if not super()._compile_into_simulation(simulation):
            return False

    def _register_members(self, simulation):
        return

# ======================================================================================
# MAGIC Weight Window Generator
# ======================================================================================


class MAGICWeightWindowGenerator(WeightWindowGenerator):
    """Simulation-owned iterative weight window generator."""

    # MC/DC framework metadata
    label = "MAGIC_weight_window_generator"
    sub_type = WW_GENERATOR_MAGIC

    active: bool

    ptype: int

    # flux tally
    flux_tally: Tally 

    # scale from lower weight to target weight
    target_scale: float

    # scale from lower weight to upper weight
    upper_scale: float

    def __init__(self, ptype: int):
        super().__init__(ptype) 
        # skirt around actually creating a tally
        self.flux_tally_ID = -1
        self.target_scale = 1.5
        self.upper_scale = 2.0

    def __call__(
        self,
        *,
        weight_target_scale: float = 1.5,
        weight_upper_scale: float = 2.0,
        tally_name: str | None = None,
        mesh: MeshBase | None = None,
        energy: NDArray[np.float64] | None = None,
        mu: NDArray[np.float64] | None = None,
        azimuthal: NDArray[np.float64] | None = None,
        time: NDArray[np.float64] | None = None,
    ) -> None:
        if weight_target_scale <= 1:
            print_error(
                "Target weight scale must be greater than or equal to 1."
            )
        self.target_scale = weight_target_scale

        if weight_upper_scale <= self.target_scale:
            print_error(
                "Upper weight scale must be greater than or equal to target "
                f"weight scale: {self.target_scale}."
            )
        self.upper_scale = weight_upper_scale

        if tally_name is None:
            tally_name =f"_{self.ptype}_MAGIC_WWG_"
        self.flux_tally = TallyTracklength(
            name=tally_name,
            mesh=mesh,
            energy=energy,
            mu=mu,
            azi=azimuthal,
            time=time,
        )

    def _get_weight_window_shape(self):
        tally = self.flux_tally

        mesh = tally.mesh
        if mesh is None:
            nx, ny, nz = 1, 1, 1
        else:
            match mesh.label:
                case "uniform_mesh":
                    nx, ny, nz = mesh.Nx, mesh.Ny, mesh.Nz
                case "structured_mesh":
                    nx, ny, nz = (
                        mesh.x.shape[0] - 1,
                        mesh.y.shape[0] - 1,
                        mesh.z.shape[0] - 1,
                    )
                case _:
                    print_error(
                        f"{type(mesh).__name__} is not supported for weight windows"
                    )

        if tally.energy is None:
            ne = 1
        else:
            ne = tally.energy.shape[0] - 1

        if tally.mu is None:
            nmu = 1
        else:
            nmu = tally.mu.shape[0] - 1
        
        if tally.azi is None:
            na = 1
        else:
            na = tally.azi.shape[0] - 1

        if tally.time is None:
            nt = 1
        else:
            nt = tally.time.shape[0] - 1

        # get weight window shape
        ww_shape = (nt, ne, nmu, na, nx, ny, nz, WeightWindows.N_WW_parameters)
        return ww_shape


    def _get_weight_window_params(self):
        ww_shape = self._get_weight_window_shape() 

        initial_global_ww = 0.5 * np.array(1, self.target_scale, self.upper_scale)
        ww_array = np.tile(initial_global_ww, ww_shape)

        ww_params = {
            "weight_windows":ww_array,
            "mesh":self.flux_tally.mesh,
            "energy":self.flux_tally.energy,
            "mu":self.flux_tally.mu,
            "azimuthal":self.flux_tally.azi,
            "time":self.tally.time,
        }
        return ww_params

    def _compile_into_simulation(self, simulation) -> bool:
        # Already compiled?
        if not super()._compile_into_simulation(simulation):
            return False

        self.flux_tally_ID = self.flux_tally.ID

        return False 

    def _register_members(self, simulation):
        simulation.tallies.append(self.flux_tally)

# ======================================================================================
# Population control
# ======================================================================================


class PopulationControl(MCDCBase):
    """Simulation-owned source-bank population-control configuration."""

    # MC/DC framework metadata
    label = "population_control"

    active: bool

    def __init__(self) -> None:
        self.active = False

    def __call__(self, active: bool = True) -> None:
        """Configure source-bank population control.

        Parameters
        ----------
        active : bool, optional
            Whether source-bank population control is enabled.

        Examples
        --------
        Enable source-bank population control:

        >>> import mcdc
        >>> simulation = mcdc.Simulation()
        >>> simulation.technique.population_control()

        Disable source-bank population control:

        >>> simulation.technique.population_control(active=False)
        """
        self.active = active


# ======================================================================================
# Simulation technique collection
# ======================================================================================


class Technique(MCDCBase):
    """Own all simulation-wide transport-technique configurations.

    Access the individual configurations through
    ``simulation.technique``. The same hierarchy is retained in the packed
    runtime simulation.
    """

    # MC/DC framework metadata
    label = "technique"

    implicit_capture: ImplicitCapture
    weighted_emission: WeightedEmission
    global_weight_roulette: GlobalWeightRoulette
    neutron_weight_windows: WeightWindows
    neutron_weight_window_generator: WeightWindowGenerator
    electron_weight_windows: WeightWindows
    electron_weight_window_generator: WeightWindowGenerator
    proton_weight_windows: WeightWindows
    proton_weight_window_generator: WeightWindowGenerator
    population_control: PopulationControl

    def __init__(self) -> None:
        # Construct every simulation-wide technique configuration
        self.implicit_capture = ImplicitCapture()
        self.weighted_emission = WeightedEmission()
        self.global_weight_roulette = GlobalWeightRoulette()
        for pname, ptype in PARTICLE_TYPE_BY_NAME.items():
            setattr(self, f"{pname}_weight_windows", WeightWindows(ptype))
            setattr(self, f"{pname}_weight_window_generator", WeightWindowGenerator(ptype))
        self.population_control = PopulationControl()

    def _compile_into_simulation(self, simulation) -> bool:
        # Already compiled?
        if not super()._compile_into_simulation(simulation):
            return False

        for pname in PARTICLE_TYPE_BY_NAME.keys():
            wwg = getattr(self, f"{pname}_weight_window_generator") 
            wwg._compile_into_simulation(simulation)
        return False 

    def _register_into_simulation(self, simulation):
        for pname in PARTICLE_TYPE_BY_NAME.keys():
            wwg = getattr(self, f"{pname}_weight_window_generator") 
            wwg._register_members(simulation)

    def weight_windows(
        self,
        weight_windows: NDArray[np.float64],
        particle_type: str = "neutron",
        mesh: MeshBase | None = None,
        energy: NDArray[np.float64] | None = None,
        mu: NDArray[np.float64] | None = None,
        azimuthal: NDArray[np.float64] | None = None,
        time: NDArray[np.float64] | None = None,
    ):
        """Configure lower, target, and upper particle weights.

        Parameters
        ----------
        weight_windows : ndarray, shape (Ne, Nx, Ny, Nz, 3)
            Lower, target, and upper weights in the final dimension. Every
            lower weight must be positive, and each window must satisfy
            ``lower <= target <= upper``.
        particle_type : str, optional
            Particle type the specified weight windows apply to. The default
            is neutron.
        mesh : MeshUniform or MeshStructured, optional
            Spatial mesh. The default is one unbounded uniform bin.
        energy : ndarray, optional
            Strictly increasing energy boundaries in eV. The default is one
            all-energy bin.
        mu : ndarray, optional
            Strictly increasing polar cosine bounds from -1 to 1. The default
            is one bin from -1 to 1.
        azimuthal : ndarray, optional
            Strictly increasing azimuthal angle bounds from -pi to pi. The
            default is one bin from -pi to pi.

        Examples
        --------
        Apply one weight window over all space and energy:

        >>> import numpy as np
        >>> import mcdc
        >>> simulation = mcdc.Simulation()
        >>> windows = np.array([0.5, 1.0, 2.0])
        >>> simulation.technique.weight_windows(windows)

        Configure weight windows on a uniform spatial mesh:

        >>> mesh = mcdc.MeshUniform(x=(-5.0, 1.0, 10))
        >>> windows = np.tile([0.25, 0.5, 1.0], (10,))
        >>> simulation.technique.weight_windows(windows, mesh=mesh)

        Configure both energy- and space-dependent windows:

        >>> energy = np.array([0.0, 0.625, 20.0e6])
        >>> windows = np.tile([0.25, 0.5, 1.0], (2, 10))
        >>> simulation.technique.weight_windows(
        ...     windows,
        ...     mesh=mesh,
        ...     energy=energy,
        ... )

        Configure full phase-dependent windows
        >>> time = np.array([0.0, 10, 100, 1000])
        >>> mu = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
        >>> azi = np.array([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
        >>> windows = np.tile([0.25, 0.5, 1.0], (3, 2, 4, 4, 10))
        >>> simulation.technique.weight_windows(
        ...     windows,
        ...     mesh=mesh,
        ...     energy=energy,
        ...     mu=mu,
        ...     azimuthal=azi,
        ...     time=time,
        ... )

        Configure weight windows for specific particle types
        >>> simulation.technique.weight_windows(
        ...     windows,
        ...     particle_type="electron",
        ...     mesh=mesh,
        ... )
        """
        ww_name = f"{particle_type}_weight_windows"
        ww_obj = getattr(self, ww_name)
        if ww_obj.active:
            print_error(
                f"Attempting to overwrite weight windows for particle type {particle_type}!"
            )
        ww_obj(
            weight_windows,
            mesh=mesh,
            energy=energy,
            mu=mu,
            azimuthal=azimuthal,
            time=time,
        )

    def MAGIC_weight_window_generator(
        self,
        particle_type: str = "neutron",
        *,
        weight_target_scale: float = 1.5,
        weight_upper_scale: float = 2.0,
        tally_name: str | None = None,
        mesh: MeshBase | None = None,
        energy: NDArray[np.float64] | None = None,
        mu: NDArray[np.float64] | None = None,
        azimuthal: NDArray[np.float64] | None = None,
        time: NDArray[np.float64] | None = None,
    ) -> None:
        wwg_name = f"{particle_type}_weight_window_generator"

        if getattr(self, wwg_name).active:
            print_error(
                f"Attempting to overwrite weight window generator for particle type {particle_type}!"
            )
        setattr(self, wwg_name, MAGICWeightWindowGenerator(PARTICLE_TYPE_BY_NAME[particle_type]))
        wwg_obj = getattr(self, wwg_name)
        wwg_obj(
            weight_target_scale=weight_target_scale,
            weight_upper_scale=weight_upper_scale,
            tally_name=tally_name,
            mesh=mesh,
            energy=energy,
            mu=mu,
            azimuthal=azimuthal,
            time=time,
        )

        ww_name = f"{particle_type}_weight_windows"
        ww_obj = getattr(self, ww_name)
        if ww_obj.active:
            #make sure its the same dimensions
            expected_shape = wwg_obj._get_weight_window_shape()
            true_shape = ww_obj._get_shape()
            if expected_shape != true_shape:
                print_error(
                    "Pre-specified weight windows for particle type "
                    f"{particle_type} has shape {true_shape}, but expected "
                    f"{expected_shape} from MAGIC specificiations!"
                )
        else:
            ww_obj(
                **wwg_obj._get_weight_window_params()
            )

