from __future__ import annotations
from typing import TYPE_CHECKING, Annotated

from mcdc.object_.technique import (
    ImplicitCapture,
    PopulationControl,
    GlobalWeightRoulette,
    WeightWindows,
    WeightedEmission,
)

if TYPE_CHECKING:
    from mcdc.object_.cell import Cell, Region
    from mcdc.object_.element import Element
    from mcdc.object_.electron_reaction import ElectronReactionBase
    from mcdc.object_.material import MaterialBase
    from mcdc.object_.nuclide import Nuclide
    from mcdc.object_.neutron_reaction import NeutronReactionBase
    from mcdc.object_.proton_reaction import ProtonReactionBase
    from mcdc.object_.source import Source
    from mcdc.object_.surface import Surface
    from mcdc.object_.tally import Tally

####

import numpy as np

from mpi4py import MPI
from numpy import float64, int64
from numpy.typing import NDArray

####

from mcdc.object_.base import ObjectSingleton
from mcdc.object_.data import DataBase, DataNone
from mcdc.object_.distribution import DistributionBase, DistributionNone
from mcdc.object_.gpu_tools import GPUMeta
from mcdc.object_.mesh import MeshBase, MeshUniform
from mcdc.object_.particle import ParticleBank
from mcdc.object_.settings import Settings
from mcdc.object_.universe import Universe, Lattice

# ======================================================================================
# Simulation
# ======================================================================================


class Simulation(ObjectSingleton):
    # Annotations for Numba mode
    label: str = "simulation"
    non_numba: list[str] = [
        "regions",
        "bank_active",
        "bank_census",
        "bank_source",
        "bank_future",
    ]

    # Physics
    data: list[DataBase]
    distributions: list[DistributionBase]
    materials: list[MaterialBase]
    elements: list[Element]
    electron_reactions: list[ElectronReactionBase]
    nuclides: list[Nuclide]
    neutron_reactions: list[NeutronReactionBase]
    sources: list[Source]
    proton_reactions: list[ProtonReactionBase]

    # Geometry
    cells: list[Cell]
    lattices: list[Lattice]
    regions: list[Region]
    surfaces: list[Surface]
    universes: list[Universe]
    meshes: list[MeshBase]

    # Tallies
    tallies: list[Tally]

    # Settings
    settings: Settings

    # Techniques
    implicit_capture: ImplicitCapture
    weighted_emission: WeightedEmission
    global_weight_roulette: GlobalWeightRoulette
    weight_windows: WeightWindows
    population_control: PopulationControl

    # Particle banks
    bank_active: ParticleBank
    bank_census: ParticleBank
    bank_source: ParticleBank
    bank_future: ParticleBank

    # Simulation parameters
    idx_work: int
    idx_cycle: int
    idx_census: int
    idx_batch: int
    dd_idx: int
    dd_N_local_source: int
    dd_local_rank: int
    k_eff: float
    k_cycle: NDArray[float64]
    k_avg: float
    k_sdv: float
    n_avg: float
    n_sdv: float
    n_max: float
    C_avg: float
    C_sdv: float
    C_max: float
    k_avg_running: float
    k_sdv_running: float
    gyration_radius: NDArray[float64]
    cycle_active: bool
    eigenvalue_tally_nuSigmaF: Annotated[NDArray[float64], (1,)]
    eigenvalue_tally_n: Annotated[NDArray[float64], (1,)]
    eigenvalue_tally_C: Annotated[NDArray[float64], (1,)]
    mpi_size: int
    mpi_rank: int
    mpi_master: bool
    mpi_work_start: int
    mpi_work_size: int
    mpi_work_size_total: int
    mpi_work_iter: Annotated[NDArray[int64], (1,)]
    runtime_total: float
    runtime_preparation: float
    runtime_simulation: float
    runtime_output: float
    runtime_bank_management: float

    # GPU metadata
    gpu_meta: GPUMeta
    source_seed: int

    def __init__(self):
        super().__init__()

        # ==============================================================================
        # Simulation objects
        # ==============================================================================

        # Physics
        self.data = [DataNone()]
        self.distributions = [DistributionNone()]
        self.materials = []
        self.elements = []
        self.electron_reactions = []
        self.nuclides = []
        self.neutron_reactions = []
        self.sources = []
        self.proton_reactions = []

        # Geometry
        self.cells = []
        self.lattices = []
        self.regions = []
        self.surfaces = []
        self.universes = [Universe("Root Universe", root=True)]
        self.meshes = []

        # Tallies
        self.tallies = []

        # Settings
        self.settings = Settings()

        # Techniques
        self.implicit_capture = ImplicitCapture()
        self.weighted_emission = WeightedEmission()
        self.global_weight_roulette = GlobalWeightRoulette()
        self.weight_windows = WeightWindows()
        self.population_control = PopulationControl()

        # ==============================================================================
        # Particle banks
        # ==============================================================================

        self.bank_active = ParticleBank(tag="active")
        self.bank_census = ParticleBank(tag="census")
        self.bank_source = ParticleBank(tag="source")
        self.bank_future = ParticleBank(tag="future")

        # ==============================================================================
        # Simulation parameters
        # ==============================================================================

        # Simulation indices
        self.idx_work = 0
        self.idx_cycle = 0
        self.idx_census = 0
        self.idx_batch = 0

        # Domain decomposition
        self.dd_idx = 0
        self.dd_N_local_source = 0
        self.dd_local_rank = 0

        # Eigenvalue simulation
        self.k_eff = 0.0
        self.k_cycle = np.ones(1)
        self.k_avg = 0.0
        self.k_sdv = 0.0
        self.n_avg = 0.0  # Neutron density
        self.n_sdv = 0.0
        self.n_max = 0.0
        self.C_avg = 0.0  # Precursor density
        self.C_sdv = 0.0
        self.C_max = 0.0
        self.k_avg_running = 0.0
        self.k_sdv_running = 0.0
        self.gyration_radius = np.zeros(1)
        self.cycle_active = False
        self.eigenvalue_tally_nuSigmaF = np.zeros(1)
        self.eigenvalue_tally_n = np.zeros(1)
        self.eigenvalue_tally_C = np.zeros(1)

        # MPI parameters
        self.mpi_size = MPI.COMM_WORLD.Get_size()
        self.mpi_rank = MPI.COMM_WORLD.Get_rank()
        self.mpi_master = self.mpi_rank == 0
        self.mpi_work_start = 0
        self.mpi_work_size = 0
        self.mpi_work_size_total = 0
        self.mpi_work_iter = np.zeros(1, dtype=int64)

        # Runtime records
        self.runtime_total = 0.0
        self.runtime_preparation = 0.0
        self.runtime_simulation = 0.0
        self.runtime_output = 0.0
        self.runtime_bank_management = 0.0

        # GPU metadata
        self.gpu_meta = GPUMeta()
        self.source_seed = 0

    def set_root_universe(self, cells=[]):
        self.universes[0].cells = cells


simulation = Simulation()
