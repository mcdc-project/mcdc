import math

# ======================================================================================
# Mathematical and numerical constants
# ======================================================================================

PI = math.pi
PI_HALF = PI / 2.0
PI_SQRT = math.sqrt(PI)

TINY = 1e-10
INF = 1e10


# ======================================================================================
# Physics and nuclear data
# ======================================================================================

# Physical constants
LIGHT_SPEED = 2.99792458e10  # cm/s
NEUTRON_MASS = 939.565413e6  # eV/c^2
ELECTRON_MASS = 510.99895069e3  # eV/c^2
BOLTZMANN_K = 8.61733326e-5  # eV/K

# Physics thresholds
ELECTRON_CUTOFF_ENERGY = 100  # eV
MU_CUTOFF = 0.999999
THERMAL_THRESHOLD_FACTOR = 400

# Particle types
PARTICLE_NEUTRON = 0
PARTICLE_ELECTRON = 1
PARTICLE_PROTON = 2
PARTICLE_ANY = 100

# Neutron multigroup energy representation
NEUTRON_MULTIGROUP_ENERGY_MIDPOINT = 0
NEUTRON_MULTIGROUP_ENERGY_MIDPOINT_LOG = 1
NEUTRON_MULTIGROUP_ENERGY_UNIFORM = 2
NEUTRON_MULTIGROUP_ENERGY_UNIFORM_LOG = 3

# Neutron reactions
NEUTRON_REACTION_TOTAL = 0
NEUTRON_REACTION_ELASTIC_SCATTERING = 1
NEUTRON_REACTION_CAPTURE = 2
NEUTRON_REACTION_INELASTIC_SCATTERING = 3
NEUTRON_REACTION_FISSION = 4
NEUTRON_REACTION_FISSION_PROMPT = 5
NEUTRON_REACTION_FISSION_DELAYED = 6

# Electron reactions
ELECTRON_REACTION_TOTAL = 100
ELECTRON_REACTION_ELASTIC_SCATTERING = 101
ELECTRON_REACTION_IONIZATION = 102
ELECTRON_REACTION_BREMSSTRAHLUNG = 103
ELECTRON_REACTION_EXCITATION = 104

# Data representations
DATA_NONE = 0
DATA_TABLE = 1
DATA_POLYNOMIAL = 2

# Interpolation laws
INTERPOLATION_HISTOGRAM = 1
INTERPOLATION_LINEAR = 2
INTERPOLATION_SEMILOGX = 3
INTERPOLATION_SEMILOGY = 4
INTERPOLATION_LOG = 5

# Probability distributions
DISTRIBUTION_NONE = 0
DISTRIBUTION_PMF = 1
DISTRIBUTION_TABULATED = 2
DISTRIBUTION_MULTITABLE = 3
DISTRIBUTION_LEVEL_SCATTERING = 4
DISTRIBUTION_EVAPORATION = 5
DISTRIBUTION_MAXWELLIAN = 6
DISTRIBUTION_KALBACH_MANN = 7
DISTRIBUTION_TABULATED_ENERGY_ANGLE = 8
DISTRIBUTION_N_BODY = 9

# Angular distributions
ANGLE_ISOTROPIC = 0
ANGLE_DISTRIBUTED = 1
ANGLE_ENERGY_CORRELATED = 2

# Reference frames
REFERENCE_FRAME_LAB = 0
REFERENCE_FRAME_COM = 1


# ======================================================================================
# Geometry
# ======================================================================================

# Axes
AXIS_X = 0
AXIS_Y = 1
AXIS_Z = 2
AXIS_T = 3

# Mesh types
MESH_UNIFORM = 0
MESH_STRUCTURED = 1

# Boundary conditions
BC_NONE = 0
BC_VACUUM = 1
BC_REFLECTIVE = 2

# Cell fills
FILL_MATERIAL = 0
FILL_UNIVERSE = 1
FILL_LATTICE = 2
FILL_NONE = 3

# Universe
UNIVERSE_ROOT = 0

# Surface types
SURFACE_PLANE_X = 1
SURFACE_PLANE_Y = 2
SURFACE_PLANE_Z = 3
SURFACE_PLANE = 4
SURFACE_CYLINDER_X = 5
SURFACE_CYLINDER_Y = 6
SURFACE_CYLINDER_Z = 7
SURFACE_CYLINDER = 8
SURFACE_SPHERE = 9
SURFACE_QUADRIC = 10
SURFACE_CONE_X = 11
SURFACE_CONE_Y = 12
SURFACE_CONE_Z = 13
SURFACE_TORUS_X = 14
SURFACE_TORUS_Y = 15
SURFACE_TORUS_Z = 16
SURFACE_TORUS = 17

# Boolean operators
BOOL_AND = -1
BOOL_OR = -2
BOOL_NOT = -3


# ======================================================================================
# Transport
# ======================================================================================

# Events are bit flags and may be combined with bitwise operations.
EVENT_NONE = 1 << 0
EVENT_SURFACE_CROSSING = 1 << 1
EVENT_LATTICE_CROSSING = 1 << 2
EVENT_LOST = 1 << 3
EVENT_COLLISION = 1 << 4
EVENT_TIME_CENSUS = 1 << 5
EVENT_TIME_BOUNDARY = 1 << 6

# Coincidence tolerances
COINCIDENCE_TOLERANCE = TINY
COINCIDENCE_TOLERANCE_DIRECTION = 1e-5
COINCIDENCE_TOLERANCE_ENERGY = 1e-5
COINCIDENCE_TOLERANCE_TIME = TINY * 1e-2


# ======================================================================================
# Tallies
# ======================================================================================

# Tally estimator types
TALLY_SURFACE_CROSSING = 0
TALLY_COLLISION = 1
TALLY_TRACKLENGTH = 2

# Track-length scores
SCORE_FLUX = 0
SCORE_DENSITY = 1
SCORE_COLLISION = 2
SCORE_CAPTURE = 3
SCORE_FISSION = 4

# Surface-crossing scores
SCORE_CURRENT_NET = 100
SCORE_CURRENT_IN = 101
SCORE_CURRENT_OUT = 102

# Collision scores
SCORE_ENERGY_DEPOSITION = 200

# Supported scores by estimator type
SUPPORTED_SCORES_SURFACE_CROSSING = {"current-net", "current-in", "current-out"}
SUPPORTED_SCORES_TRACKLENGTH = {"flux", "density", "collision", "capture", "fission"}
SUPPORTED_SCORES_COLLISION = {"energy_deposition"}
SUPPORTED_SCORES = (
    SUPPORTED_SCORES_SURFACE_CROSSING
    | SUPPORTED_SCORES_TRACKLENGTH
    | SUPPORTED_SCORES_COLLISION
)


# ======================================================================================
# Techniques and diagnostics
# ======================================================================================

# Gyration-radius types
GYRATION_RADIUS_ALL = 0
GYRATION_RADIUS_INFINITE_X = 1
GYRATION_RADIUS_INFINITE_Y = 2
GYRATION_RADIUS_INFINITE_Z = 3
GYRATION_RADIUS_ONLY_X = 4
GYRATION_RADIUS_ONLY_Y = 5
GYRATION_RADIUS_ONLY_Z = 6

# Population-control types are currently unused.
PCT_NONE = 0
PCT_COMBING = 1
PCT_COMBING_WEIGHT = 2
PCT_SPLITTING_ROULETTE = 3
PCT_SPLITTING_ROULETTE_WEIGHT = 4

# Weight-window methods are currently unused.
WW_USER = 0
WW_PREVIOUS = 1

# Weight-window modifications are currently unused.
WW_MIN = 0
WW_WOLLABER = 1

# ======================================================================================
# Target Information
# ======================================================================================

CPU = 0
GPU = 1

# ======================================================================================
# GPU settings
# ======================================================================================

# GPU strategies
GPU_STRATEGY_SIMPLE_ASYNC = 0
GPU_STRATEGY_ASYNC = 0
GPU_STRATEGY_EVENT = 1

# GPU asynchronous-operation types
GPU_ASYNC_SIMPLE = 0

# GPU storage types
GPU_STORAGE_SEPARATE = 0
GPU_STORAGE_MANAGED = 1
GPU_STORAGE_UNITED = 2
