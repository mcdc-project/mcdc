CONCEPT_LESSONS = {
    "material": {
        "concept": (
            "**Materials define the physics of particle interaction.**\n\n"
            "MCDC uses two modes:\n"
            "1. **Multi-Group (MG):** (`mcdc.MaterialMG`) **Recommended for beginners.** "
            "You define macroscopic cross-sections manually. This offers full control over the physics "
            "(scattering matrices, delayed neutrons, spectrums) without needing external data files.\n"
            "2. **Continuous Energy (CE):** (`mcdc.Material`) **Advanced.** "
            "Uses external nuclear data libraries (`MCDC_XSLIB`). You provide isotopic compositions, "
            "and MCDC handles the physics lookup. It is much more computationally intensive."
        ),
        "syntax": """# 1. Multi-Group
# Fuel example
fuel = mcdc.MaterialMG(
    capture=np.array([0.45]),    # Higher capture prevents runaway
    fission=np.array([0.55]),    # Fission [/cm]
    nu_p=np.array([2.5])         # REQUIRED for fission materials
)

# Absorber example
absorber = mcdc.MaterialMG(
    capture=np.array([1.0])
)

# 2. Continuous Energy (Requires Library)
water = mcdc.Material(
    nuclide_composition={'H1': 0.0668, 'O16': 0.0334}
)""",
        "parts": (
            "** Multi-Group Parameters (MaterialMG) **\n"
            "**Understanding G and D:**\n"
            "* **G (Energy Groups):** Instead of exact energy, we sort particles into discrete 'buckets'. "
            "Group 0 is usually the **fastest** (highest energy), and the last group is the **slowest** (thermal).\n"
            "* **D (Delayed Precursor Groups):** Most neutrons are born instantly (Prompt). Some are born later "
            "from decaying fission products (Precursors). We group these precursors by their decay time 'families'.\n\n"
            "**The Parameters:**\n"
            "\n**G**=Energy Groups and **D**=Delayed Precursor Groups:\n\n"
            "| Param | Shape | Unit | Description |\n"
            "| :--- | :--- | :--- | :--- |\n"
            "| `capture` | 1D `[G]` | $/cm$ | Macroscopic capture cross-section. |\n"
            "| `scatter` | 2D `[G,G]`| $/cm$ | Differential scatter `[g_out, g_in]`. |\n"
            "| `fission` | 1D `[G]` | $/cm$ | Macroscopic fission cross-section. |\n"
            "| `nu_s` | 1D `[G]` | - | Scattering multiplication (optional). |\n"
            "| `nu_p` | 1D `[G]` | - | Prompt fission neutron yield. |\n"
            "| `nu_d` | 2D `[D,G]`| - | Delayed precursor yield `[dg, g_in]`. |\n"
            "| `chi_p` | 2D `[G,G]`| - | Prompt fission spectrum `[g_out, g_in]`. |\n"
            "| `chi_d` | 2D `[G,D]`| - | Delayed neutron spectrum `[g_out, dg]`. |\n"
            "| `speed` | 1D `[G]` | $cm/s$ | Average particle speed. |\n"
            "| `decay_rate`| 1D `[D]` | $/s$ | Precursor group decay constant. |"
        ),
        "tips": [
            "**Shape Matters:** `scatter` is `[g_out, g_in]`. This means `scatter[0][1]` is scattering **FROM** group 1 **TO** group 0.",
            "**Supercriticality:** Ensure `capture` is higher than `fission` to prevent supercriticality and simulation issues.",
            "**CE Mode:** Requires environment variable `MCDC_XSLIB` pointing to your nuclear data directory.",
        ],
    },
    "surface": {
        "concept": (
            "**Surfaces are used to define shapes and geometry.**\n\n"
            "They are geometric boundaries that divide space into two half-spaces: "
            "**Positive (+)** and **Negative (-)**. \n\n"
            "**Crucial Concept:** With the exception of Spheres, MCDC surfaces are **infinite**.\n"
            "* A `PlaneX` is an infinite wall extending forever in Y and Z.\n"
            "* A `CylinderZ` is an infinite tube extending forever in Z.\n"
            "To create a finite shape (like a fuel pellet), you must logically cut these infinite surfaces "
            "later using Cells."
        ),
        "syntax": """# 1. Axis-Aligned Planes (Simple)
# Wall at x = -10.0, everything to the left is 'inside' (negative)
left_wall = mcdc.Surface.PlaneX(x=-10.0, boundary_condition="vacuum")

# 2. Infinite Cylinder (Tube)
# Infinite along Z-axis, radius 10
fuel_radius = mcdc.Surface.CylinderZ(center=[0.0, 0.0], radius=10.0)

# 3. Generic Plane (Angled)
# Ax + By + Cz + D = 0
# Example: A 45-degree cut
slant = mcdc.Surface.Plane(A=1.0, B=1.0, C=0.0, D=0.0)""",
        "parts": (
            "| Factory Method | Params | Description |\n"
            "| :--- | :--- | :--- |\n"
            "| `PlaneX` | `x` | Plane perpendicular to X-axis. |\n"
            "| `PlaneY` | `y` | Plane perpendicular to Y-axis. |\n"
            "| `PlaneZ` | `z` | Plane perpendicular to Z-axis. |\n"
            "| `Plane` | `A,B,C,D` | Generic Plane ($Ax+By+Cz+D=0$). |\n"
            "| `CylinderZ` | `center` [x,y], `radius` | Infinite tube along Z-axis. |\n"
            "| `Sphere` | `center` [x,y,z], `radius` | A sphere. |\n\n"
            "**Boundary Conditions:**\n"
            "| Value | Effect |\n"
            "| :--- | :--- |\n"
            '| `"interface"` | (Default) Particle passes through. |\n'
            '| `"vacuum"` | Particle is killed immediately. |\n'
            '| `"reflective"` | Particle bounces back (mirror). |'
        ),
        "tips": [
            "**The 'Viz' Trick:** Since surfaces are invisible logic, use the `viz` command to see wireframes of your defined surfaces.",
            "**Finite Shapes:** To make a finite cylinder of height 10cm, you need **three** surfaces: One `CylinderZ` (the sides) and two `PlaneZ` surfaces (top and bottom caps).",
            "**Signs:** For Cylinders and Spheres, the 'inside' is usually the **Negative (-)** side, and the 'outside' is the **Positive (+)** side.",
        ],
    },
    "cell": {
        "concept": (
            "**Cells define physical volumes by combining Surfaces.**\n\n"
            "A Cell requires two things:\n"
            "1. **Region:** A Boolean combination of surfaces (`&` AND, `|` OR, `~` NOT). "
            "It is best practice to define complex regions as variables *before* creating the cell.\n"
            "2. **Fill:** The content of the cell (a `Material` object, `Universe`, `Lattice`, or `None` for void).\n\n"
            "\n"
            "**The Viz Tool:** Boolean logic is tricky. Use the `viz` command (e.g., `viz z=0`) "
            "immediately after creating a cell to verify your shape."
        ),
        "syntax": """# 1. Define Regions (Logic)
# Sphere: Inside (-) sphere
inside_sphere = -sphere_surf

# Box: Intersection (&) of 6 half-spaces
# (Right of left walls, Left of right walls)
box = +x1 & -x2 & +y1 & -y2 & +z1 & -z2

# Channels: Union (|) of shapes
# Use parentheses to group logic if needed
channel_region = channel_1 | channel_2 | channel_3

# 2. Define Cells
# Simple: Fill sphere with fuel
c_fuel = mcdc.Cell(region=inside_sphere, fill=fuel_mat)

# Complex: The Box MINUS (~) the Sphere
# "Inside the box AND NOT inside the sphere"
c_moderator = mcdc.Cell(
    region = box & ~inside_sphere, 
    fill = water_mat
)""",
        "parts": (
            "| Param | Type | Description |\n"
            "| :--- | :--- | :--- |\n"
            "| `region` | `logic` | Boolean combination of surfaces. |\n"
            "| `fill` | `Material` | The material object filling the region. |\n"
            "| `translation`| `[x,y,z]`| Optional shift of the fill content. |\n"
            "| `rotation` | `[x,y,z]`| Optional rotation of the fill content. |\n\n"
            "**Boolean Operators:**\n"
            "| Sym | Logic | Meaning |\n"
            "| :--- | :--- | :--- |\n"
            "| `&` | **AND** | Intersection (Must be inside A *and* B). |\n"
            "| `|` | **OR** | Union (Inside A *or* B). |\n"
            "| `~` | **NOT** | Complement (Everything *outside* A). |"
        ),
        "tips": [
            "**Surface Orientation:**\n"
            " * **Planes:** `+` is the direction of the normal (usually Right/Up). `-` is opposite.\n"
            " * **Quadrics (Sphere/Cyl):** `-` is **INSIDE**. `+` is **OUTSIDE**.",
            "**Readability:** Define your region logic as variables (e.g., `core_region = ...`) before passing it to `mcdc.Cell`. It makes debugging much easier.",
            "**Void:** Use `fill=None` to create a streaming void (vacuum) region.",
        ],
    },
    "hierarchy": {
        "concept": (
            "**Hierarchies let you build complex, repeating structures efficiently.**\n\n"
            "Instead of defining 10,000 unique fuel pins, you define **one** pin and stamp it 10,000 times.\n"
            "There are three levels to this system:\n"
            "1. **Universe (The Stamp):** A collection of cells that acts as a blueprint. It has no boundaries itself.\n"
            "2. **Lattice (The Grid):** A structured map (checkerboard) where you place Universes into grid slots.\n"
            "3. **Fill (The Placement):** You place a Universe or Lattice inside a physical Cell to make it real.\n\n"
        ),
        "syntax": """# STEP 1: Create the "Stamp" (Universe)
# First, define the cells that make up a single pin
c_fuel = mcdc.Cell(region=-pin_radius, fill=uo2)
c_mod  = mcdc.Cell(region=+pin_radius, fill=water)

# Group them into a Universe
u_pin = mcdc.Universe(cells=[c_fuel, c_mod])

# STEP 2: Create the "Grid" (Lattice)
# Define a 3x3 grid using [Start, Pitch, Count] format
# This maps indices (i,j) to universes
lat = mcdc.Lattice(
    x=[-1.5, 1.0, 3],  # Start at -1.5, Width 1.0, 3 Steps
    y=[-1.5, 1.0, 3],
    universes=[
        [u_pin, u_pin, u_pin],
        [u_pin, u_void, u_pin],
        [u_pin, u_pin, u_pin]
    ]
)

# STEP 3: Place it in the world
# The lattice is infinite until you put it in a cell
main_cell = mcdc.Cell(region=core_region, fill=lat)""",
        "parts": (
            "**1. Universe `mcdc.Universe`**\n"
            "| Param | Type | Description |\n"
            "| :--- | :--- | :--- |\n"
            "| `cells` | `list` | List of Cell objects belonging to this universe. |\n"
            "| `root` | `bool` | Set `True` if this is the top-level universe. |\n\n"
            "**2. Lattice `mcdc.Lattice`**\n"
            "| Param | Type | Description |\n"
            "| :--- | :--- | :--- |\n"
            "| `x, y, z` | `list` | Grid definition: `[Start, Pitch, Count]`. |\n"
            "| `universes` | `list` | Nested list (2D or 3D) mapping grid layout. |\n\n"
            "**Grid Logic:**\n"
            "* `universes[0][0]` is the bottom-left corner (usually).\n"
            "* The dimensions of the `universes` list must match the `Count` defined in x/y/z."
        ),
        "tips": [
            "**Mental Model:** A Universe is like a PNG image file. It doesn't appear on the screen (the simulation) until you place it inside an `<img>` tag (a Cell).",
            "**The 'Mod' Trick:** When defining a lattice of fuel pins, usually the 'background' of the pin universe is the moderator (water). This ensures that when you tile them, the water connects seamlessly.",
            "**Lattice Indices:** If you are defining a 2D lattice (X-Y), the `universes` array is indexed `[y_index][x_index]`. Be careful with row/column ordering!",
        ],
    },
    "source": {
        "concept": (
            "**Sources define the 'birth' of particles.**\n\n"
            "Every particle needs an initial **Position**, **Direction**, **Energy**, and **Time**.\n"
            "You can define these as fixed values (Point Source) or intervals (Volume Source).\n\n"
            "**Multiple Sources:** You can define as many sources as you want! MCDC will randomly pick "
            "one for each particle based on the `probability` parameter. "
            "(e.g., Source A `probability=1`, Source B `probability=3` means particles are 75% likely to be born in B)."
        ),
        "syntax": """# 1. Point Source (Simplest)
# Isotropic (random direction) source at the origin
src_point = mcdc.Source(
    position=[0.0, 0.0, 0.0],
    isotropic=True,
    energy_group=0  # For MG mode (Group index 0)
)

# 2. Volume Source (Uniform Box)
# Samples uniformly x=[-1,1], y=[-1,1], z=[-5,5]
src_box = mcdc.Source(
    x=[-1.0, 1.0],
    y=[-1.0, 1.0],
    z=[-5.0, 5.0],
    isotropic=True,
    energy_group=1
)

# 3. Mixing Sources (Complex)
# 90% chance to be Background, 10% chance to be a Beam
src_bkg  = mcdc.Source(x=[-10,10], probability=0.9)
src_beam = mcdc.Source(position=[0,0,0], direction=[1,0,0], probability=0.1)""",
        "parts": (
            "| Param | Type | Description |\n"
            "| :--- | :--- | :--- |\n"
            "| `position` | `[x,y,z]` | Fixed birth point. |\n"
            "| `x, y, z` | `[min, max]` | Range for uniform spatial sampling. |\n"
            "| `isotropic`| `bool` | `True` = Random direction (4π). |\n"
            "| `direction`| `[u,v,w]` | Fixed direction vector. |\n"
            "| `energy` | `float` | Particle energy (eV) for **CE** mode. |\n"
            "| `energy_group`| `int` | Energy group index for **MG** mode. |\n"
            "| `time` | `float` or `[t1, t2]` | Birth time (snapshot or interval). |\n"
            "| `probability` | `float` | Relative weight for multi-source sampling. |"
        ),
        "tips": [
            "**The 'Lost Particle' Risk:** If a source spawns a particle inside a Void (or outside your geometry), it dies immediately. Make sure your source volume (`x,y,z`) is covered by a Material-filled Cell.",
            "**Defaults:** If you omit `x/y/z` or `position`, it defaults to 0.0. If you omit `direction`, you **must** set `isotropic=True`.",
            "**Spectrum:** You can pass a distribution to `energy_group` if you want a source that spans multiple groups (e.g., fission spectrum).",
        ],
    },
    "tally": {
        "concept": (
            "**Tallies are your sensors.**\n\n"
            "They record physical quantities during the simulation. You can place them:\n"
            "1. **Globally:** Measure total values for the whole system (`TallyGlobal`).\n"
            "2. **On Surfaces:** Measure particles crossing a boundary (`TallySurface`).\n"
            "3. **In Cells:** Measure reaction rates inside a specific volume (`TallyCell`).\n"
            "4. **On a Mesh:** Create a 2D/3D grid to map flux or fission power (`TallyMesh`).\n\n"
            "**Meshes:** To use a Mesh Tally, you must first define a `MeshUniform` (simple grid) "
            "or `MeshStructured` (custom grid)."
        ),
        "syntax": """# STEP 1: Define a Mesh (Optional, for TallyMesh)
# Option A: Uniform Grid (Start, Stop, Number of Intervals)
# Note: Use TUPLES (min, max, N)
m_reg = mcdc.MeshUniform(x=(-5.0, 5.0, 100), y=(-5.0, 5.0, 100))

# Option B: Structured Grid (Custom fence posts)
# Use Arrays for specific boundaries
m_custom = mcdc.MeshStructured(z=np.linspace(0, 10, 21))

# STEP 2: Create Tallies
# 1. Mesh Tally (Visual map of flux)
mcdc.TallyMesh(
    mesh=m_reg, 
    scores=['flux', 'fission'],
    name="plot_data"
)

# 2. Surface Tally (Leakage current)
mcdc.TallySurface(
    surface=right_wall, 
    scores=['net-current'],
    energy=np.logspace(-5, 7, 10) # Bin by energy
)

# 3. Cell Tally (Average flux in fuel)
mcdc.TallyCell(cell=c_fuel, scores=['flux'])""",
        "parts": (
            "**1. Mesh Definitions**\n"
            "| Class | Params | Description |\n"
            "| :--- | :--- | :--- |\n"
            "| `MeshUniform` | `x,y,z` | Tuples: `(start, stop, N_intervals)`. |\n"
            "| `MeshStructured`| `x,y,z` | Arrays: List of all grid points. |\n\n"
            "**2. Tally Definitions**\n"
            "| Class | Required Param | Use Case |\n"
            "| :--- | :--- | :--- |\n"
            "| `TallyGlobal` | None | System-wide averages. |\n"
            "| `TallySurface`| `surface` | Crossing rates (Leakage). |\n"
            "| `TallyCell` | `cell` | Volumetric averages (Reaction rates). |\n"
            "| `TallyMesh` | `mesh` | Spatial maps (Heat maps). |\n\n"
            "**3. Scores & Bins**\n"
            "| Score | Meaning | Bins (Optional) |\n"
            "| :--- | :--- | :--- |\n"
            "| `'flux'` | Scalar Flux ($\phi$). Traffic density. | `energy`, `time` |\n"
            "| `'net-current'` | Vector Flow ($J$). Net flow across surface. | `mu` (angle) |\n"
            "| `'fission'` | Fission Rate. Power generation. | `energy`, `time` |\n"
            "| `'density'` | Particle Density ($n$). | `time` |\n"
            "| `'collision'` | Total Collision Rate. | `energy` |"
        ),
        "tips": [
            "**Flux vs. Current:**\n"
            " * **Flux** measures 'track length per volume'. It doesn't care about direction. Use this for reaction rates.\n"
            " * **Current** measures 'particles crossing a line'. It is directional (+ or -). Use this for surface leakage.",
            "**Mesh Syntax:**\n"
            " * `MeshUniform` takes a **tuple** of 3 values: `(start, stop, N)`.\n"
            " * `MeshStructured` takes a **list/array** of $N+1$ values.",
            "**Performance:** Tallying on a mesh with millions of voxels will consume a lot of RAM and slow down the simulation.",
        ],
    },
    "settings": {
        "concept": (
            "**Settings are the 'Control Panel' for the simulation engine.**\n\n"
            "They control **Precision** (how many particles), **Mode** (Fixed Source vs. Criticality), "
            "and **Memory** (Buffer sizes). These settings are global and affect the entire run.\n\n"
            "**Two Main Modes:**\n"
            "1. **Fixed Source:** (Default) Particles start from the sources you defined. Used for shielding/detectors.\n"
            "2. **Eigenmode:** Calculates $k_{eff}$ (criticality). Particles from the previous cycle generate the source for the next."
        ),
        "syntax": """# 1. Basic Fixed Source Run
# Runs 10 batches of 1000 particles each (Total = 10,000)
mcdc.settings.N_particle = 1000
mcdc.settings.N_batch    = 10
mcdc.settings.output_name = "shielding_results.h5"

# 2. Eigenmode Run (Criticality / k-eff)
# Requires 'N_inactive' (to settle) and 'N_active' (to record)
mcdc.settings.N_particle = 5000
mcdc.settings.set_eigenmode(
    N_inactive = 10,
    N_active   = 40
)

# 3. Buffer Management (If you crash with memory errors)
mcdc.settings.active_bank_buffer = 10000  # Max particles in flight
mcdc.settings.source_bank_buffer_ratio = 2.0 # Buffer relative to N_particle""",
        "parts": (
            "**Core Parameters**\n"
            "| Param | Type | Description |\n"
            "| :--- | :--- | :--- |\n"
            "| `N_particle` | `int` | Histories to run per batch (or per cycle). |\n"
            "| `N_batch` | `int` | Number of batches (for Fixed Source statistics). |\n"
            "| `rng_seed` | `int` | Seed for reproducibility. |\n"
            "| `output_name`| `str` | Filename for HDF5 output (default 'output.h5'). |\n"
            "| `progress_bar`| `bool` | Show/Hide the progress bar. |\n\n"
            "**Eigenmode (Criticality)**\n"
            "| Method | Params | Description |\n"
            "| :--- | :--- | :--- |\n"
            "| `set_eigenmode`| `N_inactive` | Cycles run to converge source (discarded). |\n"
            "| | `N_active` | Cycles run to accumulate data (tallied). |\n\n"
            "**Memory Buffers (Advanced)**\n"
            "| Param | Description |\n"
            "| :--- | :--- |\n"
            "| `active_bank_buff` | Max number of particles alive simultaneously. |\n"
            "| `census_bank_buff` | Buffer for particles crossing time boundaries. |"
        ),
        "tips": [
            "**Statistics:** Simulation error drops with $\\frac{1}{\\sqrt{N}}$. To cut error in half, you need 4x the particles.",
            "**Bank Full Errors:** If your simulation crashes with an error about 'banks' or 'buffers', increase `active_bank_buff` or `source_bank_buffer_ratio`.",
            "**Eigenmode:** Always run 'Inactive' cycles first! The particle distribution needs time to settle into the fundamental eigenmode before you start recording valid data.",
        ],
    },
}
