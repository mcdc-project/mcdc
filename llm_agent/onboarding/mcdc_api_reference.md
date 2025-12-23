# MCDC API Reference (Quick Reference for LLM Agent)

## Materials

### MaterialMG (Multi-Group)
```python
mcdc.MaterialMG(
    name: str = '',
    capture: np.ndarray,         # [G] capture cross-section [/cm]
    scatter: np.ndarray,         # [[G,G]] scatter matrix - MUST BE 2D!
    fission: np.ndarray = None,  # [G] fission cross-section [/cm]
    nu_p: np.ndarray = None,     # [G] prompt fission neutron yield (REQUIRED if fission)
    nu_s: np.ndarray = None,     # [G] scattering multiplication
    chi_p: np.ndarray = None,    # [Gout, Gin] prompt fission spectrum
    speed: np.ndarray = None,    # [G] energy group speed [cm/s]
)
# PHYSICS RULE: For subcritical systems: fission * nu_p < capture + scatter
# Example: fission=0.05, nu_p=2.5, capture=0.15 -> 0.125 < 0.15 ✓
```

### Material (Continuous Energy)
```python
mcdc.Material(
    name: str = '',
    nuclide_composition: dict[str, float],  # {'U235': 0.02, 'U238': 0.98}
)
# Use lookup_material tool to get correct nuclide compositions!
```

---

## Surfaces

### PlaneX, PlaneY, PlaneZ
```python
mcdc.Surface.PlaneX(x: float, boundary_condition: str = 'none')
mcdc.Surface.PlaneY(y: float, boundary_condition: str = 'none')
mcdc.Surface.PlaneZ(z: float, boundary_condition: str = 'none')
# boundary_condition: 'none' | 'vacuum' | 'reflective'
```

### CylinderX, CylinderY, CylinderZ
```python
# IMPORTANT: center is 2D - perpendicular to cylinder axis!
mcdc.Surface.CylinderX(center: [y, z], radius: float, boundary_condition: str = 'none')
mcdc.Surface.CylinderY(center: [x, z], radius: float, boundary_condition: str = 'none')  
mcdc.Surface.CylinderZ(center: [x, y], radius: float, boundary_condition: str = 'none')
# CylinderZ at origin: center=[0.0, 0.0] NOT [0,0,0]
```

### Sphere
```python
mcdc.Surface.Sphere(center: [x, y, z], radius: float, boundary_condition: str = 'none')
# center is 3D for spheres
```

---

## Cells and Regions

### Cell
```python
mcdc.Cell(
    region: Region,              # Boolean expression of surfaces
    fill: Material | Universe,   # What fills the cell
    translation: [x, y, z] = [0, 0, 0],  # For universe fills
    rotation: [rx, ry, rz] = [0, 0, 0],  # Rotation angles (degrees)
)
```

### Region Operators
```python
inside = -sphere           # Inside surface (negative side)
outside = +sphere          # Outside surface (positive side)
intersection = +s1 & -s2   # AND: must satisfy both
union = region1 | region2  # OR: satisfies either
complement = ~region       # NOT: everything except region
```

---

## Source

```python
mcdc.Source(
    x: [xmin, xmax] = None,      # Position range
    y: [ymin, ymax] = None,
    z: [zmin, zmax] = None,
    position: [x, y, z] = None,  # Point source (alternative to x,y,z)
    isotropic: bool = True,      # Isotropic emission
    energy_group: int = 0,       # For MG problems
    time: [tmin, tmax] = 0.0,    # Time range
    probability: float = 1.0,    # Relative probability
)
# TIP: Use small source region unless volumetric source explicitly needed
# Good default: x=[-0.1, 0.1] (point-like)
```

---

## Tallies

### Meshes
```python
mesh = mcdc.MeshUniform(
    x: (xmin, xmax, Nx),  # N cells in x
    y: (ymin, ymax, Ny),
    z: (zmin, zmax, Nz),
)

mesh = mcdc.MeshStructured(
    x: [x0, x1, x2, ...],  # Explicit grid points
    y: [y0, y1, y2, ...],
    z: [z0, z1, z2, ...],
)
# RULE: Keep mesh coarse! Total cells = Nx*Ny*Nz < 10000
# N_particle >= Total_Mesh_Cells to avoid variance errors
```

### TallyMesh
```python
mcdc.TallyMesh(
    mesh: MeshUniform | MeshStructured,
    scores: list[str] = ['flux'],  # 'flux', 'fission', 'total', etc.
)

### TallyCell
```python
mcdc.TallyCell(
    cell: Cell,           # Single cell - NOT cells= (common mistake!)
    scores: list[str] = ['flux'],
)
```

### TallySurface
```python
mcdc.TallySurface(
    surface: Surface,     # Single surface - NOT surfaces=
    scores: list[str] = ['flux'],
)
```

### TallyGlobal
```python
mcdc.TallyGlobal(
    scores: list[str] = ['flux'],  # Tally over entire geometry
    energy: list[float] = None,    # Energy bin edges
    time: list[float] = None,      # Time bin edges
)
```

---

## Universe & Lattice

### Universe
```python
mcdc.Universe(
    cells: list[Cell],
)

# to set root universe
mcdc.simulation.set_root_universe(
    cells: list[Cell],
)
```

### Lattice
```python
mcdc.Lattice(
    x: (xmin, xmax, Nx),  # Grid in x
    y: (ymin, ymax, Ny),
    z: (zmin, zmax, Nz),
    universes: list[Universe],  # Fill pattern
)
```

---

## Settings

```python
mcdc.settings.N_particle = 1000   # Number of particles per batch
mcdc.settings.N_batch = 2         # Number of batches
mcdc.settings.active_bank_buffer = 1000  # For fission problems
```

### Particle Count Guidelines:
- Point/small source: N_particle >= 1000
- Large source with mesh: N_particle >= Total_Mesh_Cells
- Fission problems: may need active_bank_buffer if supercritical

### Eigenmode (k-eigenvalue)
```python
mcdc.settings.set_eigenmode(
    N_inactive: int = 0,         # Inactive cycles (source convergence)
    N_active: int = 0,           # Active cycles (for statistics)
    k_init: float = 1.0,         # Initial k guess
    gyration_radius: str = None, # 'all', 'infinite-x', 'only-z', etc.
    save_particle: bool = False, # Save final particle bank
)
# Total cycles = N_inactive + N_active
```

---

## Common Scores for Tallies
- 'flux' - Scalar flux
- 'fission' - Fission rate
- 'total' - Total reaction rate
- 'capture' - Capture rate
- 'scatter' - Scatter rate
- 'net-current' - Net current (surface tallies)
