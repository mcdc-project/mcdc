# 20 MeV CubeSat: STEP file to MC/DC input

`input.py` builds the 1U CubeSat geometry by hand. `Cubesat_Geometry_CAD.step` holds the same 40 parts as a CAD model. This page shows how to turn the STEP file into MC/DC geometry with GEOUNED.

```
Cubesat_Geometry_CAD.step ─► GEOUNED (FreeCAD Python) ─► cubesat_geouned_mcdc.py ─► assign materials ─► MC/DC
```

## Packages

| Package | Version used |
|---|---|
| FreeCAD | 1.1.1 (supplies the Python that GEOUNED runs in) |
| GEOUNED | `github.com/gunnarrl/GEOUNED`, branch `dev-collaboration` (adds the `"mcdc"` output format) |
| MC/DC | `dev` branch |

Install GEOUNED into FreeCAD's Python (macOS path shown):

```bash
FREECAD_PY=/Applications/FreeCAD.app/Contents/Resources/bin/python
git clone -b dev-collaboration https://github.com/gunnarrl/GEOUNED.git
$FREECAD_PY -m pip install -e GEOUNED
```

## Inputs

| File | Contents |
|---|---|
| `Cubesat_Geometry_CAD.step` | 40 box solids, `Body1`–`Body40`, in mm, no material tags |
| `input.py` | Hand-built model: materials, ECSS neutron source, SV tallies |
| `ecss_20MeV.csv` | ECSS neutron spectrum cut at 20 MeV |

## 1. Convert the STEP file

`convert.py`, run with `$FREECAD_PY convert.py`:

```python
import FreeCAD
import geouned

# Make GEOUNED's void box the 15 cm source cube from input.py
# (xmin, ymin, zmin, xmax, ymax, zmax in mm). The default box only pads the
# CubeSat by 1 cm, so source points near the cube corners would land outside
# the model.
def fixed_box(self, padding=10.0):
    self.geometry_bounding_box = FreeCAD.BoundBox(-25, -25, -25, 125, 125, 125)
    return self.geometry_bounding_box

geouned.CadToCsg._get_geometry_bounding_box = fixed_box

geo = geouned.CadToCsg()
geo.load_step_file(filename="Cubesat_Geometry_CAD.step")
geo.start()
geo.export_csg(geometryName="cubesat_geouned", outFormat=("mcdc",))
```

This writes `cubesat_geouned_mcdc.py`, which contains surfaces and cells only. Each body becomes one cell named `/Cubesat_Geometry_CAD<n>`, followed by GEOUNED's void cells. Every cell has `fill=None`.

## 2. Assign materials

The body order matches the part order in `input.py`:

| Bodies | Part | Material |
|---|---|---|
| 1–12 | Rails | Al7075 |
| 13–18 | Shear panels | Al6061 |
| 19–28 | Solar panels | Silicon |
| 29–32 | Antenna strips | Copper |
| 33, 35, 37, 39 | OBC, EPS, ADCS, Comms boards | FR4 |
| 34, 36, 38, 40 | OBC, EPS, ADCS, Comms SVs | Silicon, LiCoO2, Silicon, Silicon |

Load the GEOUNED file and fill every cell. Use the materials defined in `input.py`:

```python
ns = {}
exec(open("cubesat_geouned_mcdc.py").read(), ns)
cells = [c for c in ns.values() if isinstance(c, mcdc.Cell)]

body_material = (
    [m_al7075] * 12 + [m_al6061] * 6 + [m_silicon] * 10 + [m_copper] * 4
    + [m_epoxy, m_silicon, m_epoxy, m_licoo2, m_epoxy, m_silicon, m_epoxy, m_silicon]
)
prefix = "/Cubesat_Geometry_CAD"
for c in cells:
    if c.name.startswith(prefix):
        c.fill = body_material[int(c.name[len(prefix):]) - 1]
    else:
        c.fill = m_void  # GEOUNED void cells; fill=None would lose particles

simulation.set_model(cells)
```

The source, tallies and settings are the same as in `input.py`.
