from typing import Dict, List, Optional
from dataclasses import dataclass

# Geometry Pattern Templates


@dataclass
class GeometryPattern:
    """Template for a common geometry pattern."""

    name: str
    description: str
    surfaces: List[str]
    cells: List[str]
    region_pattern: str
    example_file: Optional[str] = None


GEOMETRY_PATTERNS: Dict[str, GeometryPattern] = {
    "infinite_medium": GeometryPattern(
        name="Infinite Medium",
        description="Homogeneous infinite medium with reflective boundaries",
        surfaces=["Two parallel planes with reflective BCs"],
        cells=["Single cell between planes"],
        region_pattern="+plane_min & -plane_max, fill=material",
        example_file="azurv1.json",
    ),
    "slab_layers": GeometryPattern(
        name="Slab Layers",
        description="1D layered geometry (shielding, materials)",
        surfaces=["Parallel planes defining layer boundaries"],
        cells=["Adjacent cells: +plane_i & -plane_{i+1}"],
        region_pattern="+s1 & -s2 fills m1; +s2 & -s3 fills m2; etc.",
        example_file="slab_reed.json",
    ),
    "pin_cell": GeometryPattern(
        name="Pin Cell",
        description="Cylindrical fuel pin in square/hex moderator pitch",
        surfaces=["CylinderZ for pin", "4 planes for pitch (reflective BCs)"],
        cells=[
            "-cylinder fills fuel (inside cylinder)",
            "+cylinder & inside_pitch fills moderator",
        ],
        region_pattern="-cy fills fuel; +cy & +x1 & -x2 & +y1 & -y2 fills moderator",
        example_file="inf_pin_ce.json",
    ),
    "sphere_in_box": GeometryPattern(
        name="Sphere in Box",
        description="Spherical region inside cubic boundary",
        surfaces=["Sphere at center", "6 bounding planes with vacuum BCs"],
        cells=[
            "-sphere fills inner_material",
            "inside_box & ~(-sphere) fills outer_material",
        ],
        region_pattern="-sphere fills A; inside_box & ~inside_sphere fills B",
        example_file="sphere_in_cube.json",
    ),
    "nested_shells": GeometryPattern(
        name="Nested Shells",
        description="Concentric spheres or cylinders (fuel pellet in cladding)",
        surfaces=["Inner sphere/cylinder", "Outer sphere/cylinder", "Bounding planes"],
        cells=[
            "-inner fills fuel",
            "+inner & -outer fills cladding (shell region)",
            "+outer fills exterior",
        ],
        region_pattern="-inner fills A; +inner & -outer fills B; +outer fills C",
        example_file="fuel_array_packaged.json",
    ),
    "streaming_channel": GeometryPattern(
        name="Streaming/Dog-Leg Channel",
        description="Void channel through shielding (Kobayashi benchmark)",
        surfaces=["Multiple planes defining channel segments"],
        cells=[
            "void_channel = union of channel segments",
            "shield = bounding_box & ~void_channel",
        ],
        region_pattern="channel_1 | channel_2 | ... fills void; box & ~void_channel fills shield",
        example_file="kobayashi3-TD.json",
    ),
    "container_cells": GeometryPattern(
        name="Container Cells (Assembly Array)",
        description="Multiple copies of assembly with translation/rotation - REQUIRES SEPARATE REGIONS",
        surfaces=[
            "Bounding planes for overall domain (with vacuum BCs)",
            "Mid-plane to SEPARATE container regions (e.g., mid_x at x=0)",
        ],
        cells=[
            "inner_universe = Universe(cells=[fuel, cover, water])",
            "container_left = Cell(region=+min_x & -mid_x & ..., fill=inner_universe, translation=[-5,0,0])",
            "container_right = Cell(region=+mid_x & -max_x & ..., fill=inner_universe, translation=[5,0,0], rotation=[0,10,0])",
        ],
        region_pattern="CRITICAL: Container cells must have DIFFERENT, NON-OVERLAPPING regions. Use mid_x plane to split: left uses +min_x & -mid_x, right uses +mid_x & -max_x",
        example_file="fuel_array_packaged.json",
    ),
    "lattice_array": GeometryPattern(
        name="Lattice Array (Reactor Core)",
        description="Regular array of pin cells or assemblies",
        surfaces=["Pin cylinder", "Lattice bounding planes"],
        cells=[
            "Pin universe with fuel + moderator cells",
            "Assembly lattice arranging pin universes",
            "Core lattice arranging assembly universes",
        ],
        region_pattern="Multi-level: pin → assembly → core via Lattice and Universe",
        example_file=None,  # No example in current set
    ),
    "hexagonal_lattice": GeometryPattern(
        name="Hexagonal Lattice",
        description="Hexagonal array pattern (VVER, pebble bed)",
        surfaces=["Pin cylinders", "Hexagonal boundary planes"],
        cells=["Pin universes arranged in HexLattice with specified pitch"],
        region_pattern="HexLattice with center, pitch, and universes array",
        example_file=None,  # No example in current set
    ),
    "moving_geometry": GeometryPattern(
        name="Moving/Transient Geometry",
        description="Time-dependent geometry (control rod, pulsed systems)",
        surfaces=["Surfaces with .move() specifying velocity/time"],
        cells=["Cells using moving surfaces"],
        region_pattern="surface.move([[dx,dy,dz], ...], [t1, t2, ...])",
        example_file="moving_pellet.json",
    ),
}


# Pattern Detection Keywords

PATTERN_KEYWORDS = {
    "infinite_medium": ["infinite", "homogeneous", "reflective", "benchmark"],
    "slab_layers": ["slab", "layer", "1d", "shield", "multilayer"],
    "pin_cell": ["pin", "fuel rod", "coolant", "pitch", "cladding"],
    "sphere_in_box": ["sphere", "cube", "box", "centered", "inside"],
    "nested_shells": ["shell", "concentric", "nested", "annular", "pellet"],
    "streaming_channel": [
        "channel",
        "void",
        "streaming",
        "duct",
        "dog-leg",
        "kobayashi",
    ],
    "container_cells": [
        "copy",
        "array",
        "translation",
        "rotation",
        "assembly",
        "duplicate",
    ],
    "lattice_array": ["lattice", "reactor", "core", "assembly", "array", "grid"],
    "hexagonal_lattice": ["hexagonal", "hex", "vver", "pebble"],
    "moving_geometry": [
        "moving",
        "transient",
        "time-dependent",
        "control rod",
        "pulsed",
    ],
}


def detect_patterns(prompt: str) -> List[str]:
    """
    Detect which geometry patterns are likely relevant to a prompt.

    Returns list of pattern names in order of relevance.
    """
    prompt_lower = prompt.lower()
    scores = {}

    # Count keyword matches for each pattern
    for pattern_name, keywords in PATTERN_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in prompt_lower)
        if score > 0:
            scores[pattern_name] = score

    # Sort by score descending
    return sorted(scores.keys(), key=lambda x: scores[x], reverse=True)


def get_pattern_guidance(pattern_names: List[str]) -> str:
    """
    Generate guidance text for detected patterns.
    """
    if not pattern_names:
        return ""

    lines = ["## Detected Geometry Patterns\n"]

    for name in pattern_names[:3]:  # Top 3 patterns
        pattern = GEOMETRY_PATTERNS.get(name)
        if pattern:
            lines.append(f"### {pattern.name}")
            lines.append(f"**Structure**: {pattern.description}")
            lines.append(f"**Region Pattern**: `{pattern.region_pattern}`")
            if pattern.example_file:
                lines.append(f"**Reference**: See `{pattern.example_file}`")
            lines.append("")

    return "\n".join(lines)
