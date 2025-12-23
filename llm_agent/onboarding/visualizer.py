import ast
import subprocess
from pathlib import Path
from typing import Dict, Any
from rich.console import Console


class GeometryVisualizer:
    """
    Visualizes MCDC geometry from ScriptBuilder state.

    Supports:
    - 2D slice visualization when cells are defined
    - 3D wireframe visualization for surfaces only
    """

    def __init__(self, builder: Any, console: Console = None):
        """
        Initialize visualizer with a ScriptBuilder instance.

        Args:
            builder: ScriptBuilder instance containing geometry definitions
            console: Optional Rich Console for output (creates new one if not provided)
        """
        self.builder = builder
        self.console = console or Console()

    def _extract_cell_regions(self) -> Dict[str, str]:
        """
        Extract region expressions from cell entries, resolving variable references.

        Returns:
            Dict mapping cell names to their region expression strings
        """
        cell_logic_map = {}

        # First, build a map of region variables to their expressions
        region_vars = {}
        for entry in self.builder.entries:
            if entry["type"] == "region":
                # Parse: inside_sphere = -sphere
                code = entry["code"]
                if "=" in code:
                    var_name = code.split("=")[0].strip()
                    expression = code.split("=", 1)[1].strip()
                    region_vars[var_name] = expression

        # Then extract cell regions, resolving references
        for entry in self.builder.entries:
            if entry["type"] == "cell":
                try:
                    tree = ast.parse(entry["code"])
                    for node in ast.walk(tree):
                        if isinstance(node, ast.Call):
                            for keyword in node.keywords:
                                if keyword.arg == "region":
                                    # Extract the source text for the region value
                                    region_str = ast.get_source_segment(
                                        entry["code"], keyword.value
                                    )
                                    # Resolve if it's a variable reference
                                    if region_str in region_vars:
                                        region_str = region_vars[
                                            region_str
                                        ]  # use the expression instead of the variable name
                                    cell_logic_map[entry["name"]] = region_str
                                    break
                except Exception as e:
                    print(
                        f"Warning: Could not parse region for cell {entry['name']}: {e}"
                    )

        return cell_logic_map

    def _generate_slice_code(
        self, cell_logic_map: Dict[str, str], axis: str, position: float
    ) -> str:
        """Generate Python code for 2D slice visualization. Uses prewritten code, but injects cell logic and axis/position."""
        return f"""
# --- SLICE VISUALIZATION APPENDED BY AGENT ---
import matplotlib.pyplot as plt
import numpy as np

# INJECTED SETTINGS
CELL_REGIONS = {str(cell_logic_map)}
SLICE_AXIS = '{axis}'
SLICE_VAL = {position}
BOUNDS = 15.0
RES = 150

def run_slice_viz(local_vars):
    print(f"Scanning {{RES}}x{{RES}} pixels at {{SLICE_AXIS.upper()}}={{SLICE_VAL}}...")

    # 1. Identify Surfaces
    surfaces = {{}}
    for name, obj in local_vars.items():
        if hasattr(obj, 'A') and hasattr(obj, 'J'):
            surfaces[name] = obj

    # 2. Helper: Evaluate Surface Equation
    def eval_surf(s, x, y, z):
        return (getattr(s,'A',0)*x**2 + getattr(s,'B',0)*y**2 + getattr(s,'C',0)*z**2 +
                getattr(s,'D',0)*x*y  + getattr(s,'E',0)*y*z  + getattr(s,'F',0)*z*x +
                getattr(s,'G',0)*x    + getattr(s,'H',0)*y    + getattr(s,'I',0)*z + 
                getattr(s,'J',0))

    # 3. Setup Dynamic Grid based on Axis
    u = np.linspace(-BOUNDS, BOUNDS, RES)
    v = np.linspace(-BOUNDS, BOUNDS, RES)
    U, V = np.meshgrid(u, v)
    
    # Map 2D grid (U,V) to 3D coordinates (PX, PY, PZ)
    if SLICE_AXIS == 'z':
        PX, PY, PZ = U, V, np.full_like(U, SLICE_VAL)
        xlabel, ylabel = 'X [cm]', 'Y [cm]'
    elif SLICE_AXIS == 'y':
        PX, PY, PZ = U, np.full_like(U, SLICE_VAL), V
        xlabel, ylabel = 'X [cm]', 'Z [cm]'
    elif SLICE_AXIS == 'x':
        PX, PY, PZ = np.full_like(U, SLICE_VAL), U, V
        xlabel, ylabel = 'Y [cm]', 'Z [cm]'

    img = np.zeros((RES, RES)) - 1 
    sorted_surfs = sorted(surfaces.keys(), key=len, reverse=True)
    cell_names = list(CELL_REGIONS.keys())
    
    # 4. Scan Grid
    for i in range(RES):
        for j in range(RES):
            # Get real 3D coordinates for this pixel
            px, py, pz = PX[i,j], PY[i,j], PZ[i,j]
            
            for c_idx, c_name in enumerate(cell_names):
                logic = CELL_REGIONS[c_name]
                try:
                    for s_name in sorted_surfs:
                        if s_name in logic:
                            # Evaluate using the 3D coordinate for this pixel
                            val = eval_surf(surfaces[s_name], px, py, pz)
                            logic = logic.replace(f"+{{s_name}}", str(val > 0))
                            logic = logic.replace(f"-{{s_name}}", str(val < 0))
                    
                    logic = logic.replace("&", " and ").replace("|", " or ").replace("~", " not ")
                    if eval(logic):
                        img[i,j] = c_idx
                        break 
                except Exception:
                    pass

    # 5. Plot
    fig, ax = plt.subplots(figsize=(8,8))
    
    if len(cell_names) > 0:
        cmap = plt.get_cmap('tab20', len(cell_names))
    else:
        cmap = plt.get_cmap('Greys')
        
    masked_img = np.ma.masked_where(img == -1, img)
    
    ax.imshow(masked_img, origin='lower', extent=[-BOUNDS, BOUNDS, -BOUNDS, BOUNDS], 
               cmap=cmap, vmin=0, vmax=len(cell_names)-1)
    
    # Legend
    from matplotlib.patches import Patch
    patches = [Patch(color=cmap(i), label=name) for i, name in enumerate(cell_names)]
    ax.legend(handles=patches, loc='upper right', title="Cells")
    
    ax.set_title(f"Cell Region Slice ({{SLICE_AXIS.upper()}}={{SLICE_VAL}})")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3, linestyle='--')
    plt.show()

try:
    run_slice_viz(locals())
except Exception as e:
    print(f"Slice Viz Error: {{e}}")
"""

    def _generate_wireframe_code(self) -> str:
        """Generate Python code for 3D wireframe visualization. Uses prewritten code, but injects surfaces"""
        return r"""
# --- WIREFRAME VISUALIZATION APPENDED BY AGENT ---
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import math

def run_wireframe_viz(local_vars):
    BOUNDS = 15.0
    RES = 20
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title("Surface Wireframe Preview")
    ax.set_xlim(-BOUNDS, BOUNDS); ax.set_ylim(-BOUNDS, BOUNDS); ax.set_zlim(-BOUNDS, BOUNDS)
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    
    grid = np.linspace(-BOUNDS, BOUNDS, RES)
    U, V = np.meshgrid(grid, grid)
    found = 0

    print("-" * 40)
    print("Scanning Surfaces...")

    for name, obj in local_vars.items():
        if not (hasattr(obj, 'A') and hasattr(obj, 'J')): continue
        
        try:
            A, B, C = getattr(obj,'A',0), getattr(obj,'B',0), getattr(obj,'C',0)
            G, H, I, J = getattr(obj,'G',0), getattr(obj,'H',0), getattr(obj,'I',0), getattr(obj,'J',0)
            
            if abs(A)+abs(B)+abs(C) < 1e-9:
                if abs(I) > 1e-9:   # Plane Z
                    val = -J/I; ax.plot_surface(U, V, np.full_like(U, val), alpha=0.2, color='blue')
                elif abs(G) > 1e-9: # Plane X
                    val = -J/G; ax.plot_surface(np.full_like(U, val), U, V, alpha=0.2, color='red')
                elif abs(H) > 1e-9: # Plane Y
                    val = -J/H; ax.plot_surface(U, np.full_like(U, val), V, alpha=0.2, color='green')
                found += 1

            # --- CYLINDER Z ---
            elif abs(C) < 1e-9 and abs(A-B) < 1e-5 and abs(A) > 1e-9: 
                x0, y0 = -G/(2*A), -H/(2*B)
                r = math.sqrt(max(0, x0**2 + y0**2 - J/A))
                z = np.linspace(-BOUNDS, BOUNDS, RES)
                th = np.linspace(0, 2*np.pi, RES)
                TH, Z = np.meshgrid(th, z)
                Xg = r * np.cos(TH) + x0
                Yg = r * np.sin(TH) + y0
                ax.plot_surface(Xg, Yg, Z, alpha=0.3, color='cyan')
                found += 1

            # --- CYLINDER X ---
            elif abs(A) < 1e-9 and abs(B-C) < 1e-5 and abs(B) > 1e-9:
                y0, z0 = -H/(2*B), -I/(2*C)
                r = math.sqrt(max(0, y0**2 + z0**2 - J/B))
                x = np.linspace(-BOUNDS, BOUNDS, RES)
                th = np.linspace(0, 2*np.pi, RES)
                Xg, TH = np.meshgrid(x, th)
                Yg = r * np.cos(TH) + y0
                Zg = r * np.sin(TH) + z0
                ax.plot_surface(Xg, Yg, Zg, alpha=0.3, color='orange')
                found += 1

            # --- CYLINDER Y---
            elif abs(B) < 1e-9 and abs(A-C) < 1e-5 and abs(A) > 1e-9:
                x0, z0 = -G/(2*A), -I/(2*C)
                r = math.sqrt(max(0, x0**2 + z0**2 - J/A))
                y = np.linspace(-BOUNDS, BOUNDS, RES)
                th = np.linspace(0, 2*np.pi, RES)
                Yg, TH = np.meshgrid(y, th)
                Xg = r * np.cos(TH) + x0
                Zg = r * np.sin(TH) + z0
                ax.plot_surface(Xg, Yg, Zg, alpha=0.3, color='yellow')
                found += 1

            # --- SPHERE ---
            elif abs(A-B) < 1e-5 and abs(A-C) < 1e-5 and abs(A) > 1e-9: # Sphere
                x0, y0, z0 = -G/(2*A), -H/(2*B), -I/(2*C)
                r = math.sqrt(max(0, x0**2 + y0**2 + z0**2 - J/A))
                u = np.linspace(0, 2*np.pi, RES)
                v = np.linspace(0, np.pi, RES)
                Xg = r * np.outer(np.cos(u), np.sin(v)) + x0
                Yg = r * np.outer(np.sin(u), np.sin(v)) + y0
                Zg = r * np.outer(np.ones(np.size(u)), np.cos(v)) + z0
                ax.plot_surface(Xg, Yg, Zg, alpha=0.3, color='magenta')
                found += 1
                
        except Exception as e:
            print(f"  ! Error plotting {name}: {e}")

    if found == 0:
        ax.text(0,0,0, "No Surfaces", color='k')
    plt.show()

try:
    run_wireframe_viz(locals())
except Exception as e:
    print(f"Wireframe Error: {e}")
"""

    def run(self, axis: str = "z", position: float = 0.0):
        """
        Run geometry visualization.

        If cells are defined, shows 2D slice at the specified axis/position.
        Otherwise, shows 3D wireframe of surfaces.

        Args:
            axis: 'x', 'y', or 'z' for slice plane orientation
            position: Position along the axis for the slice
        """
        cell_logic_map = self._extract_cell_regions()
        has_cells = len(cell_logic_map) > 0

        # Select visualization mode
        if has_cells:
            plot_code = self._generate_slice_code(cell_logic_map, axis, position)
        else:
            plot_code = self._generate_wireframe_code()

        # Create temp script and execute
        viz_file = "temp_viz_script.py"
        base_script = self.builder.get_script(include_run=False)
        full_script = base_script + "\n" + plot_code
        Path(viz_file).write_text(full_script)

        try:
            subprocess.run(["python", viz_file], check=True)
            self.console.print("[success]Visualization closed.[/success]")
        except subprocess.CalledProcessError:
            self.console.print("[error]Visualization failed.[/error]")
        finally:
            Path(viz_file).unlink(missing_ok=True)


def run_visualization(
    builder: Any, console: Console, axis: str = "z", position: float = 0.0
):
    """
    Convenience function to run visualization without creating a class instance.

    Args:
        builder: ScriptBuilder instance
        console: Rich Console for output
        axis: Slice axis ('x', 'y', or 'z')
        position: Position along the slice axis
    """
    visualizer = GeometryVisualizer(builder, console)
    visualizer.run(axis=axis, position=position)
