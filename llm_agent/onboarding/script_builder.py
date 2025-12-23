from typing import List, Dict, Set, Any, Optional
from pathlib import Path
import ast
import re
import copy
from collections import defaultdict, deque


class ScriptBuilder:
    """
    Tracks the state of the MCDC script being built.
    Now supports Undo and Delete operations.
    """

    def __init__(self):
        self.imports = ["import mcdc", "import numpy as np", ""]
        # Store entries as dictionaries
        # Format: {'code': str, 'type': str, 'name': str}
        self.entries: List[Dict[str, str]] = []

        self.defined: Dict[str, Set[str]] = {
            "material": set(),
            "surface": set(),
            "cell": set(),
            "universe": set(),
            "lattice": set(),
            "mesh": set(),
            "source": set(),
            "tally": set(),
            "settings": set(),
            "region": set(),  # Track intermediate region variables
        }

        # Checkpoints for rollback support (list of (entries, defined) tuples)
        self._checkpoints: List[tuple] = []

    def add_line(self, code: str, entity_type: str, name: str):
        """
        Add a line of code and register the entity.

        If entity already exists, replaces it (deduplication).
        """
        # DEDUPLICATION: If entity already exists, replace it
        if entity_type in self.defined and name in self.defined[entity_type]:
            for i, entry in enumerate(self.entries):
                if entry["type"] == entity_type and entry["name"] == name:
                    self.entries[i] = {"code": code, "type": entity_type, "name": name}
                    # Don't need to update defined or reorder since name already tracked
                    return

        self.entries.append({"code": code, "type": entity_type, "name": name})

        # Auto-create key if it doesn't exist to prevent KeyErrors
        if entity_type not in self.defined:
            self.defined[entity_type] = set()

        self.defined[entity_type].add(name)

        # Auto-sort to fix dependencies and grouping
        self.reorder()

    def deduplicate_settings(self):
        """
        Remove duplicate entries, keeping the last value for each one.

        Call this after batch execution to clean up.
        """
        # Track which setting keys we've seen (going backwards)
        seen_keys = set()
        new_entries = []

        # keep last occurrence (reverse order)
        for entry in reversed(self.entries):
            if entry["type"] == "settings":
                # Extract setting key from code (e.g., 'N_particle' from 'mcdc.settings.N_particle = 100')
                code = entry["code"]
                # Check for duplicates by looking at specific setting keys
                lines = code.strip().split("\n")
                unique_lines = []
                for line in lines:
                    if "mcdc.settings." in line:
                        # Extract key like 'N_particle'
                        match = (
                            line.split("mcdc.settings.")[1].split("=")[0].strip()
                            if "mcdc.settings." in line
                            else None
                        )
                        # Check if key is already seen
                        if match and match not in seen_keys:
                            seen_keys.add(match)
                            unique_lines.append(line)
                    elif "set_eigenmode" in line:
                        if "set_eigenmode" not in seen_keys:
                            seen_keys.add("set_eigenmode")
                            unique_lines.append(line)
                    else:
                        unique_lines.append(line)
                if unique_lines:
                    entry["code"] = "\n".join(unique_lines)
                    new_entries.append(entry)
            else:
                new_entries.append(entry)

        # Reverse back to original order
        self.entries = list(reversed(new_entries))

    def reorder(self):
        """
        Topologically sorts the script entries based on variable dependencies.

        Updates:
        1. Ensures 'settings' always appear at the bottom.
        2. Parses 'region' strings to find hidden dependencies.
        3. Bubbles up commented entries to the top of their type-block.
        """
        # Separate types that should stay at the end from those that need dependency ordering
        free_entries = []
        graph_entries = []

        # These types should stay at the end in their insertion order
        end_types = {"settings", "source", "tally", "hierarchy"}

        for entry in self.entries:
            if entry["type"] in end_types:
                free_entries.append(entry)
            else:
                graph_entries.append(entry)

        # Map names to indices (relative to graph_entries)
        name_to_idx = {entry["name"]: i for i, entry in enumerate(graph_entries)}

        # Build dependency graph
        adj = defaultdict(set)
        in_degree = defaultdict(int)

        def add_dependency(u, v):
            """u depends on v (v must come before u)"""
            if v != u and u not in adj[v]:
                adj[v].add(u)
                in_degree[u] += 1

        for i, entry in enumerate(graph_entries):
            # AST parsing for direct variable usage
            try:
                tree = ast.parse(entry["code"])
                for node in ast.walk(tree):
                    if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                        ref = node.id
                        if ref in name_to_idx:
                            add_dependency(i, name_to_idx[ref])
            except Exception:
                continue

        # Kahn's Algorithm for topological sorting
        queue = deque()
        for i in range(len(graph_entries)):
            if in_degree[i] == 0:
                queue.append(i)

        sorted_indices = []
        while queue:
            u = queue.popleft()
            sorted_indices.append(u)

            # Sort neighbors for deterministic output
            for v in sorted(list(adj[u])):
                in_degree[v] -= 1
                if in_degree[v] == 0:
                    queue.append(v)

        # Cycle/error handling (should never happen)
        if len(sorted_indices) < len(graph_entries):
            seen = set(sorted_indices)
            for i in range(len(graph_entries)):
                if i not in seen:
                    sorted_indices.append(i)

        # Construct preliminary list
        new_entries = [graph_entries[i] for i in sorted_indices]

        def depends(idx_a, idx_b):
            """Returns True if entry at new index A depends on entry at new index B"""
            # Check original graph using names
            name_a = new_entries[idx_a]["name"]
            name_b = new_entries[idx_b]["name"]

            # Map back to original indices to check 'adj'
            orig_a = name_to_idx[name_a]
            orig_b = name_to_idx[name_b]

            # Since we only swap adjacent items, we just need to check if A depends on B directly or indirectly.
            # But 'adj' stores direct edges: adj[b] contains a if a depends on b.
            return orig_a in adj[orig_b]

        for i in range(len(new_entries)):
            # Check if this entry has a comment (header)
            if new_entries[i]["code"].strip().startswith("#"):

                # Bubble up
                curr = i
                while curr > 0:
                    prev = curr - 1
                    curr_ent = new_entries[curr]
                    prev_ent = new_entries[prev]

                    # Stop if different type
                    if curr_ent["type"] != prev_ent["type"]:
                        break

                    # Stop if previous one also has a comment (don't reorder headers)
                    if prev_ent["code"].strip().startswith("#"):
                        break

                    # Stop if dependency exists (Current depends on Previous)
                    if depends(curr, prev):
                        break

                    # SWAP
                    new_entries[prev], new_entries[curr] = (
                        new_entries[curr],
                        new_entries[prev],
                    )

                    curr -= 1  # Update index

        # append remaining
        new_entries.extend(free_entries)

        self.entries = new_entries
        return True

    def get_script(self, include_run: bool = True, include_header: bool = False) -> str:
        """
        Reconstructs the script.

        Args:
            include_run: Include mcdc.run() at the end
            include_header: Include metadata header with generation info
        """
        lines = []

        # Add metadata header if requested
        if include_header:
            from datetime import datetime

            lines.append('"""')
            lines.append(f"MCDC Input Script")
            lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            lines.append(f"Generator: MCDC-Agent (Automatic Script Generation)")
            lines.append(f"")
            summary = self.get_state_summary()
            lines.append(f"Entities: {summary['entity_count']} total")
            lines.append(f"  Materials: {', '.join(summary['materials']) or 'None'}")
            lines.append(f"  Surfaces: {len(summary['surfaces'])}")
            lines.append(f"  Cells: {len(summary['cells'])}")
            if summary["missing_requirements"]:
                lines.append(
                    f"  WARNING - Missing: {', '.join(summary['missing_requirements'])}"
                )
            lines.append('"""')
            lines.append("")

        lines.extend(self.imports)
        lines.append("")

        last_type = None

        for entry in self.entries:
            # Add a blank line if switching types
            current_type = entry["type"]
            if last_type and current_type != last_type:
                geo_types = ["surface", "cell", "universe", "lattice"]
                if not (current_type in geo_types and last_type in geo_types):
                    lines.append("")

            lines.append(entry["code"])
            last_type = current_type

        if include_run:
            lines.append("")
            lines.append("# === RUN ===")
            lines.append("mcdc.run()")

        return "\n".join(lines)

    def parse_and_load(self, filepath: str) -> str:
        """
        Parses an existing Python file and populates the ScriptBuilder state.
        Uses AST to identify MCDC entities and variable names.
        """
        path = Path(filepath)
        if not path.exists():
            return f"Error: File {filepath} not found."

        try:
            source_code = path.read_text()
            tree = ast.parse(source_code)
        except Exception as e:
            return f"Error parsing syntax: {e}"

        # Reset current state
        self.reset()

        # Helper to extract source code segment from a node
        def get_segment(node):
            return ast.get_source_segment(source_code, node)

        # MCDC Class to Type Mapping
        type_map = {
            "Material": "material",
            "MaterialMG": "material",
            "Surface": "surface",
            "Cell": "cell",
            "Universe": "universe",
            "Lattice": "lattice",
            "MeshUniform": "mesh",
            "MeshStructured": "mesh",
            "Mesh": "mesh",
            "Source": "source",
            "Tally": "tally",  # Covers TallyGlobal, TallySurface, etc.
        }

        count = 0

        for node in tree.body:
            # Handle Imports
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                code = get_segment(node)
                if code not in self.imports:
                    self.imports.append(code)
                continue

            # Handle Assignments (m1 = mcdc.Material(...))
            if isinstance(node, ast.Assign):
                # We assume single assignment for MCDC entities (m1 = ...)
                target = node.targets[0]
                value = node.value

                # Check if it's an MCDC call
                if isinstance(value, ast.Call):
                    # Resolve function name (handle mcdc.Material and mcdc.Surface.PlaneX)
                    func_name = ""
                    if isinstance(value.func, ast.Attribute):
                        if (
                            isinstance(value.func.value, ast.Name)
                            and value.func.value.id == "mcdc"
                        ):
                            func_name = value.func.attr  # e.g. 'Material'
                        elif isinstance(
                            value.func.value, ast.Attribute
                        ):  # e.g. mcdc.Surface.PlaneX
                            func_name = value.func.value.attr  # 'Surface'

                    # Determine Type
                    entity_type = "other"
                    for key, val in type_map.items():
                        if func_name.startswith(key):
                            entity_type = val
                            break

                    # Extract Name
                    var_name = target.id if isinstance(target, ast.Name) else "unknown"

                    self.add_line(get_segment(node), entity_type, var_name)
                    count += 1
                    continue

                # Handle Region Variables (CSG expressions like: pellet_x = -CylinderX & +bot_x & -top_x)
                if isinstance(value, (ast.BinOp, ast.UnaryOp)):
                    var_name = target.id if isinstance(target, ast.Name) else "unknown"
                    self.add_line(get_segment(node), "region", var_name)
                    count += 1
                    continue

                # Handle Settings (mcdc.settings.x = y)
                if isinstance(target, ast.Attribute) and isinstance(
                    target.value, ast.Attribute
                ):
                    if target.value.attr == "settings":
                        self.add_line(get_segment(node), "settings", target.attr)
                        count += 1
                        continue

            # Handle Standalone Expressions (e.g., mcdc.Source(...) without assignment)
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
                code_segment = get_segment(node)
                func = node.value.func

                # Handle mcdc.xxx(...) calls
                if isinstance(func, ast.Attribute):
                    attr_name = func.attr

                    # Source calls
                    if "Source" in attr_name:
                        self.add_line(
                            code_segment,
                            "source",
                            f"source_{len(self.defined['source'])}",
                        )
                        count += 1
                        continue

                    # Tally calls (TallyMesh, TallyGlobal, TallySurface, etc.)
                    if "Tally" in attr_name:
                        self.add_line(
                            code_segment, "tally", f"tally_{len(self.defined['tally'])}"
                        )
                        count += 1
                        continue

                    # Skip mcdc.run()
                    if attr_name == "run":
                        continue

                # Handle mcdc.simulation.set_root_universe(...) - nested attribute access
                if isinstance(func, ast.Attribute) and isinstance(
                    func.value, ast.Attribute
                ):
                    parent_attr = func.value.attr
                    child_attr = func.attr

                    # Capture set_root_universe as a 'hierarchy' type
                    if (
                        parent_attr == "simulation"
                        and child_attr == "set_root_universe"
                    ):
                        self.add_line(code_segment, "hierarchy", "root_universe")
                        count += 1
                        continue

            # Add everything else as 'other' (comments, math, etc)
            code_segment = get_segment(node)
            if code_segment and "mcdc.run" not in code_segment:
                self.add_line(code_segment, "other", "generic_code")

        return f"Successfully loaded {count} entities from {path.name}."

    def has_entity(self, entity_type: str, name: str) -> bool:
        return name in self.defined.get(entity_type, set())

    def undo_last(self) -> str:
        """Remove the most recently added entity."""
        if not self.entries:
            return "Nothing to undo."

        last_entry = self.entries.pop()
        # Use discard to avoid errors if key missing
        if last_entry["type"] in self.defined:
            self.defined[last_entry["type"]].discard(last_entry["name"])

        return f"Undid creation of {last_entry['type']} '{last_entry['name']}'."

    def delete_entity(self, entity_type: str, name: str) -> bool:
        """Delete a specific entity by name and type."""
        if not self.has_entity(entity_type, name):
            return False

        # Filter out the entry with matching type and name
        initial_count = len(self.entries)
        self.entries = [
            e
            for e in self.entries
            if not (e["type"] == entity_type and e["name"] == name)
        ]

        if len(self.entries) < initial_count:
            if entity_type in self.defined:
                self.defined[entity_type].discard(name)
            return True
        return False

    def replace_entity(self, entity_type: str, name: str, new_code: str) -> bool:
        """
        Replace an entity's code while preserving its position in the script.
        This is better than delete+recreate for debugging.
        Returns True if replacement was successful.
        """
        for entry in self.entries:
            if entry["type"] == entity_type and entry["name"] == name:
                entry["code"] = new_code
                self.reorder()
                return True
        return False

    def insert_entity(
        self,
        code: str,
        entity_type: str,
        name: str,
        before: str = None,
        after: str = None,
        position: int = None,
    ) -> bool:
        """
        Insert a new entity at a specific position in the script.
        """
        if self.has_entity(entity_type, name):
            return False

        new_entry = {"code": code, "type": entity_type, "name": name}

        insert_idx = None
        if position is not None:
            insert_idx = max(0, min(position, len(self.entries)))
        elif before:
            for idx, entry in enumerate(self.entries):
                if entry["name"] == before:
                    insert_idx = idx
                    break
        elif after:
            for idx, entry in enumerate(self.entries):
                if entry["name"] == after:
                    insert_idx = idx + 1
                    break

        if insert_idx is not None:
            self.entries.insert(insert_idx, new_entry)
        else:
            self.entries.append(new_entry)

        if entity_type not in self.defined:
            self.defined[entity_type] = set()
        self.defined[entity_type].add(name)

        self.reorder()
        return True

    def get_code_by_type(self, entity_type: str) -> str:
        lines = [e["code"] for e in self.entries if e["type"] == entity_type]
        if not lines:
            return "No entries defined."
        return "\n".join(lines)

    def reset(self):
        self.__init__()

    def get_dependencies(self, entity_name: str) -> List[str]:
        """
        Extract variable dependencies for a given entity.

        Parses the entity's code using AST to find all referenced variables.
        For cells, also extracts surface names from region strings.

        Args:
            entity_name: Name of the entity to analyze

        Returns:
            List of variable names that this entity depends on
        """
        # Find the entry
        entry = None
        for e in self.entries:
            if e["name"] == entity_name:
                entry = e
                break

        if entry is None:
            return []

        dependencies = set()
        code = entry["code"]

        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                    ref = node.id
                    # Only include if it's a defined entity (not builtins)
                    for entity_type, names in self.defined.items():
                        if ref in names and ref != entity_name:
                            dependencies.add(ref)
                            break
        except Exception:
            pass

        return list(dependencies)

    def get_state_summary(self) -> Dict[str, Any]:
        """
        Export a JSON-compatible summary of the current script state.

        Useful for LLM to "see" what has been built without reading raw code.

        Returns:
            Dictionary with structure:
            {
                "materials": ["fuel", "water"],
                "surfaces": ["s1 (PlaneX)", "s2 (CylinderZ)"],
                "cells": ["cell_1 (fill=fuel)"],
                "regions": ["inside_sphere", "inside_box"],
                "sources": ["source_0"],
                "tallies": ["tally_mesh"],
                "settings": ["N_particle", "N_batch"],
                "missing_requirements": ["source", "settings"],
                "entity_count": 10
            }
        """
        summary = {
            "materials": [],
            "surfaces": [],
            "cells": [],
            "regions": [],
            "universes": [],
            "lattices": [],
            "meshes": [],
            "sources": [],
            "tallies": [],
            "settings": [],
            "missing_requirements": [],
            "entity_count": len(self.entries),
        }

        for entry in self.entries:
            entry_type = entry["type"]
            name = entry["name"]
            code = entry["code"]

            if entry_type == "material":
                summary["materials"].append(name)
            elif entry_type == "surface":
                # Extract surface type
                surface_type = self._extract_surface_type(code)
                summary["surfaces"].append(f"{name} ({surface_type})")
            elif entry_type == "cell":
                # Extract fill
                fill = self._extract_fill(code)
                summary["cells"].append(f"{name} (fill={fill})")
            elif entry_type == "region":
                summary["regions"].append(name)
            elif entry_type == "universe":
                summary["universes"].append(name)
            elif entry_type == "lattice":
                summary["lattices"].append(name)
            elif entry_type == "mesh":
                summary["meshes"].append(name)
            elif entry_type == "source":
                summary["sources"].append(name)
            elif entry_type == "tally":
                summary["tallies"].append(name)
            elif entry_type == "settings":
                summary["settings"].append(name)

        # Check for missing requirements
        if not self.defined.get("material"):
            summary["missing_requirements"].append("material")
        if not self.defined.get("source"):
            summary["missing_requirements"].append("source")
        if not self.defined.get("settings"):
            summary["missing_requirements"].append("settings")

        return summary

    def _extract_surface_type(self, code: str) -> str:
        """Extract surface type from code (e.g., PlaneX, Sphere)."""
        for surface_type in [
            "PlaneX",
            "PlaneY",
            "PlaneZ",
            "CylinderX",
            "CylinderY",
            "CylinderZ",
            "Sphere",
        ]:
            if surface_type in code:
                return surface_type
        return "Surface"

    def _extract_fill(self, code: str) -> str:
        """Extract fill parameter value from cell code."""
        match = re.search(r"fill\s*=\s*(\w+)", code)
        if match:
            return match.group(1)
        return "unknown"

    def save_checkpoint(self, name: Optional[str] = None) -> str:
        """
        Save a checkpoint of the current state for potential rollback. Useful for batch execution

        Args:
            name: Optional name for the checkpoint

        Returns:
            Checkpoint identifier (index or name)
        """
        checkpoint = {
            "name": name or f"checkpoint_{len(self._checkpoints)}",
            "entries": copy.deepcopy(self.entries),
            "defined": copy.deepcopy(self.defined),
            "imports": copy.copy(self.imports),
        }
        self._checkpoints.append(checkpoint)
        return checkpoint["name"]

    def restore_checkpoint(self, name: Optional[str] = None) -> bool:
        """
        Restore state to a previous checkpoint.

        Args:
            name: Name of checkpoint to restore. If None, restores the most recent.

        Returns:
            True if restored successfully, False if checkpoint not found
        """
        if not self._checkpoints:
            return False

        # Find checkpoint by name or get the last one
        checkpoint = None
        if name is None:
            checkpoint = self._checkpoints.pop()
        else:
            for i, cp in enumerate(self._checkpoints):
                if cp["name"] == name:
                    checkpoint = cp
                    # Remove this and all later checkpoints
                    self._checkpoints = self._checkpoints[:i]
                    break

        if checkpoint is None:
            return False

        # Restore state
        self.entries = checkpoint["entries"]
        self.defined = checkpoint["defined"]
        self.imports = checkpoint["imports"]

        return True

    def clear_checkpoints(self):
        """Clear all saved checkpoints to free memory."""
        self._checkpoints = []

    def get_checkpoint_count(self) -> int:
        """Return the number of saved checkpoints."""
        return len(self._checkpoints)
