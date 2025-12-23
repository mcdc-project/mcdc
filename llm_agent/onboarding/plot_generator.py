import json
import ast
import re
from typing import Dict, Any, List
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


class PlotGenerator:
    def __init__(self, builder, llm):
        self.builder = builder
        self.llm = llm

    def _parse_kwargs(self, code_str: str) -> Dict[str, bool]:
        """Scans code for specific keyword arguments."""
        return {
            "has_time": "time=" in code_str,
            "has_energy": "energy=" in code_str,
            "has_mu": "mu=" in code_str,
            "has_azi": "azi=" in code_str,
        }

    def _extract_scores(self, code_str: str) -> List[str]:
        """Parses the python code string to find the 'scores' list."""
        try:
            tree = ast.parse(code_str)
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    for keyword in node.keywords:
                        if keyword.arg == "scores":
                            if isinstance(keyword.value, ast.List):
                                scores = []
                                # get all scores from the list
                                for elt in keyword.value.elts:
                                    if isinstance(elt, ast.Constant):
                                        scores.append(elt.value)
                                    elif isinstance(elt, ast.Str):
                                        scores.append(elt.s)
                                return scores
            return []
        except:
            return []

    def _determine_hdf5_path(self, code_str: str, entry_type: str, index: int) -> str:
        """
        Determines the correct HDF5 path by looking for an explicit 'name' arg.
        If missing, falls back to MCDC default naming conventions (mesh_tally_0, etc).
        """
        # Try to find explicit name="..."
        try:
            tree = ast.parse(code_str)
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    for keyword in node.keywords:
                        if keyword.arg == "name":
                            if isinstance(keyword.value, (ast.Constant, ast.Str)):
                                val = (
                                    keyword.value.value
                                    if isinstance(keyword.value, ast.Constant)
                                    else keyword.value.s
                                )
                                return f"tallies/{val}"
        except:
            pass

        # MCDC defaults based on type
        if "TallyMesh" in code_str:
            return f"tallies/mesh_tally_{index}"
        elif "TallySurface" in code_str:
            return f"tallies/surface_tally_{index}"
        elif "TallyCell" in code_str:
            return f"tallies/cell_tally_{index}"
        elif "TallyGlobal" in code_str:
            return f"tallies/global_tally_{index}"

        # fallback
        return f"tallies/tally_{index}"

    def _get_hdf5_schema(self) -> Dict[str, Any]:
        """Builds a map of the output.h5 structure."""
        schema = {}
        # get all tally entries
        tally_entries = [e for e in self.builder.entries if e["type"] == "tally"]

        # Separate counters for MCDC default naming logic
        type_counters = {
            "TallyMesh": 0,
            "TallySurface": 0,
            "TallyCell": 0,
            "TallyGlobal": 0,
        }

        for entry in tally_entries:
            name = entry["name"]  # Python variable name
            code = entry["code"]
            dims = self._parse_kwargs(code)
            scores = self._extract_scores(code)
            if not scores:
                scores = ["flux"]  # default to flux if no scores are specified

            # Identify Type for Counter
            current_type = "Unknown"
            if "TallyMesh" in code:
                current_type = "TallyMesh"
            elif "TallySurface" in code:
                current_type = "TallySurface"
            elif "TallyCell" in code:
                current_type = "TallyCell"
            elif "TallyGlobal" in code:
                current_type = "TallyGlobal"

            # Determine correct path
            hdf5_path = self._determine_hdf5_path(
                code, current_type, type_counters.get(current_type, 0)
            )

            # Increment counter for this type
            if current_type in type_counters:
                type_counters[current_type] += 1

            tally_info = {
                "python_var": name,
                "scores": scores,
                "dims": dims,
                "type": current_type,
                "hdf5_path": hdf5_path,
            }

            # Populate Grids
            tally_info["grids"] = []
            if current_type == "TallyMesh":
                tally_info["grids"] = ["x", "y", "z"]

            if dims["has_time"]:
                tally_info["grids"].append("time")
            if dims["has_energy"]:
                tally_info["grids"].append("energy")
            if dims["has_mu"]:
                tally_info["grids"].append("mu")

            schema[name] = tally_info

        return schema

    def generate_script(self, user_request: str = "") -> str:
        schema = self._get_hdf5_schema()

        system_prompt = """You are an expert Python developer for MCDC particle transport simulations.
Your task is to write a SPECIFIC, LINEAR post-processing script using `h5py` and `matplotlib`.

### GOAL
Write a concise script (~30-50 lines) that directly plots the data requested.
**DO NOT** write generic helper functions.
**DO NOT** write loops to search for data.
**HARDCODE** the paths based on the Schema below.

### HDF5 FILE STRUCTURE
The output file `output.h5` has this exact structure:
1.  **Grids:** `tallies/{{name}}/grid/{{grid_name}}`
    * Mesh Grids: `x`, `y`, `z`
    * Other: `time`, `energy`
2.  **Data:** `tallies/{{name}}/{{score}}/mean` and `tallies/{{name}}/{{score}}/sdev`
    * **CRITICAL:** Data is SQUEEZED.
    * Scalar (TallyCell): `shape=()` -> Use `[()]` to read.
    * 2D Mesh (XZ): `shape=(Nx, Nz)`
    * Time Series: `shape=(Nt,)`

### SCHEMA (Use these exact variables)
{schema}

### VALIDATION RULES
1.  **Check the Score:** The Schema lists the `scores` available for each tally (e.g., `['fission']`).
2.  **Ignore Invalid Requests:** If the user asks for "flux" but the Schema only has "fission", **YOU MUST USE "fission"**.
    * Add a comment: `# User requested flux, but only fission is available in output.h5`
3.  **No Hallucinations:** Do not try to access `std_dev_rel` or scores not in the Schema.

### PLOTTING LOGIC
1.  **Direct Execution:** Load data -> Calculate -> Plot.
2.  **Hardcoded Paths:** `x = f['tallies/mesh_tally_0/grid/x'][:]`
    * **IMPORTANT:** Use the `hdf5_path` from the schema (e.g., `tallies/mesh_tally_0`) NOT the python variable name.
3.  **Data Processing:**
    * Always compute midpoints: `x_mid = 0.5 * (x[1:] + x[:-1])`.
    * If using `pcolormesh`, use `data.T` (transpose) if needed.
    * Relative Error: `rel_err = sdev / mean` (handle divide-by-zero if needed).

### USER REQUEST
{request}

### OUTPUT
Return ONLY valid Python code. No markdown code blocks.
"""

        prompt = ChatPromptTemplate.from_messages(
            [("system", system_prompt), ("user", "Generate the processing script.")]
        )

        chain = prompt | self.llm | StrOutputParser()

        raw_output = chain.invoke(
            {"schema": json.dumps(schema, indent=2), "request": user_request}
        )

        clean_code = raw_output.strip()
        if clean_code.startswith("```"):
            clean_code = clean_code.split("\n", 1)[1]
        if clean_code.endswith("```"):
            clean_code = clean_code.rsplit("\n", 1)[0]

        return clean_code.strip()
