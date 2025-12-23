import json
import re
import pandas as pd
from pathlib import Path
from langchain.tools import tool
from .script_builder import ScriptBuilder
from typing import Any, Optional
from pydantic import BaseModel, Field


ONBOARDING_DIR = Path(__file__).parent


# Material Database for CE Materials


class MaterialDatabase:
    """Load and query nuclear/material data from CSV files."""

    def __init__(self):
        self.nuclear_data = None
        self.material_props = None
        self._load_data()

    def _load_data(self):
        """Load CSV files."""
        try:
            self.nuclear_data = pd.read_csv(ONBOARDING_DIR / "nuclear_data.csv")
            self.material_props = pd.read_csv(
                ONBOARDING_DIR / "material_properties.csv"
            )
        except Exception as e:
            print(f"Warning: Could not load material data: {e}")

    def lookup(self, query: str) -> str:
        """
        Look up material or isotope data.

        Args:
            query: Material name (e.g., 'UO2', 'water') or isotope (e.g., 'U235')

        Returns:
            Formatted string with data
        """
        query_lower = query.lower().strip()
        results = []

        # Search material_properties.csv by name or aliases
        if self.material_props is not None:
            for _, row in self.material_props.iterrows():
                name = str(row["name"]).lower()
                aliases = str(row.get("aliases", "")).lower()

                # Check if query matches name or any alias
                if query_lower == name or query_lower in aliases.split(";"):
                    results.append(self._format_material(row))

        # Search nuclear_data.csv for isotope
        if self.nuclear_data is not None:
            isotope_match = self.nuclear_data[
                self.nuclear_data["Isotope"].str.lower() == query_lower
            ]
            if not isotope_match.empty:
                results.append(self._format_isotope(isotope_match.iloc[0]))

            # Also search by element
            element_match = self.nuclear_data[
                self.nuclear_data["Element"].str.lower() == query_lower
            ]
            if not element_match.empty:
                results.append(self._format_element(element_match))

        if not results:
            # List available materials
            available = []
            if self.material_props is not None:
                available = list(self.material_props["name"].values)
            return f"No data found for '{query}'. Available materials: {', '.join(available[:10])}"

        return "\n\n".join(results)

    def _format_material(self, row) -> str:
        """Format material properties row."""
        name = row["name"]
        formula = row["formula"]
        density = row["density"]

        result = f"Material: {name}\n"
        result += f"Formula: {formula}\n"
        result += f"Density: {density} g/cm³\n\n"

        # Parse formula to get nuclide composition
        composition = self._parse_formula(formula)
        if composition:
            result += "Nuclide composition (for mcdc.Material):\n"
            result += "```python\n"
            result += f"{name} = mcdc.Material(\n"
            result += "    nuclide_composition={\n"
            for nuclide, fraction in composition.items():
                # Get atomic mass for number density calculation
                result += f"        '{nuclide}': {fraction:.6f},  # atom fraction\n"
            result += "    }\n"
            result += ")\n"
            result += "```\n"

        return result

    def _format_isotope(self, row) -> str:
        """Format single isotope data."""
        result = f"Isotope: {row['Isotope']}\n"
        result += f"Element: {row['Element']}\n"
        result += f"Atomic Mass: {row['Mass_u']:.6f} u\n"
        result += f"Natural Abundance: {row['Abundance']:.4%}\n"
        result += f"Dominant isotope: {'Yes' if row['Dominant'] == 1 else 'No'}\n"
        return result

    def _format_element(self, df) -> str:
        """Format all isotopes of an element."""
        element = df.iloc[0]["Element"]
        result = f"Element: {element} - All Isotopes\n"
        for _, row in df.iterrows():
            dominant = " (dominant)" if row["Dominant"] == 1 else ""
            result += f"  {row['Isotope']}: {row['Mass_u']:.4f} u, {row['Abundance']:.4%}{dominant}\n"
        return result

    def _parse_formula(self, formula: str) -> dict:
        """Parse chemical formula into nuclide fractions."""
        # Simple parser for formulas like H2O, UO2, etc.
        pattern = r"([A-Z][a-z]?)(\d*\.?\d*)"  # H2O -> (H, 2), (O, 1)
        matches = re.findall(pattern, formula)

        if not matches:
            return {}

        composition = {}
        total = 0

        for element, count_str in matches:
            if not element:
                continue
            count = float(count_str) if count_str else 1.0

            # Find dominant isotope for this element
            if self.nuclear_data is not None:
                dominant = self.nuclear_data[
                    (self.nuclear_data["Element"] == element)
                    & (self.nuclear_data["Dominant"] == 1)
                ]
                if not dominant.empty:
                    for _, row in dominant.iterrows():
                        isotope = row["Isotope"]
                        abundance = row["Abundance"]
                        composition[isotope] = count * abundance
                        total += count * abundance

        # Normalize to atom fractions
        if total > 0:
            composition = {k: v / total for k, v in composition.items()}

        return composition


# Pydantic Args Schemas


class WriteCodeArgs(BaseModel):
    code: str = Field(
        ...,
        description="Raw Python code for MCDC (e.g., 'absorber = mcdc.MaterialMG(capture=np.array([1.0]))')",
    )
    entity_type: str = Field(
        ...,
        description="Type: 'material', 'surface', 'region', 'cell', 'hierarchy', 'source', 'tally', 'settings'",
    )
    name: str = Field(..., description="Variable name being defined")


class ManageScriptArgs(BaseModel):
    action: str = Field(
        ...,
        description="'get' (view script), 'delete' (remove entity), 'clear' (reset)",
    )
    entity_type: Optional[str] = Field(
        None, description="Required for delete: entity type"
    )
    name: Optional[str] = Field(None, description="Required for delete: entity name")


class SearchExamplesArgs(BaseModel):
    query: str = Field(
        ..., description="Search query (e.g., 'sphere', 'slab', 'pin cell', 'lattice')"
    )


class SearchDocsArgs(BaseModel):
    query: str = Field(..., description="Search query for MCDC documentation")


class LookupMaterialArgs(BaseModel):
    query: str = Field(
        ...,
        description="Material name (e.g., 'UO2', 'water', 'ss_304') or isotope (e.g., 'U235', 'H1').",
    )


# --- Tool Factory ---


def get_mcdc_tools(builder: ScriptBuilder, retriever: Any):
    """
    Create the 5 MCDC tools.

    Args:
        builder: ScriptBuilder instance for script state
        retriever: Vectorstore retriever for documentation search

    Returns:
        List of 5 LangChain tools
    """

    # Initialize material database
    material_db = MaterialDatabase()

    @tool(args_schema=WriteCodeArgs)
    def write_code(code: str, entity_type: str, name: str) -> str:
        """
        Write raw MCDC Python code to the script.

        Use this to add materials, surfaces, regions, cells, sources, tallies, settings.
        The code should be valid Python that uses mcdc.* functions.

        EXAMPLES:
        - Material (MG): write_code(code="fuel = mcdc.MaterialMG(fission=np.array([0.2]), nu_p=np.array([1.2]), capture=np.array([0.3]))", entity_type="material", name="fuel")
        - Material (CE): write_code(code="fuel = mcdc.Material(nuclide_composition={'U235': 0.04, 'U238': 0.96, 'O16': 2.0})", entity_type="material", name="fuel")
        - Surface: write_code(code="sphere = mcdc.Surface.Sphere(center=[0,0,0], radius=2.0)", entity_type="surface", name="sphere")
        - Region: write_code(code="inside_sphere = -sphere", entity_type="region", name="inside_sphere")
        - Cell: write_code(code="fuel_cell = mcdc.Cell(region=inside_sphere, fill=fuel)", entity_type="cell", name="fuel_cell")
        - Hierarchy: write_code(code="universe = mcdc.Universe(cells=[fuel_cell, water_cell])", entity_type="hierarchy", name="lattice")
        - Source: write_code(code="mcdc.Source(x=[-0.1, 0.1], isotropic=True, energy_group=0)", entity_type="source", name="source")
        - Mesh: write_code(code="mesh = mcdc.MeshStructured(x=np.linspace(-10, 10, 201), z=np.linspace(-5, 5, 101))", entity_type="mesh", name="mesh")
        - Tally: write_code(code="mcdc.TallyMesh(mesh=mesh, scores=["fission"])", entity_type="tally", name="tally")
        - Settings: write_code(code="mcdc.settings.N_particle=1000", entity_type="settings", name="settings")

        KEY RULES:
        - For CE materials, ALWAYS use lookup_material() first to get exact nuclide compositions
        - scatter MUST be 2D: np.array([[1.0]]) not np.array([1.0])
        - Fission materials MUST have nu_p (for MG) or fissile isotopes (for CE)
        """
        valid_types = {
            "material",
            "surface",
            "region",
            "cell",
            "hierarchy",
            "source",
            "tally",
            "settings",
            "mesh",
        }
        if entity_type not in valid_types:
            return f"ERROR: Invalid entity_type '{entity_type}'. Must be one of: {valid_types}"

        # Check for duplicates
        if builder.has_entity(entity_type, name):
            if builder.replace_entity(entity_type, name, code):
                return f"Updated {entity_type} '{name}'"
            return f"ERROR: Could not update {entity_type} '{name}'"

        builder.add_line(code, entity_type, name)
        return f"Created {entity_type} '{name}'"

    @tool(args_schema=ManageScriptArgs)
    def manage_script(
        action: str, entity_type: Optional[str] = None, name: Optional[str] = None
    ) -> str:
        """
        Manage the MCDC script - view, delete entities, or clear.

        ACTIONS:
        - get: View current script and defined entities
        - delete: Remove a specific entity (requires entity_type and name)
        - clear: Reset script to empty state
        """
        if action == "get":
            summary = {k: list(v) for k, v in builder.defined.items() if v}
            return f"CURRENT SCRIPT:\n{builder.get_script()}\n\nDEFINED ENTITIES:\n{json.dumps(summary, indent=2)}"

        elif action == "delete":
            if not entity_type or not name:
                return "ERROR: delete requires entity_type and name"
            if builder.delete_entity(entity_type, name):
                return f"Deleted {entity_type} '{name}'"
            return f"ERROR: Entity not found: {entity_type} '{name}'"

        elif action == "clear":
            builder.reset()
            return "Script cleared"

        return f"ERROR: Unknown action '{action}'. Use 'get', 'delete', or 'clear'."

    @tool(args_schema=SearchExamplesArgs)
    def search_examples(query: str) -> str:
        """
        Search example MCDC scripts for reference.

        Returns example prompts and the tool_calls to create them.
        Use when uncertain about geometry, materials, or tallies.
        """
        examples_dir = ONBOARDING_DIR / "examples"
        if not examples_dir.exists():
            return "ERROR: Examples directory not found"

        query_lower = query.lower()
        matches = []
        for json_file in examples_dir.glob("*.json"):
            try:
                data = json.loads(json_file.read_text())
                name = data.get("name", "").lower()
                prompt = data.get("prompt", "").lower()

                score = 0
                if query_lower in name:
                    score += 2
                if query_lower in prompt:
                    score += 1

                if score > 0:
                    matches.append((score, data))
            except Exception:
                continue

        if not matches:
            available = [f.stem for f in examples_dir.glob("*.json")]
            return f"No examples found for '{query}'. Available: {', '.join(available)}"

        matches.sort(key=lambda x: -x[0])
        best = matches[0][1]

        result = f"=== Example: {best['name']} ===\n"
        result += f"Prompt: {best['prompt']}\n\n"
        result += "Tool calls:\n"
        for i, call in enumerate(best.get("tool_calls", []), 1):
            args = call.get("args", {})
            args_str = ", ".join(
                f'{k}="{v}"' if isinstance(v, str) else f"{k}={v}"
                for k, v in args.items()
            )
            result += f"{i}. {call['tool']}({args_str})\n"

        return result

    @tool(args_schema=SearchDocsArgs)
    def search_docs(query: str) -> str:
        """
        Search MCDC documentation for API reference.

        Use for questions about MCDC functions, parameters, and usage.
        """
        try:
            docs = retriever.invoke(query)
            formatted = []
            for i, doc in enumerate(docs, 1):
                source = doc.metadata.get("source", "Unknown").replace(
                    "llm_agent/corpus/", ""
                )
                formatted.append(f"[Source {i}: {source}]\n{doc.page_content}")
            return "\n\n---\n\n".join(formatted) if formatted else "No results found."
        except Exception as e:
            return f"Error searching docs: {e}"

    @tool(args_schema=LookupMaterialArgs)
    def lookup_material(query: str) -> str:
        """
        Look up nuclear data for continuous energy (CE) materials.

        REQUIRED before creating any mcdc.Material() with nuclide_composition.
        Returns exact nuclide compositions and atomic masses from reference data.

        EXAMPLES:
        - lookup_material(query='UO2') -> uranium dioxide composition
        - lookup_material(query='water') -> H2O composition
        - lookup_material(query='U235') -> U-235 atomic mass and abundance
        - lookup_material(query='ss_304') -> stainless steel 304 composition

        IMPORTANT: Always use this tool for CE materials. Do NOT guess nuclide compositions.
        """
        return material_db.lookup(query)

    @tool
    def search_api(query: str) -> str:
        """
        Search MCDC API reference for correct function signatures.

        ALWAYS use this FIRST when unsure about parameter names or syntax.
        Returns exact signatures for MCDC classes and functions.

        EXAMPLES:
        - search_api("TallyCell") -> shows correct parameter: cell= not cells=
        - search_api("CylinderZ") -> shows center is 2D: [x, y] not [x,y,z]
        - search_api("MaterialMG") -> shows all parameters and physics rules
        - search_api("Source") -> shows source definition syntax
        """
        api_file = ONBOARDING_DIR / "mcdc_api_reference.md"
        if not api_file.exists():
            return "API reference file not found"

        content = api_file.read_text()
        query_lower = query.lower()

        # Find relevant sections
        sections = content.split("## ")
        matches = []

        for section in sections:
            if query_lower in section.lower():
                # Clean up and limit length
                section_text = "## " + section if section else section
                if len(section_text) > 800:
                    section_text = section_text[:800] + "..."
                matches.append(section_text)

        if not matches:
            # If no section match, search line by line
            lines = content.split("\n")
            relevant = []
            for i, line in enumerate(lines):
                if query_lower in line.lower():
                    # Include context (2 lines before, 5 after)
                    start = max(0, i - 2)
                    end = min(len(lines), i + 6)
                    relevant.extend(lines[start:end])
                    relevant.append("---")

            if relevant:
                return "\n".join(relevant[:50])  # Limit output
            return f"No API reference found for '{query}'. Try: MaterialMG, Source, Cell, Surface, TallyCell, TallyMesh"

        return "\n".join(matches[:3])  # Top 3 matches

    return [
        write_code,
        manage_script,
        search_examples,
        search_docs,
        lookup_material,
        search_api,
    ]
