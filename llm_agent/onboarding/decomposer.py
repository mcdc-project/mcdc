from typing import List, Dict, Any
import json
import re
from dataclasses import dataclass, asdict
from enum import Enum
from .patterns import detect_patterns, get_pattern_guidance


class WorkflowStep(Enum):
    """Valid workflow step types in order of execution."""

    MATERIAL = "material"
    SURFACE = "surface"
    CELL = "cell"
    HIERARCHY = "hierarchy"
    SOURCE = "source"
    TALLY = "tally"
    SETTINGS = "settings"


@dataclass
class DecomposedTask:
    """Represents a single decomposed task."""

    step: str
    instruction: str
    priority: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# Step execution order
STEP_ORDER = ["material", "surface", "cell", "hierarchy", "source", "tally", "settings"]


class Decomposer:
    """
    Decomposes complex user prompts into MCDC workflow steps using LLM.

    Usage:
        decomposer = Decomposer(llm)
        tasks = decomposer.decompose("Create a sphere of fuel inside water...")
    """

    PROMPT = """Break this simulation request into ordered workflow steps.

## Steps (in order):
- material: Define materials (fuel, water, etc.)
- surface: Create geometric surfaces (planes, spheres, cylinders)
- cell: Define cells with regions and material fills
- hierarchy: Universes/lattices (ONLY if copying/translating geometry)
- source: Define particle source
- tally: Define tallies for scoring
- settings: Simulation parameters (N_particle, etc.)

## Rules:
1. ONE task per step type (combine all materials into one task, etc.)
2. Skip hierarchy unless universes/lattices/translations are needed
3. Use complement (~) for nested cells to avoid overlap

{pattern_context}

## Example:
Request: "Sphere of fuel in water cube, tally fission"
```json
[
  {{"step": "material", "instruction": "Create fuel material and water material"}},
  {{"step": "surface", "instruction": "Create sphere at origin; 6 bounding planes for cube with vacuum BCs"}},
  {{"step": "cell", "instruction": "Create fuel_cell inside sphere; water_cell in box but outside sphere using ~"}},
  {{"step": "source", "instruction": "Isotropic source in fuel region"}},
  {{"step": "tally", "instruction": "Mesh tally for fission"}},
  {{"step": "settings", "instruction": "Set N_particle=1000, N_batch=2"}}]
```

## Request:
{user_prompt}

## Response (JSON array only):"""

    def __init__(self, llm: Any):
        self.llm = llm

    def decompose(self, user_prompt: str) -> List[DecomposedTask]:
        """Decompose a user prompt into workflow tasks using LLM."""
        return self._llm_decompose(user_prompt)

    def _llm_decompose(self, user_prompt: str) -> List[DecomposedTask]:
        """Use LLM to decompose the prompt with pattern-aware context."""
        # Detect relevant geometry patterns
        detected = detect_patterns(user_prompt)
        pattern_context = get_pattern_guidance(detected) if detected else ""

        prompt = self.PROMPT.format(
            user_prompt=user_prompt, pattern_context=pattern_context
        )
        response = self.llm.invoke(prompt)

        # Extract content
        content = response.content if hasattr(response, "content") else str(response)
        if isinstance(content, list):
            content = "\n".join(
                b.get("text", "") if isinstance(b, dict) else str(b) for b in content
            )

        # Parse JSON
        return self._parse_json(content)

    def _parse_json(self, content: str) -> List[DecomposedTask]:
        """Parse JSON from LLM response."""
        # Clean markdown
        content = re.sub(r"```json\s*", "", content)
        content = re.sub(r"```\s*", "", content)

        # Find JSON array
        match = re.search(r"\[\s*\{[\s\S]*\}\s*\]", content)
        if not match:
            raise ValueError("No JSON array found")

        data = json.loads(match.group())

        tasks = []
        for item in data:
            if isinstance(item, dict) and "step" in item and "instruction" in item:
                tasks.append(
                    DecomposedTask(
                        step=item["step"].lower(),
                        instruction=item["instruction"],
                        priority=(
                            STEP_ORDER.index(item["step"].lower())
                            if item["step"].lower() in STEP_ORDER
                            else 99
                        ),
                    )
                )

        tasks.sort(key=lambda t: t.priority)
        return tasks

    def _validate_and_order(self, tasks: List[DecomposedTask]) -> List[DecomposedTask]:
        """Validate tasks and sort by workflow order. Used by tests."""
        # Assign priority based on workflow order
        for task in tasks:
            if task.step in STEP_ORDER:
                task.priority = STEP_ORDER.index(task.step)
            else:
                task.priority = 99

        # Sort by priority
        tasks.sort(key=lambda t: t.priority)

        # Ensure minimum required steps
        existing_steps = {t.step for t in tasks}
        required = ["material", "surface", "cell", "source", "tally", "settings"]

        for req in required:
            # catch any missing steps and add them
            if req not in existing_steps:
                tasks.append(
                    DecomposedTask(
                        step=req,
                        instruction=f"Define {req} based on simulation requirements",
                        priority=STEP_ORDER.index(req) if req in STEP_ORDER else 99,
                    )
                )

        tasks.sort(key=lambda t: t.priority)
        return tasks

    def get_plan_summary(self, tasks: List[DecomposedTask]) -> str:
        """Generate human-readable summary."""
        lines = ["Decomposed Plan:", "-" * 40]
        for i, task in enumerate(tasks, 1):
            lines.append(f"{i}. [{task.step.upper()}] {task.instruction}")
        lines.append(f"Total steps: {len(tasks)}")
        return "\n".join(lines)


def decompose_prompt(prompt: str, llm: Any = None) -> List[Dict[str, str]]:
    """Convenience function to decompose a prompt."""
    decomposer = Decomposer(llm)
    tasks = decomposer.decompose(prompt)
    return [task.to_dict() for task in tasks]
