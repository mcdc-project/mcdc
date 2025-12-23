from llm_agent.onboarding.concepts import CONCEPT_LESSONS
from llm_agent.onboarding.debugger import DebugHandler
from llm_agent.onboarding.plot_generator import PlotGenerator
from llm_agent.onboarding.visualizer import GeometryVisualizer
from llm_agent.onboarding.batch_executor import BatchExecutor
from llm_agent.utils import (
    load_llm,
    load_retriever,
    create_rag_chain_with_prompt,
    parse_agent_response,
)
from llm_agent.onboarding.script_builder import ScriptBuilder
from llm_agent.onboarding.tools import get_mcdc_tools
from llm_agent.onboarding.decomposer import Decomposer
from langchain.agents import create_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from pathlib import Path
import subprocess
import ast
import argparse
import logging

# UI Imports
from prompt_toolkit import PromptSession
from prompt_toolkit.styles import Style as PromptStyle
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.keys import Keys
from rich.console import Console
from rich.markdown import Markdown
from rich.syntax import Syntax
from rich.panel import Panel
from rich.theme import Theme
from rich.status import Status
from rich.prompt import Confirm

# Define a custom theme for Rich
custom_theme = Theme(
    {
        "info": "dim cyan",
        "warning": "yellow",
        "error": "bold red",
        "success": "bold green",
        "agent": "bold blue",
        "user": "bold magenta",
        "step": "magenta bold reverse",
        "code": "bold white",
        "markdown.code": "bold dark_green",
        "code": "bold dark_green",
    }
)

console = Console(theme=custom_theme)


class BackToMenu(Exception):
    """Raised when the user presses ESC to return to the main menu."""

    pass


class MCDCAgent:
    """
    Interactive agent for learning MCDC through an 8-step workflow.
    """

    STEP_KEYWORDS = {
        "material": "MaterialMG Material capture scatter fission nuclide_composition density multi-group continuous-energy cross-section macroscopic",
        "surface": "Surface PlaneX PlaneY PlaneZ CylinderX CylinderY CylinderZ Sphere boundary_condition vacuum reflective interface geometry normal",
        "cell": "Cell region fill boolean operators intersection union & |",
        "hierarchy": "Universe Lattice fill translation rotation repetition grid 3D array hexagonal square",
        "source": "Source position energy direction isotropic white_direction time energy_group spectrum point_source volume_source uniform",
        "tally": "TallyMesh TallyCell TallySurface scores flux collision fission net-current mesh energy_bins detector mu_bins",
        "settings": "settings N_particle N_batch eigenmode census output population_control variance_reduction convergence active inactive",
        "run": "run execute simulate output h5 tally results k-effective convergence statistics batch cycle",
    }

    def __init__(self, llm):
        self.builder = ScriptBuilder()

        # specific style for the input prompt cursor
        self.prompt_style = PromptStyle.from_dict(
            {
                "prompt": "#00aa00 bold",  # Green bold prompt
            }
        )
        self.session = PromptSession()

        self.kb = KeyBindings()

        @self.kb.add(Keys.Escape)
        def _(event):
            event.app.exit(result="__ESCAPE__")

        self.doc_filter = {"type": {"$nin": ["paper", "source_code", "internal_code"]}}

        # Set up retriever and RAG chain for Q&A
        self.retriever = load_retriever(k=5, search_filter=self.doc_filter)

        rag_prompt = """You are MCDC-Agent, explaining Monte Carlo particle transport concepts to new users.
Question: {question}
Documentation: {context}
Provide a helpful answer that includes:
1. A simple code example (if relevant)
2. Clear explanation of key concepts (assume little to no nuclear physics background)
3. Brief description of required parameters if applicable
Use friendly, educational tone. Keep it concise but informative.
Answer:"""
        self.rag_prompt = rag_prompt
        self.rag_chain = create_rag_chain_with_prompt(llm, self.retriever, rag_prompt)

        try:
            self.tools = get_mcdc_tools(self.builder, self.retriever)
            self.llm = llm

            system_prompt = """You are MCDC-Agent, creating Monte Carlo simulations with MCDC.

## EXAMPLE: Sphere in water cube
User: "Sphere of multigroup fuel (radius 1.5, center 2,2,2) in scattering cube (0-4), tally fission"

Tool calls:
1. write_code(code="pure_f = mcdc.MaterialMG(fission=np.array([0.05]), nu_p=np.array([2.5]), capture=np.array([0.15]))", entity_type="material", name="pure_f")
2. write_code(code="pure_s = mcdc.MaterialMG(scatter=np.array([[0.9]]), capture=np.array([0.1]))", entity_type="material", name="pure_s")
3. write_code(code="sx1 = mcdc.Surface.PlaneX(x=0.0, boundary_condition='vacuum')", entity_type="surface", name="sx1")
4. write_code(code="sx2 = mcdc.Surface.PlaneX(x=4.0, boundary_condition='vacuum')", entity_type="surface", name="sx2")
5. write_code(code="sy1 = mcdc.Surface.PlaneY(y=0.0, boundary_condition='vacuum')", entity_type="surface", name="sy1")
6. write_code(code="sy2 = mcdc.Surface.PlaneY(y=4.0, boundary_condition='vacuum')", entity_type="surface", name="sy2")
7. write_code(code="sz1 = mcdc.Surface.PlaneZ(z=0.0, boundary_condition='vacuum')", entity_type="surface", name="sz1")
8. write_code(code="sz2 = mcdc.Surface.PlaneZ(z=4.0, boundary_condition='vacuum')", entity_type="surface", name="sz2")
9. write_code(code="sphere = mcdc.Surface.Sphere(center=[2.0, 2.0, 2.0], radius=1.5)", entity_type="surface", name="sphere")
10. write_code(code="inside_sphere = -sphere", entity_type="region", name="inside_sphere")
11. write_code(code="inside_box = +sx1 & -sx2 & +sy1 & -sy2 & +sz1 & -sz2", entity_type="region", name="inside_box")
12. write_code(code="mcdc.Cell(region=inside_box & ~inside_sphere, fill=pure_s)", entity_type="cell", name="outer_cell")
13. write_code(code="sphere_cell = mcdc.Cell(region=inside_sphere, fill=pure_f)", entity_type="cell", name="sphere_cell")
14. write_code(code="mcdc.Source(x=[-1.0, 1.0], isotropic=True, energy_group=0)", entity_type="source", name="source")
15. write_code(code="mesh = mcdc.MeshUniform(x=(0.0, 4.0, 4), y=(0.0, 4.0, 4), z=(0.0, 4.0, 4))", entity_type="tally", name="mesh")
16. write_code(code="mcdc.TallyMesh(mesh=mesh, scores=['fission'])", entity_type="tally", name="tally")
17. write_code(code="mcdc.settings.N_particle = 1000\\nmcdc.settings.N_batch = 2", entity_type="settings", name="settings")

## KEY RULES - GEOMETRY
- Regions: +surface (outside), -surface (inside), & (and), | (or), ~ (complement)
- scatter MUST be 2D: np.array([[1.0]]) not np.array([1.0])

## KEY RULES - PHYSICS (CRITICAL!)
- For SUBCRITICAL systems: fission * nu_p < capture + scatter (prevents particle bank overflow)
- Example safe values: fission=0.05, nu_p=2.5, capture=0.15 gives 0.125 < 0.15 ✓
- If you don't follow this rule, simulation will crash with "Particle active bank is full"
- Default to using multigroup physics (MaterialMG) unless otherwise specified

## KEY RULES - SOURCE/MESH/PARTICLE SCALING
Analyze the simulation geometry before setting N_particle:
- POINT/SMALL SOURCE (e.g., x=[-0.1, 0.1]): N_particle >= 1000 is sufficient
- LARGE/VOLUMETRIC SOURCE (covers significant geometry): N_particle >= Total_Mesh_Cells
- DEFAULT: Use a small, localized source unless the user requests volume-filling source
- MESH RESOLUTION: Keep mesh coarse. Total cells < 10000 to avoid variance errors
- Example: MeshUniform(x=(0, 4, 10), y=(0, 4, 10), z=(0, 4, 10)) = 100 cells, needs N_particle >= 1000

## TOOLS
1. write_code(code, entity_type, name) - Add MCDC code
2. manage_script(action) - View/delete/clear script
3. search_examples(query) - Find example code
4. search_docs(query) - Search documentation
5. lookup_material(query) - Get nuclide data for CE materials
6. search_api(query) - FIRST CHOICE for syntax/parameters (e.g., "TallyCell", "CylinderZ")
"""

            self.agent = create_agent(
                model=self.llm, tools=self.tools, system_prompt=system_prompt
            )
        except Exception as e:
            console.print(
                f"[error]Warning: Failed to initialize agent properly: {e}[/error]"
            )
            raise

    def get_input(self, prompt_text=""):
        """
        Input handler
        """
        if prompt_text:
            console.print(f"[user]{prompt_text}[/user]")

        # Get user input
        result = self.session.prompt(
            [("class:prompt", "> ")], style=self.prompt_style, key_bindings=self.kb
        )

        # Check if the user pressed Escape
        if result == "__ESCAPE__":
            raise BackToMenu()

        return result.strip()

    def expand_query(self, query: str, step: str) -> str:
        """
        Expand query with step-specific keywords, to help with RAG performance
        """
        base_query = f"{step} {query}"
        keywords = self.STEP_KEYWORDS.get(step, "")
        return f"{base_query} {keywords}"

    def get_step_retriever(self, step: str):
        """Get a retriever filtered by step/section. Falls back to default retriever."""
        try:
            step_filter = {"$and": [self.doc_filter, {"section": step}]}
            return load_retriever(k=5, search_filter=step_filter)
        except Exception:
            return self.retriever

    def teach_concept(self, step: str) -> bool:
        lesson = CONCEPT_LESSONS.get(step)
        if not lesson:
            console.print(f"[error]No lesson found for {step}[/error]")
            return False
        console.print("\n")
        console.rule(f"[step] STEP: {step.upper()} [/step]")

        # Concept Block
        console.print(
            Panel(Markdown(lesson["concept"]), title="Concept", border_style="magenta")
        )

        # Syntax
        if "syntax" in lesson:
            console.print("\n[bold]Syntax Template:[/bold]")
            syntax_highlighted = Syntax(
                lesson["syntax"],
                "python",
                theme="monokai",
                line_numbers=False,
                word_wrap=True,
            )
            console.print(syntax_highlighted)

        # Key Parts
        console.print("\n[bold]Parameters:[/bold]")
        console.print(Markdown(lesson["parts"]))

        # Tips
        if "tips" in lesson:
            console.print("\n[bold yellow]Tips & Common Mistakes:[/bold yellow]")
            for tip in lesson["tips"]:
                console.print(Markdown(f"* {tip}"))

        step_rag_chain = create_rag_chain_with_prompt(
            self.llm, self.get_step_retriever(step), self.rag_prompt
        )

        # Question loop
        while True:
            q = self.get_input("Ask a question (press enter to continue):")
            if not q:
                break

            try:
                expanded_q = self.expand_query(q, step)
                console.print(f"\n[agent]AGENT:[/agent] ", end="")

                # Use status spinner while waiting for response
                with console.status("[bold blue]Thinking...", spinner="dots"):
                    response = ""
                    for chunk in step_rag_chain.stream(expanded_q):
                        response += str(chunk)

                console.print(Markdown(response))
                console.print("")

            except Exception as e:
                console.print(f"\n[error]Error: {e}[/error]")

        console.print("")
        return Confirm.ask(f"[bold]Ready to create your {step}?[/bold]", default=True)

    def _parse_agent_response(self, response) -> str:
        """Wrapper function for parse_agent_response"""
        return parse_agent_response(response)

    def _print_script(self):
        console.print("\n")
        console.rule("[bold cyan]CURRENT SCRIPT[/bold cyan]")
        script_content = self.builder.get_script()
        syntax = Syntax(
            script_content, "python", theme="monokai", line_numbers=True, word_wrap=True
        )
        console.print(syntax)
        console.rule("[bold cyan]END SCRIPT[/bold cyan]")
        console.print("\n")

    def execute_batch(
        self, plan: list, max_iterations: int = 5, enable_validation: bool = True
    ) -> str:
        """
        Execute a decomposed plan in batch mode without user interaction.

        Delegates to BatchExecutor for the actual execution logic.

        Args:
            plan: List of dicts with 'step' and 'instruction' keys
            max_iterations: Max tool call attempts per step (default 5)
            enable_validation: Run dry-run validation after generation (default True)

        Returns:
            Generated script as string
        """
        executor = BatchExecutor(
            agent=self.agent,
            builder=self.builder,
            console=console,
            response_parser=self._parse_agent_response,
            enable_validation=enable_validation,
        )
        return executor.execute(plan, max_iterations)

    def _handle_view_mode(self):
        """Handle view mode, where user can view and edit the script"""
        while True:
            # Show Script & Prompt
            self._print_script()
            console.print(
                "[bold]View Mode:[/bold] Press [green]Enter[/green] to return, type [cyan]viz[/cyan] to visualize geometry, or type an instruction to edit."
            )
            command = self.get_input()

            if not command:
                console.print("Returning to menu...")
                break

            # Handle 'viz' command for geometry visualization
            if command.lower().startswith("viz"):
                import re

                axis = "z"
                position = 0.0
                # Parse: "viz z 5", "viz z=5", "viz z=5.0", etc.
                match = re.search(r"([xyz])\s*=?\s*([-\d.]+)", command.lower())
                if match:
                    axis = match.group(1)
                    try:
                        position = float(match.group(2))
                    except ValueError:
                        pass
                self.run_visualization(axis=axis, position=position)
                continue

            current_context = self.builder.get_script()

            messages = [
                {
                    "role": "user",
                    "content": (
                        f"Here is the current MCDC script:\n```python\n{current_context}\n```\n\n"
                        f"EDIT/ADD TASK: {command}\n"
                        f"IMPORTANT: You are editing/adding to an existing script. Use the variable names shown above."
                    ),
                }
            ]

            # Execution Loop (Handles retries and questions)
            # We loop until the task is marked complete or cancelled
            task_complete = False

            while not task_complete:

                # Attempt Loop (Handles "laziness" where agent doesn't act)
                for attempt in range(3):
                    try:
                        with console.status(
                            "[bold yellow]Processing edit...", spinner="dots"
                        ):
                            # Snapshot before execution
                            script_before = self.builder.get_script()
                            response = self.agent.invoke({"messages": messages})
                            script_after = self.builder.get_script()

                        output = self._parse_agent_response(response)
                        if not output or not output.strip():
                            output = "(No text response provided by Agent)"

                        # Check outcomes
                        script_changed = script_before != script_after

                        clarification_phrases = [
                            "should that be",
                            "what",
                            "which",
                            "i suggest",
                            "i recommend",
                            "does this look correct",
                            "would you like",
                            "do you want",
                            "can you confirm",
                            "how about",
                            "unable to",
                            "cannot create",
                            "please specify",
                            "?",
                            "propose",
                            "intend to",
                            "clarify",
                            "confirm",
                        ]
                        is_question = any(
                            phrase in output.lower() for phrase in clarification_phrases
                        )

                        # SUCCESS (Script Changed)
                        if script_changed:
                            self.builder.reorder()
                            console.print(
                                Panel(
                                    Markdown(output),
                                    title="Agent (Executed)",
                                    border_style="green",
                                )
                            )
                            task_complete = True
                            break

                        # QUESTION (Agent needs info)
                        if is_question:
                            console.print(
                                Panel(
                                    Markdown(output),
                                    title="Agent (Question)",
                                    border_style="blue",
                                )
                            )
                            user_reply = self.get_input(
                                "Response (or Enter to cancel):"
                            )

                            if not user_reply:
                                console.print("Edit cancelled.")
                                task_complete = True
                                break  # Cancel task

                            # Add interaction to history
                            messages.append({"role": "assistant", "content": output})

                            # Inject force-execute instruction if confirmed
                            confirmation_keywords = [
                                "yes",
                                "y",
                                "ok",
                                "okay",
                                "sure",
                                "correct",
                                "go ahead",
                            ]
                            if user_reply.strip().lower() in confirmation_keywords:
                                user_reply += " (SYSTEM: The user confirmed. EXECUTE the required tool calls IMMEDIATELY.)"

                            messages.append({"role": "user", "content": user_reply})
                            break

                        # FAILURE (No Change, No Question)
                        # The agent probably just talked without acting. Force retry.
                        if attempt < 2:
                            console.print(
                                f"[dim red]System: Agent returned text but did not execute tools. Retrying ({attempt+1}/3)...[/dim red]"
                            )
                            messages.append({"role": "assistant", "content": output})
                            messages.append(
                                {
                                    "role": "user",
                                    "content": "SYSTEM ERROR: You responded with text but DID NOT execute any tools. The script has NOT changed. You must CALL the functions (e.g., create_surface, replace_entity) to perform the task.",
                                }
                            )
                            continue  # Next attempt
                        else:
                            console.print(
                                Panel(
                                    Markdown(output),
                                    title="Agent (Failed)",
                                    border_style="red",
                                )
                            )
                            console.print(
                                "[error]Agent failed to execute tools after multiple attempts.[/error]"
                            )
                            task_complete = True
                            break

                    except Exception as e:
                        console.print(f"[error]Error: {e}[/error]")
                        task_complete = True
                        break

    def run_visualization(self, axis: str = "z", position: float = 0.0):
        """
        Smart Visualizer:
        - Extracts 'region' strings directly from ScriptBuilder history.
        - Supports slicing along X, Y, or Z axis.

        Delegates to GeometryVisualizer for the actual visualization logic.
        """
        visualizer = GeometryVisualizer(self.builder, console)
        visualizer.run(axis=axis, position=position)

    def create_step(self, step: str) -> str:
        console.print("\n")
        console.rule(f"[bold]CREATE YOUR {step.upper()}[/bold]")

        while True:
            # Get User Input
            prompt_text = f"Describe {step}(s) to add, 'view' to edit, or 'viz' to plot"
            if step == "surface":
                prompt_text += " (e.g., 'sphere radius 5')"
            elif step == "material":
                prompt_text += " (e.g., 'water')"

            goal = self.get_input(f"{prompt_text} (or Enter to finish):")

            if not goal:
                console.print(f"[success]Finished defining {step}s.[/success]")
                break

            # Handle Special Commands (Viz, View, Undo)
            if goal.lower().startswith("viz"):
                # Check if we are doing 2D Slicing (Cells exist) or 3D Wireframe
                has_cells = len(self.builder.defined.get("cell", set())) > 0
                viz_axis = "z"
                viz_pos = 0.0

                if has_cells:
                    slice_input = self.get_input(
                        "Enter slice (e.g., 'z=5', 'y=0') [Default: z=0]:"
                    )
                    if slice_input:
                        import re

                        match = re.search(
                            r"([xyz])\s*=?\s*([-\d.]+)", slice_input.lower()
                        )
                        if match:
                            viz_axis, viz_pos = match.group(1), float(match.group(2))

                self.run_visualization(axis=viz_axis, position=viz_pos)
                continue

            if goal.lower() == "view":
                self._handle_view_mode()
                continue

            if goal.lower() == "undo":
                result = self.builder.undo_last()
                console.print(f"[warning]{result}[/warning]")
                continue

            # Setup Agent Context
            current_context = self.builder.get_script()
            messages = [
                {
                    "role": "user",
                    "content": (
                        f"Here is the current MCDC script:\n```python\n{current_context}\n```\n\n"
                        f"TASK: Create/Modify {step}(s) based on this description: {goal}\n"
                        f"IMPORTANT: Use existing variable names from the script where appropriate."
                    ),
                }
            ]

            # Execution Loop (Similar to View Mode)
            task_complete = False

            while not task_complete:

                # Attempt Loop (Handles "laziness" or failures)
                for attempt in range(3):
                    try:
                        with console.status(
                            f"[bold yellow]Drafting {step} (Attempt {attempt+1})...[/bold yellow]",
                            spinner="dots",
                        ):
                            # Snapshot before
                            script_before = self.builder.get_script()

                            response = self.agent.invoke({"messages": messages})
                            output = self._parse_agent_response(response)

                            # Snapshot after
                            script_after = self.builder.get_script()

                        if not output or not output.strip():
                            output = "(No text response provided by Agent)"

                        # Did the script change?
                        script_changed = script_before != script_after

                        # Detect Question/Error
                        clarification_phrases = [
                            "should that be",
                            "what",
                            "which",
                            "i suggest",
                            "i recommend",
                            "does this look correct",
                            "would you like",
                            "do you want",
                            "can you confirm",
                            "how about",
                            "unable to",
                            "cannot create",
                            "please specify",
                            "?",
                            "propose",
                            "intend to",
                            "clarify",
                            "confirm",
                        ]
                        is_question = any(
                            phrase in output.lower() for phrase in clarification_phrases
                        )
                        is_error = "ERROR" in output or "exception" in output.lower()

                        # SUCCESS
                        if script_changed:
                            # Reorder dependencies to be safe
                            self.builder.reorder()

                            console.print(
                                Panel(
                                    Markdown(output),
                                    title="[bold green]SUCCESS[/bold green]",
                                    border_style="green",
                                )
                            )

                            # Show the specific code block for this step type
                            new_code = self.builder.get_code_by_type(step)
                            console.print(
                                f"\n[bold]Current {step.upper()} definitions:[/bold]"
                            )
                            console.print(Syntax(new_code, "python", theme="monokai"))

                            task_complete = True
                            break  # Exit attempt loop

                        # QUESTION OR AGENT ERROR
                        if is_question or is_error:
                            border_color = "red" if is_error else "blue"
                            title = "ERROR" if is_error else "QUESTION"
                            console.print(
                                Panel(
                                    Markdown(output),
                                    title=f"[bold {border_color}]{title}[/bold {border_color}]",
                                    border_style=border_color,
                                )
                            )

                            user_reply = self.get_input(
                                "Your answer (or Enter to cancel):"
                            )

                            if not user_reply:
                                console.print("[dim]Action cancelled.[/dim]")
                                task_complete = True
                                break

                            # Append history and loop back to 'while not task_complete'
                            messages.append({"role": "assistant", "content": output})

                            confirmation_keywords = [
                                "yes",
                                "y",
                                "ok",
                                "okay",
                                "sure",
                                "correct",
                                "go ahead",
                            ]
                            if user_reply.strip().lower() in confirmation_keywords:
                                user_reply += " (SYSTEM: The user confirmed. EXECUTE the required tool calls IMMEDIATELY.)"

                            messages.append({"role": "user", "content": user_reply})
                            break  # Break attempt loop to restart invoke with new messages

                        # LAZINESS (Text but no Action)
                        # The agent talked but didn't call a tool, and didn't ask a question.
                        if attempt < 2:
                            console.print(
                                f"[dim red]System: Agent returned text but did not execute tools. Retrying ({attempt+1}/3)...[/dim red]"
                            )
                            messages.append({"role": "assistant", "content": output})
                            messages.append(
                                {
                                    "role": "user",
                                    "content": "SYSTEM ERROR: You responded with text but DID NOT execute any tools. The script has NOT changed. You must CALL the functions (e.g., create_surface, etc) to perform the task.",
                                }
                            )
                            continue  # Try next attempt immediately
                        else:
                            # Final failure
                            console.print(
                                Panel(
                                    Markdown(output),
                                    title="Agent (Failed)",
                                    border_style="red",
                                )
                            )
                            console.print(
                                "[error]Agent failed to execute tools after multiple attempts.[/error]"
                            )
                            task_complete = True
                            break

                    except Exception as e:
                        console.print(f"[error]Error: {e}[/error]")
                        task_complete = True
                        break

        # End of outer while loop (user pressed Enter on prompt)
        self._print_script()
        return self.builder.get_script()

    def run_onboarding(self):
        console.print("\n")
        console.rule("[bold cyan]MCDC Onboarding[/bold cyan]")
        console.print("Welcome! Select a category to edit or add components.")
        console.print(
            "MCDC scripts are built incrementally. Start with materials and surfaces, then move to cells, hierarchy, sources, tallies, and settings.\n"
        )
        console.print("You can also view/edit the full script or save your progress.\n")
        console.print(
            "[dim]Note: You can type 'undo' at any prompt to undo the last action.[/dim]"
        )
        console.print(
            "[dim]Tip: Use 'view' to see the current script and make edits.[/dim]"
        )
        console.print(
            "[dim]Tip: Use 'viz' to visualize your geometry after defining surfaces and cells.[/dim]"
        )

        # Map Menu Options to (Internal Step Name, Display Label)
        menu_options = {
            "1": ("material", "Materials"),
            "2": ("surface", "Surfaces"),
            "3": ("cell", "Cells"),
            "4": ("hierarchy", "Universes & Lattices"),
            "5": ("source", "Sources"),
            "6": ("tally", "Tallies"),
            "7": ("settings", "Settings"),
        }

        while True:
            console.print("\n[bold]Main Menu:[/bold]")
            try:
                # Print menu with counts
                for key, (step_id, label) in menu_options.items():
                    # Check how many items exist for this step
                    if step_id == "hierarchy":
                        count = len(self.builder.defined.get("universe", [])) + len(
                            self.builder.defined.get("lattice", [])
                        )
                    else:
                        count = len(self.builder.defined.get(step_id, set()))

                    status = (
                        f"[green]({count} defined)[/green]"
                        if count > 0
                        else "[dim](empty)[/dim]"
                    )
                    console.print(f"  [{key}] {label} {status}")

                console.print("  \[v] View/Edit/Add Full Script")
                console.print("  \[l] Load Script from File")
                console.print("  \[d] Debug Existing Script")
                console.print("  \[s] Save & Exit")
                console.print("  \[q] Quit (No Save)")

                choice = self.get_input("Select option:")

                if choice in menu_options:
                    step_id, label = menu_options[choice]

                    # First time visiting this step? Teach the concept.
                    is_empty = False
                    if step_id == "hierarchy":
                        is_empty = (
                            len(self.builder.defined.get("universe", []))
                            + len(self.builder.defined.get("lattice", []))
                        ) == 0
                    else:
                        is_empty = len(self.builder.defined.get(step_id, set())) == 0

                    if is_empty:
                        # If user says "No" to "Ready to create?", we just go back to menu
                        if self.teach_concept(step_id):
                            self.create_step(step_id)
                    else:
                        # Already knows it, go straight to builder
                        self.create_step(step_id)

                elif choice.lower() == "v":
                    self._handle_view_mode()

                elif choice.lower() == "l":
                    filepath = self.get_input(
                        "Enter path to python script (or drag into terminal):"
                    )
                    filepath = filepath.strip('"').strip("'")

                    if filepath:
                        with console.status(
                            f"[bold yellow]Parsing {filepath}...[/bold yellow]"
                        ):
                            result = self.builder.parse_and_load(filepath)

                        if "Error" in result:
                            console.print(f"[bold red]{result}[/bold red]")
                        else:
                            console.print(f"[bold green]{result}[/bold green]")
                            # Show the user what we loaded
                            self._print_script()

                elif choice.lower() == "d":
                    debugger = DebugHandler(
                        agent=self.agent,
                        builder=self.builder,
                        session=self.session,
                        console=console,
                        retriever=self.retriever,
                    )
                    debugger.run()

                elif choice.lower() == "s":
                    # Save Input Script
                    final_script = self.builder.get_script()
                    console.print("\n")
                    console.print(Syntax(final_script, "python", theme="monokai"))

                    input_filename = self.get_input(
                        "Filename for Input Deck (e.g., input.py):"
                    )
                    if not input_filename:
                        input_filename = "mcdc_input.py"
                    if not input_filename.endswith(".py"):
                        input_filename += ".py"

                    try:
                        Path(input_filename).write_text(final_script)
                        console.print(
                            f"[success]Saved input to {input_filename}[/success]"
                        )
                    except Exception as e:
                        console.print(f"[error]Error saving input: {e}[/error]")
                        continue

                    # Trigger Output Processing Workflow
                    if Confirm.ask(
                        "Would you like to generate an output-processing script?",
                        default=True,
                    ):

                        console.print("\n[bold]Output Processing Configuration[/bold]")
                        console.print(
                            "Press [Enter] to generate default plots (Heatmaps for meshes, Spectra for energy)."
                        )
                        console.print(
                            "Or describe what you want (e.g., 'Plot fission rate vs time for the fuel cell')."
                        )

                        user_request = self.get_input("Preferences:")
                        plot_gen = PlotGenerator(self.builder, self.llm)

                        with console.status(
                            "[bold blue]Generating processing script...[/bold blue]"
                        ):
                            try:
                                proc_code = plot_gen.generate_script(user_request)
                            except Exception as e:
                                console.print(
                                    f"[error]Failed to generate script: {e}[/error]"
                                )
                                proc_code = None

                        if proc_code:
                            # Display the generated script
                            console.print("\n")
                            console.print(
                                Panel(
                                    Syntax(proc_code, "python", theme="monokai"),
                                    title="Generated Processing Script",
                                )
                            )

                            proc_filename = self.get_input(
                                f"Filename for Processing Script (default: process_{input_filename}):"
                            )
                            if not proc_filename:
                                proc_filename = f"process_{input_filename}"
                            if not proc_filename.endswith(".py"):
                                proc_filename += ".py"

                            try:
                                Path(proc_filename).write_text(proc_code)
                                console.print(
                                    f"[success]Saved processing script to {proc_filename}[/success]"
                                )
                                console.print(
                                    f"\n[info]To run: python {input_filename} && python {proc_filename}[/info]"
                                )
                            except Exception as e:
                                console.print(
                                    f"[error]Error saving processing script: {e}[/error]"
                                )
                    break

                elif choice.lower() == "q":
                    if Confirm.ask("Quit without saving?"):
                        console.print("[dim]Exiting...[/dim]")
                        break
                else:
                    console.print("[error]Invalid option[/error]")
            except BackToMenu:
                console.print("\n[bold yellow]Returning to Main Menu...[/bold yellow]")
                continue


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MCDC-Agent: Interactive Monte Carlo simulation builder"
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Enable batch mode for automatic script generation",
    )
    parser.add_argument(
        "--input_file",
        type=str,
        help="Path to text file containing simulation prompt (for batch mode)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="mcdc_input.py",
        help="Output filename for generated script (default: mcdc_input.py)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview decomposed plan without execution",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Enable verbose logging (DEBUG level)"
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        default=True,
        help="Enable dry-run validation after script generation (default: True)",
    )
    parser.add_argument(
        "--no-validate",
        action="store_false",
        dest="validate",
        help="Disable dry-run validation",
    )

    args = parser.parse_args()

    # Configure logging - silent by default, verbose on request
    log_level = logging.DEBUG if args.verbose else logging.WARNING
    logging.basicConfig(
        level=log_level, format="%(asctime)s [%(levelname)s] %(message)s"
    )
    try:
        llm = load_llm(temperature=0.1)
        agent = MCDCAgent(llm)

        if args.batch:
            # Batch mode
            if not args.input_file:
                console.print("[error]Batch mode requires --input_file[/error]")
                exit(1)

            input_path = Path(args.input_file)
            if not input_path.exists():
                console.print(f"[error]Input file not found: {args.input_file}[/error]")
                exit(1)

            prompt = input_path.read_text().strip()
            console.print(f"[info]Loaded prompt from: {args.input_file}[/info]")
            console.print(
                f"[dim]{prompt[:200]}{'...' if len(prompt) > 200 else ''}[/dim]"
            )

            # Decompose the prompt
            decomposer = Decomposer(llm)
            tasks = decomposer.decompose(prompt)

            console.print(f"\n[info]Decomposed into {len(tasks)} steps:[/info]")
            for i, task in enumerate(tasks, 1):
                console.print(f"  {i}. [{task.step.upper()}] {task.instruction}")

            if args.dry_run:
                console.print("\n[warning]Dry run - no execution[/warning]")
                exit(0)

            # Execute batch
            result = agent.execute_batch(
                [t.to_dict() for t in tasks], enable_validation=args.validate
            )

            # Save result
            output_path = Path(args.output)
            output_path.write_text(result)
            console.print(
                f"\n[success]Generated script saved to: {args.output}[/success]"
            )

        else:
            # Interactive mode
            agent.run_onboarding()

    except KeyboardInterrupt:
        console.print("\n\n[warning]Exiting. Your progress was not saved.[/warning]")
    except Exception as e:
        console.print(f"\n[error]Fatal error: {e}[/error]")
        import traceback

        traceback.print_exc()
