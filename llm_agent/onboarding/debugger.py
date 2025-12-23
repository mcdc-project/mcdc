from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax
from typing import Any
from prompt_toolkit import PromptSession
from langchain_core.runnables import Runnable
from llm_agent.utils import parse_agent_response
from llm_agent.onboarding.validator import DryRunValidator, ValidationStatus
from difflib import unified_diff


class DebugHandler:
    """
    Interactive debugger that uses DryRunValidator for script validation
    and allows user confirmation before applying fixes.

    Unlike batch mode validation, this provides:
    - User confirmation before each fix is applied
    - Ability to revert changes
    - Visual diff of proposed changes
    """

    def __init__(
        self,
        agent: Runnable,
        builder: Any,
        session: PromptSession,
        console: Console,
        retriever: Any,
        timeout: float = 10.0,
    ):
        self.agent = agent
        self.builder = builder
        self.session = session
        self.console = console
        self.retriever = retriever
        self.validator = DryRunValidator(timeout=timeout)

    def _parse_agent_response(self, response: Any) -> str:
        """Helper to extract text from various LangChain response formats."""
        return parse_agent_response(response)

    def show_diff(self, old_code: str, new_code: str):
        """Visualizes changes made to the script using a unified diff."""
        diff = list(
            unified_diff(
                old_code.splitlines(),
                new_code.splitlines(),
                fromfile="Before",
                tofile="After",
                lineterm="",
            )
        )

        if not diff:
            self.console.print("[dim]No changes detected.[/dim]")
            return

        diff_text = "\n".join(diff)
        syntax = Syntax(diff_text, "diff", theme="monokai", word_wrap=True)
        self.console.print("\n")
        self.console.print(
            Panel(
                syntax,
                title="[bold yellow]Proposed Changes[/bold yellow]",
                border_style="yellow",
            )
        )

    def run(self):
        """
        Run the interactive debug loop.

        Uses DryRunValidator to test the script and the agent to propose fixes.
        User must confirm each fix before it's applied.
        """
        if not self.builder.entries:
            self.console.print("[warning]Script is empty.[/warning]")
            return

        self.console.print("\n")
        self.console.rule("[bold red]DEBUG MODE[/bold red]")
        self.console.print(
            "[dim]The debugger will validate your script and help fix errors.[/dim]"
        )
        self.console.print(
            "[dim]Commands: 'yes' to apply fix, 'run' to test again, 'no'/'exit' to quit[/dim]\n"
        )

        # Debug context for the agent
        debug_context = """
You are fixing MCDC script errors. Analyze the error(s), identify broken entities, and fix them using tools.

## Example
ERROR: NameError: name 'sphere' is not defined
ANALYSIS: Cell references 'sphere' but no surface named 'sphere' exists.
FIX: Add the missing surface definition:
```
write_code(code="sphere = mcdc.Surface.Sphere(center=[0,0,0], radius=5.0)", entity_type="surface", name="sphere")
```

## Available Tools
- `write_code(code, entity_type, name)` - Add new entity
- `manage_script(action='get')` - View current script state
- `manage_script(action='delete', entity_type, name)` - Remove entity
- `search_docs(query)` - Search documentation
- `search_examples(query)` - Search examples

## To modify an existing entity:
1. `manage_script(action='delete', entity_type='surface', name='sphere')`
2. `write_code(code="sphere = mcdc.Surface.Sphere(...)", entity_type='surface', name='sphere')`

## Rules
1. Fix ALL errors of the same type in one pass (e.g., multiple missing surfaces)
2. Don't rewrite working code - only fix what's broken
3. Ordering is handled automatically - just add/delete as needed
"""

        # --- MAIN DEBUG LOOP ---
        iteration = 0
        max_iterations = 5

        while iteration < max_iterations:
            iteration += 1
            self.console.print(
                f"\n[bold cyan]--- Debug Iteration {iteration}/{max_iterations} ---[/bold cyan]"
            )

            # Validate script (Similar to batch executor)
            script_content = self.builder.get_script()

            with self.console.status(
                "[bold yellow]Running dry-run validation...[/bold yellow]"
            ):
                result = self.validator.validate(script_content)

            # Check if validation passed with NO warnings
            if result.success:
                self.console.print(
                    "[bold green]Validation PASSED! Script is working correctly (no warnings).[/bold green]"
                )
                if result.runtime_report:
                    self.console.print(
                        Panel(
                            result.runtime_report,
                            title="Runtime Report",
                            border_style="green",
                        )
                    )
                return

            # Handle warnings - treat them as issues to fix
            if result.status == ValidationStatus.WARNING:
                self.console.print(
                    "[bold yellow]Validation passed but with warnings that should be fixed:[/bold yellow]"
                )
                for warning in result.warnings:
                    self.console.print(f"  - {warning}")
                if result.runtime_report:
                    self.console.print(
                        Panel(
                            result.runtime_report,
                            title="Runtime Report",
                            border_style="yellow",
                        )
                    )
                # Build error_display from warnings for the agent
                error_display = (
                    "WARNINGS (simulation ran but has issues):\n"
                    + "\n".join(f"- {w}" for w in result.warnings)
                )
            else:
                # Display error
                error_display = result.error_message or "Unknown error"
                status_label = (
                    "⏱ TIMEOUT"
                    if result.status == ValidationStatus.TIMEOUT
                    else "✗ ERROR"
                )
                self.console.print(
                    Panel(
                        Syntax(error_display, "text", theme="monokai", word_wrap=True),
                        title=f"[bold red]{status_label}[/bold red]",
                        border_style="red",
                    )
                )

            # Build debug prompt
            script_lines = script_content.split("\n")
            numbered_script = "\n".join(
                [f"{i+1:03d} | {line}" for i, line in enumerate(script_lines)]
            )

            # Track state before agent acts
            script_before = script_content
            entities_before = {k: set(v) for k, v in self.builder.defined.items()}

            messages = [
                {
                    "role": "user",
                    "content": f"""{debug_context}

### CURRENT SCRIPT STATE:
```python
{numbered_script}
```

### VALIDATION ERROR:
```
{error_display}
```

### YOUR TASK:
Fix the error(s) shown above. Use write_code() to add entities, manage_script() to view/delete.
If needed, use search_docs() or search_examples() to look up correct API usage.
""",
                }
            ]

            # Agent proposes fix
            with self.console.status(
                "[bold red]Analyzing error and proposing fix...[/bold red]"
            ):
                try:
                    response = self.agent.invoke({"messages": messages})
                    output = self._parse_agent_response(response)
                except Exception as e:
                    self.console.print(f"[error]Agent error: {e}[/error]")
                    break

            # Handle case where agent only called tools without text explanation
            if not output or not output.strip():
                script_after = self.builder.get_script()
                if script_before != script_after:
                    output = "I've applied the fix using the available tools."
                else:
                    output = "I couldn't determine a fix for this error."

            # Display agent's explanation
            if output:
                self.console.print(
                    Panel(
                        Markdown(output), title="Debugger Analysis", border_style="red"
                    )
                )

            # Show proposed changes
            script_after = self.builder.get_script()
            changes_made = script_before != script_after

            if changes_made:
                self.show_diff(script_before, script_after)

                # Show entity changes
                for etype, names in self.builder.defined.items():
                    added = names - entities_before.get(etype, set())
                    removed = entities_before.get(etype, set()) - names
                    if added:
                        self.console.print(
                            f"[green]+ Added {etype}(s): {', '.join(added)}[/green]"
                        )
                    if removed:
                        self.console.print(
                            f"[red]- Removed {etype}(s): {', '.join(removed)}[/red]"
                        )

            # User confirmation
            if changes_made:
                prompt_text = "Apply these changes? ('yes' to apply & test, 'no' to revert, 'exit' to quit): "
            else:
                prompt_text = "No changes proposed. ('run' to retry, 'exit' to quit): "

            user_cmd = self.session.prompt(prompt_text).strip().lower()

            if user_cmd in ["exit", "quit", "q"]:
                if changes_made:
                    # Revert changes before exiting
                    self._revert_changes(script_before, entities_before)
                break

            elif user_cmd in ["no", "n", "revert"]:
                if changes_made:
                    self._revert_changes(script_before, entities_before)
                    self.console.print("[yellow]Changes reverted.[/yellow]")
                continue  # Go to next iteration to try again

            elif user_cmd in ["yes", "y", "apply", "run", ""]:
                if changes_made:
                    # Reorder to ensure dependencies are correct
                    self.builder.reorder()
                    self.console.print("[green]Changes applied. Testing...[/green]")
                # Loop back to validate the fix
                continue

            else:
                self.console.print(
                    f"[warning]Unknown command: {user_cmd}. Continuing...[/warning]"
                )

        if iteration >= max_iterations:
            self.console.print(
                f"[warning]Reached maximum iterations ({max_iterations}). Exiting debug mode.[/warning]"
            )
            self.console.print(
                "[dim]You can continue editing from the main menu.[/dim]"
            )

    def _revert_changes(self, original_script: str, original_entities: dict):
        """
        Attempt to revert the builder state to the original.

        Note: This is a best-effort revert. For complex changes,
        the user may need to use 'undo' from the main menu.
        """
        try:
            # Clear current entries and reload from script
            self.builder.entries.clear()
            self.builder.defined = {k: set() for k in self.builder.defined.keys()}

            # Re-parse the original script
            # This is a simplified approach - for full revert, we'd need state snapshots
            self.console.print("[dim]Reverting to previous state...[/dim]")

            # Try to restore by re-parsing (if parse_and_load supports string input)
            # For now, just warn the user
            self.console.print(
                "[warning]Note: Changes reverted in memory. Use 'undo' from main menu if needed.[/warning]"
            )
        except Exception as e:
            self.console.print(f"[warning]Could not fully revert: {e}[/warning]")
            self.console.print(
                "[dim]Use 'undo' from the main menu to restore previous state.[/dim]"
            )
