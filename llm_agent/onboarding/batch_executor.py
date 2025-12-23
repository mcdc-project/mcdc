from typing import List, Dict, Any, Optional
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from llm_agent.utils import parse_agent_response
from llm_agent.onboarding.validator import (
    DryRunValidator,
    ValidationResult,
    ValidationStatus,
)


class BatchExecutor:
    """
    Executes a decomposed plan in batch mode without user interaction.

    For each step:
    1. Save checkpoint
    2. Inject instruction into agent
    3. Execute with retry logic
    4. If step fails, restore checkpoint and continue/abort
    """

    def __init__(
        self,
        agent: Any,
        builder: Any,
        console: Console,
        response_parser: callable = None,
        enable_validation: bool = True,
        validation_timeout: float = 10.0,
        max_corrections: int = 5,
    ):
        """
        Initialize batch executor.

        Args:
            agent: LangChain agent for executing instructions
            builder: ScriptBuilder instance for script management
            console: Rich Console for output
            response_parser: Optional callable to parse agent responses
            enable_validation: Run dry-run validation after generation (default True)
            validation_timeout: Max seconds for validation run (default 10)
            max_corrections: Max self-correction attempts (default 3)
        """
        self.agent = agent
        self.builder = builder
        self.console = console
        self._parse_response = response_parser or parse_agent_response
        self.enable_validation = enable_validation
        self.max_corrections = max_corrections
        self.validator = (
            DryRunValidator(timeout=validation_timeout) if enable_validation else None
        )

    def execute(self, plan: List[Dict[str, str]], max_iterations: int = 5) -> str:
        """
        Execute a decomposed plan in batch mode.

        Args:
            plan: List of dicts with 'step' and 'instruction' keys
            max_iterations: Max tool call attempts per step (default 5)

        Returns:
            Generated script as string
        """
        self.console.print("\n")
        self.console.rule("[bold cyan]BATCH EXECUTION MODE[/bold cyan]")

        total_steps = len(plan)
        successful_steps = 0
        failed_steps = []

        for step_num, task in enumerate(plan, 1):
            step_type = task.get("step", "unknown")
            instruction = task.get("instruction", "")

            # print step info
            self.console.print(
                f"\n[step] Step {step_num}/{total_steps}: {step_type.upper()} [/step]"
            )
            self.console.print(
                f"[dim]{instruction[:100]}{'...' if len(instruction) > 100 else ''}[/dim]"
            )

            # Save checkpoint before this step
            checkpoint_name = f"step_{step_num}_{step_type}"
            self.builder.save_checkpoint(checkpoint_name)

            # For error-prone entity types, instruct agent to look up API first
            api_lookup_types = {"source", "tally", "cell", "surface"}
            api_lookup_instruction = ""
            if step_type in api_lookup_types:
                api_lookup_instruction = (
                    f"FIRST: Call search_api('{step_type}') to verify correct parameter names. "
                    f"Common mistakes: TallyCell uses cell= not cells=, Source uses position= not pos=, "
                    f"CylinderZ center is 2D [x,y] not 3D.\n\n"
                )

            # Build message for agent
            current_context = self.builder.get_script()
            messages = [
                {
                    "role": "user",
                    "content": (
                        f"Current MCDC script:\n```python\n{current_context}\n```\n\n"
                        f"{api_lookup_instruction}"
                        f"TASK ({step_type.upper()}): {instruction}\n\n"
                        f"STRICT RULES:\n"
                        f"1. Execute this task IMMEDIATELY using the appropriate tools.\n"
                        f"2. Do NOT ask for confirmation - this is batch mode.\n"
                        f"3. Create ONLY what is explicitly requested. Do NOT add:\n"
                        f"   - set_root_universe (unless specifically asked)\n"
                        f"   - settings (unless specifically asked)\n"
                        f"   - sources or tallies (unless specifically asked)\n"
                        f"4. Additional components will be requested in separate steps."
                    ),
                }
            ]

            # Execution loop with retries, similar to agent.py but do not allow questions
            step_success = False

            for attempt in range(max_iterations):
                try:
                    with self.console.status(
                        f"[bold yellow]Executing {step_type} (attempt {attempt+1}/{max_iterations})...",
                        spinner="dots",
                    ):
                        script_before = self.builder.get_script()
                        response = self.agent.invoke({"messages": messages})
                        script_after = self.builder.get_script()

                    output = self._parse_response(response)
                    script_changed = script_before != script_after

                    if script_changed:
                        self.builder.reorder()
                        self.console.print(
                            f"[success]Step completed successfully[/success]"
                        )
                        if output:
                            self.console.print(
                                f"[dim]{output[:150]}{'...' if len(output) > 150 else ''}[/dim]"
                            )
                        step_success = True
                        successful_steps += 1
                        break

                    # Check if agent is asking questions (shouldn't in batch mode)
                    if "?" in output:
                        # Provide more context and retry
                        messages.append({"role": "assistant", "content": output})
                        messages.append(
                            {
                                "role": "user",
                                "content": (
                                    "This is BATCH MODE - do not ask questions. "
                                    "Use your best judgment and execute the task with reasonable defaults. "
                                    "Proceed with tool execution NOW."
                                ),
                            }
                        )
                        continue

                    # Agent didn't change script - force retry
                    if attempt < max_iterations - 1:
                        self.console.print(
                            f"[dim]No changes detected, retrying...[/dim]"
                        )
                        messages.append({"role": "assistant", "content": output})
                        messages.append(
                            {
                                "role": "user",
                                "content": (
                                    "SYSTEM: You responded but did NOT execute any tools. "
                                    "The script has NOT changed. You MUST call tools to complete this task."
                                ),
                            }
                        )

                except Exception as e:
                    self.console.print(
                        f"[error]Error in attempt {attempt+1}: {e}[/error]"
                    )
                    if attempt == max_iterations - 1:
                        break

            if not step_success:
                self.console.print(
                    f"[error]Step failed after {max_iterations} attempts[/error]"
                )
                failed_steps.append(step_type)

                # Restore checkpoint and continue
                self.builder.restore_checkpoint(checkpoint_name)
                self.console.print(
                    f"[warning]Rolled back to checkpoint: {checkpoint_name}[/warning]"
                )

        # Summary
        self.console.print("\n")
        self.console.rule("[bold cyan]BATCH EXECUTION COMPLETE[/bold cyan]")
        self.console.print(f"[info]Successful: {successful_steps}/{total_steps}[/info]")
        if failed_steps:
            self.console.print(
                f"[warning]Failed steps: {', '.join(failed_steps)}[/warning]"
            )

        # Clear checkpoints
        self.builder.clear_checkpoints()

        # Deduplicate settings to remove any repeated entries
        self.builder.deduplicate_settings()

        # Get final script
        final_script = self.builder.get_script()

        # Validate if enabled
        if self.enable_validation and self.validator:
            final_script = self._validate_and_correct(final_script)

        return final_script

    def _validate_and_correct(self, script: str) -> str:
        """
        Run dry-run validation and attempt self-correction if it fails.

        Args:
            script: The generated MCDC script

        Returns:
            Corrected script (or original if validation passed/corrections failed)
        """
        self.console.print("\n")
        self.console.rule("[bold cyan]DRY-RUN VALIDATION[/bold cyan]")

        for attempt in range(self.max_corrections + 1):
            # Run validation
            with self.console.status(
                f"[bold yellow]Validating script (attempt {attempt + 1})...",
                spinner="dots",
            ):
                result = self.validator.validate(script)

            if result.success:
                self.console.print(
                    f"[success]Validation PASSED (no warnings)[/success]"
                )
                if result.runtime_report:
                    self.console.print(f"[dim]{result.runtime_report}[/dim]")
                return script

            if result.status == ValidationStatus.WARNING:
                self.console.print(
                    f"[warning]Validation passed with warnings:[/warning]"
                )
                for warning in result.warnings:
                    self.console.print(f"  - {warning}")
                # Treat warnings as issues to fix - continue to correction logic below

            # Validation failed or has warnings that need fixing
            if result.status != ValidationStatus.WARNING:
                self.console.print(f"[error]Validation FAILED[/error]")
                self.console.print(f"[dim]{result.error_message}[/dim]")

            if attempt >= self.max_corrections:
                self.console.print(
                    f"[error]Max correction attempts ({self.max_corrections}) reached.[/error]"
                )
                break

            # Attempt self-correction
            self.console.print(
                f"[info]Attempting self-correction ({attempt + 1}/{self.max_corrections})...[/info]"
            )
            corrected = self._attempt_correction(script, result)

            if corrected and corrected != script:
                script = corrected
                self.console.print("[info]Script modified, re-validating...[/info]")
            else:
                self.console.print(
                    "[warning]Agent could not make corrections.[/warning]"
                )
                break

        return script

    def _attempt_correction(
        self, script: str, validation_result: ValidationResult
    ) -> Optional[str]:
        """
        Let the agent attempt to fix a validation error or warning.

        Args:
            script: Current script that failed validation or has warnings
            validation_result: The validation failure/warning details

        Returns:
            Corrected script or None if correction failed
        """
        # Build issue description
        if validation_result.status == ValidationStatus.WARNING:
            issue_desc = "Validation PASSED but with WARNINGS that need fixing:\\n\\n"
            issue_desc += "\\n".join(f"- {w}" for w in validation_result.warnings)
        else:
            issue_desc = f"Validation FAILED with this error:\\n\\n```\\n{validation_result.error_message}\\n```"

        # Build correction prompt - encourage batch fixing of similar errors
        messages = [
            {
                "role": "user",
                "content": (
                    f"The generated MCDC script has issues:\n\n"
                    f"{issue_desc}\n\n"
                    f"Current script:\n```python\n{script}\n```\n\n"
                    f"IMPORTANT: Fix ALL similar issues at once. If multiple entities have the same "
                    f"type of error (e.g., all cylinders have wrong syntax), fix them ALL in ONE response.\n\n"
                    f"Use the write_code tool to fix or replace entities. Common fixes:\n"
                    f"- NameError: A variable is used before definition -> define it first\n"
                    f"- TypeError: Wrong parameter type -> check the API\n"
                    f"- Particles escaping/lost: Geometry has gaps -> fix cell regions to cover all space\n"
                    f"  * Ensure outer boundary uses vacuum boundary_condition\n"
                    f"  * Check boolean region logic: use & (and), | (or), ~ (complement)\n"
                    f"  * Ensure cells don't have gaps between them\n\n"
                    f"COMMON MCDC SYNTAX ISSUES:\n"
                    f"- CylinderX: mcdc.Surface.CylinderX(center=[y, z], radius=r) NOT center=[x,y,z]\n"
                    f"- CylinderY: mcdc.Surface.CylinderY(center=[x, z], radius=r)\n"
                    f"- CylinderZ: mcdc.Surface.CylinderZ(center=[x, y], radius=r)\n"
                    f"- Source bounds: Ensure x, y, z ranges are within geometry (not too large)\n"
                    f"- Mesh cells: Keep total cells reasonable (Nx*Ny*Nz < 10000) to avoid variance errors\n"
                ),
            }
        ]

        try:
            script_before = self.builder.get_script()
            response = self.agent.invoke({"messages": messages})
            script_after = self.builder.get_script()

            output = self._parse_response(response)

            # Check if script changed
            if script_before != script_after:
                self.builder.reorder()
                self.console.print(
                    f"[dim]{output[:150] if output else 'Corrections applied'}[/dim]"
                )
                return self.builder.get_script()
            else:
                self.console.print(
                    f"[dim]Agent response: {output[:100] if output else '(no output)'}[/dim]"
                )
                return None

        except Exception as e:
            self.console.print(f"[error]Correction attempt failed: {e}[/error]")
            return None


def execute_batch(
    agent: Any,
    builder: Any,
    console: Console,
    plan: List[Dict[str, str]],
    max_iterations: int = 5,
    response_parser: callable = None,
    enable_validation: bool = True,
    validation_timeout: float = 10.0,
    max_corrections: int = 5,
) -> str:
    """
    Convenience function to execute a batch plan without creating a class instance.

    Args:
        agent: LangChain agent
        builder: ScriptBuilder instance
        console: Rich Console for output
        plan: List of dicts with 'step' and 'instruction' keys
        max_iterations: Max attempts per step
        response_parser: Optional callable to parse agent responses
        enable_validation: Run dry-run validation after generation (default True)
        validation_timeout: Max seconds for validation run (default 10)
        max_corrections: Max self-correction attempts (default 5)

    Returns:
        Generated script as string
    """
    executor = BatchExecutor(
        agent,
        builder,
        console,
        response_parser,
        enable_validation=enable_validation,
        validation_timeout=validation_timeout,
        max_corrections=max_corrections,
    )
    return executor.execute(plan, max_iterations)
