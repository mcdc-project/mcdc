import subprocess
import tempfile
import re
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import List, Optional, Any


class ValidationStatus(Enum):
    """Validation result status."""

    SUCCESS = "success"
    ERROR = "error"
    WARNING = "warning"
    TIMEOUT = "timeout"


@dataclass
class ValidationResult:
    """Result of a dry-run validation."""

    status: ValidationStatus
    runtime_report: Optional[str] = None
    error_message: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    raw_output: str = ""  # Full output for debugging

    @property
    def success(self) -> bool:
        return self.status == ValidationStatus.SUCCESS

    def __str__(self) -> str:
        if self.success:
            return f"PASS\n{self.runtime_report or ''}"
        elif self.status == ValidationStatus.WARNING:
            return f"WARNING\n" + "\n".join(self.warnings)
        elif self.status == ValidationStatus.TIMEOUT:
            return f"TIMEOUT - Script stalled (no runtime report produced)"
        else:
            return f"FAILED\n{self.error_message or 'Unknown error'}"


class DryRunValidator:
    """
    Run generated MCDC scripts with minimal particles to detect errors.

    Injects dry-run settings (N_particle=100, N_batch=1) before mcdc.run()
    and parses output to detect success, errors, or warnings. Could cause issues with
    variance due to small number of particles (large source or mesh).

    Key behavior:
    - A successful MCDC run ALWAYS prints "Runtime report:" at the end
    - If we don't see this within timeout, something is wrong (stall, infinite loop, etc.)
    - Warnings like "particles escaping" indicate geometry issues even if run completes

    Usage:
        validator = DryRunValidator()
        result = validator.validate(script_content)
        if not result.success:
            print(f"Error: {result.error_message}")
    """

    # Settings injected for dry-run validation
    DRY_RUN_SETTINGS = """
# --- DRY-RUN VALIDATION SETTINGS (auto-injected) ---
mcdc.settings.N_particle = 100
mcdc.settings.N_batch = 1
# --- END DRY-RUN SETTINGS ---
"""

    # Pattern to match the runtime report (required for success)
    RUNTIME_REPORT_PATTERN = re.compile(
        r"Runtime report:\s*\n.*Total\s+\|\s+[\d.]+\s+seconds",
        re.DOTALL | re.IGNORECASE,
    )

    # Patterns that indicate warnings (geometry issues, etc.)
    # TODO: Add more patterns
    WARNING_PATTERNS = [
        (
            re.compile(r"particle[s]?\s+(escap|lost|leak)", re.IGNORECASE),
            "Particles escaping geometry bounds - check cell regions for gaps",
        ),
        (
            re.compile(r"A particle is lost at", re.IGNORECASE),
            "Particle lost at specific location - geometry has gaps",
        ),
        (
            re.compile(r"(active|census|source|future)_bank_buffer", re.IGNORECASE),
            "Particle buffer issue - may need to increase buffer size",
        ),
        (
            re.compile(r"bank\s+is\s+full", re.IGNORECASE),
            "Particle bank overflow - too many particles being tracked",
        ),
        (
            re.compile(r"buffer\s+(filling|overflow|full)", re.IGNORECASE),
            "Particle buffer overflow",
        ),
        (
            re.compile(r"^ERROR:", re.IGNORECASE | re.MULTILINE),
            "Error detected in simulation",
        ),
        (re.compile(r"warning:", re.IGNORECASE), "Warning detected in simulation"),
        (
            re.compile(r"lost\s+\d+\s+particle", re.IGNORECASE),
            "Particles lost during simulation",
        ),
        (
            re.compile(r"blankout fn for report_lost", re.IGNORECASE),
            "Particle lost - geometry has gaps (Numba decorator binding issue)",
        ),
    ]

    def __init__(self, timeout: float = 10.0, cleanup: bool = True):
        """
        Initialize validator.

        Args:
            timeout: Maximum seconds to wait for simulation (default 10)
                     If script doesn't produce runtime report by then, it's a stall
            cleanup: Whether to remove temp files after validation (default True)
        """
        self.timeout = timeout
        self.cleanup = cleanup

    def validate(self, script: str) -> ValidationResult:
        """
        Execute script with dry-run settings and analyze output.

        Args:
            script: Complete MCDC Python script as string

        Returns:
            ValidationResult with status, runtime report or error message
        """
        # Inject dry-run settings before mcdc.run()
        modified_script = self._inject_dry_run_settings(script)

        # Write to temp file
        temp_dir = tempfile.mkdtemp(prefix="mcdc_validate_")
        script_path = Path(temp_dir) / "dry_run_script.py"
        script_path.write_text(modified_script)

        try:
            # Execute script with timeout
            result = subprocess.run(
                [sys.executable, str(script_path)],
                capture_output=True,
                text=True,
                timeout=self.timeout,
                cwd=temp_dir,
            )

            # Combine stdout and stderr
            output = result.stdout + "\n" + result.stderr

            return self._analyze_output(output, result.returncode)

        except subprocess.TimeoutExpired as e:
            # Script stalled - get whatever output we have
            partial_output = ""
            if e.stdout:
                partial_output = (
                    e.stdout.decode() if isinstance(e.stdout, bytes) else e.stdout
                )
            if e.stderr:
                err = e.stderr.decode() if isinstance(e.stderr, bytes) else e.stderr
                partial_output += "\n" + err

            # Check if there's an error in the partial output
            error_msg = self._extract_error(partial_output)
            warnings = self._extract_warnings(partial_output)

            # Check if simulation is running (either mode):
            # - Fixed source: progress bar [====] XX%
            # - Eigenmode: k-effective table with numbered rows
            progress_bar_pattern = re.compile(r"\[\=*\s*\]\s*\d+%")
            eigenmode_pattern = re.compile(
                r"^\s*\d+\s+[\d.]+\s+", re.MULTILINE
            )  # "1  0.96330  ..."

            has_progress_bar = progress_bar_pattern.search(partial_output)
            has_eigenmode_output = eigenmode_pattern.search(partial_output)
            simulation_running = has_progress_bar or has_eigenmode_output

            if simulation_running and not error_msg and not warnings:
                # Simulation is running fine, geometry is valid
                mode = "eigenmode" if has_eigenmode_output else "fixed source"
                return ValidationResult(
                    status=ValidationStatus.SUCCESS,
                    runtime_report=f"Simulation running ({mode}, timed out at {self.timeout}s but no errors detected)",
                    raw_output=partial_output,
                )

            # Build diagnostic message for actual failures
            diag_msg = f"Script timed out after {self.timeout}s without producing runtime report.\n"
            if error_msg:
                diag_msg += f"Last error: {error_msg}\n"
            if warnings:
                diag_msg += f"Warnings: {', '.join(warnings)}\n"
            if partial_output:
                # Get last few lines for context
                last_lines = partial_output.strip().split("\n")[-5:]
                diag_msg += f"Last output:\n" + "\n".join(last_lines)

            return ValidationResult(
                status=ValidationStatus.TIMEOUT,
                error_message=diag_msg,
                warnings=warnings,
                raw_output=partial_output,
            )

        except Exception as e:
            return ValidationResult(
                status=ValidationStatus.ERROR,
                error_message=f"Execution failed: {str(e)}",
            )
        finally:
            # Cleanup temp files
            if self.cleanup:
                self._cleanup_temp(temp_dir)

    def _inject_dry_run_settings(self, script: str) -> str:
        """Insert dry-run settings before mcdc.run() call."""
        # Find mcdc.run() call
        run_pattern = re.compile(r"^(\s*mcdc\.run\s*\()", re.MULTILINE)

        match = run_pattern.search(script)
        if match:
            # Insert settings before run()
            insert_pos = match.start()
            return (
                script[:insert_pos] + self.DRY_RUN_SETTINGS + "\n" + script[insert_pos:]
            )
        else:
            # No mcdc.run() found - append settings and run at end
            return script + "\n" + self.DRY_RUN_SETTINGS + "\nmcdc.run()\n"

    def _analyze_output(self, output: str, returncode: int) -> ValidationResult:
        """
        Analyze subprocess output to determine validation result.

        A successful MCDC run ALWAYS produces "Runtime report:" at the end.
        """
        # Extract warnings first
        warnings = self._extract_warnings(output)

        # Check for runtime report (required for success)
        runtime_match = self.RUNTIME_REPORT_PATTERN.search(output)

        if runtime_match:
            # Found runtime report - simulation completed
            # Extract the full runtime report section
            report_start = output.find("Runtime report:")
            if report_start >= 0:
                # Find end of report (next blank line or end)
                report_section = output[report_start:]
                lines = report_section.split("\n")
                report_lines = []
                for line in lines:
                    if line.strip() == "" and report_lines:
                        break
                    report_lines.append(line)
                runtime_report = "\n".join(report_lines)
            else:
                runtime_report = runtime_match.group()

            if warnings:
                # Success but with warnings
                return ValidationResult(
                    status=ValidationStatus.WARNING,
                    runtime_report=runtime_report,
                    warnings=warnings,
                    raw_output=output,
                )

            return ValidationResult(
                status=ValidationStatus.SUCCESS,
                runtime_report=runtime_report,
                raw_output=output,
            )

        # No runtime report found - this is a failure
        # Even if returncode is 0, no runtime report means something is wrong
        error_msg = self._extract_error(output)

        if not error_msg and returncode != 0:
            error_msg = f"Script failed with exit code {returncode}"
        elif not error_msg:
            # Script exited but no runtime report - unusual
            error_msg = "Script completed but no runtime report was produced"

        return ValidationResult(
            status=ValidationStatus.ERROR,
            error_message=error_msg,
            warnings=warnings,
            raw_output=output,
        )

    def _extract_warnings(self, output: str) -> List[str]:
        """
        Extract warning messages from output.

        Checks for explicit warning patterns that indicate problems
        (particles lost, buffer issues, etc.)
        """
        warnings = []
        for pattern, message in self.WARNING_PATTERNS:
            if pattern.search(output):
                warnings.append(message)
        return warnings

    def _extract_error(self, output: str) -> Optional[str]:
        """Extract the most relevant error message from output."""
        # Look for Python exceptions
        error_patterns = [
            # Traceback with specific error
            re.compile(r"(\w+Error: .+?)(?:\n|$)", re.MULTILINE),
            # MCDC-specific errors
            re.compile(r"(MCDC Error: .+?)(?:\n|$)", re.IGNORECASE | re.MULTILINE),
            # Generic error lines
            re.compile(r"(Error: .+?)(?:\n|$)", re.IGNORECASE | re.MULTILINE),
            # Exception lines
            re.compile(r"(Exception: .+?)(?:\n|$)", re.MULTILINE),
        ]

        # Return first error message found
        for pattern in error_patterns:
            match = pattern.search(output)
            if match:
                return match.group(1).strip()

        # Return last non-empty line if no specific error found
        lines = [l.strip() for l in output.strip().split("\n") if l.strip()]
        if lines:
            # Return last 3 lines for context
            return "\n".join(lines[-3:])

        return None

    def _cleanup_temp(self, temp_dir: str) -> None:
        """Remove temporary directory and all contents."""
        import shutil

        try:
            shutil.rmtree(temp_dir)
        except Exception:
            pass  # Best effort cleanup


def validate_script(script: str, timeout: float = 10.0) -> ValidationResult:
    """
    Convenience function for quick validation.

    Args:
        script: MCDC script content
        timeout: Max execution time in seconds

    Returns:
        ValidationResult
    """
    validator = DryRunValidator(timeout=timeout)
    return validator.validate(script)
