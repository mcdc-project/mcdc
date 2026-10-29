"""Share command-line conversion failure handling across projectile generators."""

# ======================================================================================
# Command-line option
# ======================================================================================


def add_keep_going_option(parser):
    """Add the common opt-in continuation flag to an argument parser."""
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="Continue after conversion errors and summarize failures at the end",
    )


# ======================================================================================
# Conversion failure reporting
# ======================================================================================


class ConversionErrors:
    """Report the first failed input or collect failures for a final summary.

    Each generator catches its expected source-format and I/O exceptions and
    passes them to handle. Successful conversions remain the generator's
    responsibility. Both reporting modes terminate with exit status 1 when
    failures occur; finish returns normally when no failures were collected.
    """

    def __init__(self, parser, *, keep_going=False):
        self.parser = parser
        self.keep_going = keep_going
        self.failures = []

    def handle(self, name, error, *, progress=None):
        """Attach an input identifier and either stop or retain the failure.

        Close an active progress bar before reporting a fail-fast error.
        Callers invoke finish after all conversion phases so an optional
        secondary source, such as PSTAR, participates in the same summary.
        """
        message = f"{name}: {error}"
        if not self.keep_going:
            if progress is not None:
                progress.close()
            self.parser.exit(1, f"{message}\n")
        self.failures.append(message)

    def finish(self):
        """Report every collected failure and exit unsuccessfully if any exist."""
        if self.failures:
            self.parser.exit(
                1,
                f"Conversion failed for {len(self.failures)} file(s):\n"
                + "\n".join(self.failures)
                + "\n",
            )
