"""Run the project's complete quality check."""

from __future__ import annotations

import subprocess
from collections.abc import Sequence

CHECKS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("ruff lint and auto-fix", ("ruff", "check", "--fix", ".")),
    ("ruff format", ("ruff", "format", ".")),
    ("mypy", ("mypy",)),
    ("pytest", ("pytest",)),
)


def run(command: Sequence[str]) -> int:
    """Run one check and return its exit code."""
    completed = subprocess.run(command, check=False)
    return completed.returncode


def main() -> int:
    """Run all checks in order, stopping at the first failure."""
    total = len(CHECKS)
    for index, (description, command) in enumerate(CHECKS, start=1):
        print(f"[{index}/{total}] {description}...", flush=True)
        if exit_code := run(command):
            return exit_code

    print("\nAll checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
