"""
Create a CPU-safe requirements file for GitHub Actions.

ResearchEase AI Version 14
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


EXCLUDED_PACKAGES = {
    "torch",
    "torchvision",
    "torchaudio",
    "triton",
}


def normalize_package_name(requirement: str) -> str:
    """
    Extract and normalize a package name from a requirement line.

    Examples:
        torch==2.7.1 -> torch
        sentence-transformers>=3 -> sentence-transformers
        package[extra]>=1.0 -> package
    """

    value = requirement.strip()

    if value.startswith("-e "):
        value = value[3:].strip()

    if " @ " in value:
        value = value.split(" @ ", maxsplit=1)[0]

    package_name = re.split(
        r"[<>=!~;\[\s]",
        value,
        maxsplit=1,
    )[0]

    return package_name.strip().lower().replace("_", "-")


def should_exclude(line: str) -> bool:
    """
    Return True when a requirement should not be copied to CI.
    """

    stripped = line.strip()

    if not stripped:
        return False

    if stripped.startswith("#"):
        return False

    if stripped.startswith(("-", "--")):
        return False

    package_name = normalize_package_name(stripped)

    if package_name in EXCLUDED_PACKAGES:
        return True

    if package_name.startswith("nvidia-"):
        return True

    return False


def create_ci_requirements(
    source_path: Path,
    output_path: Path,
) -> None:
    """
    Create a requirements file without GPU-specific packages.
    """

    if not source_path.exists():
        raise FileNotFoundError(
            f"Requirements file not found: {source_path}"
        )

    source_lines = source_path.read_text(
        encoding="utf-8",
    ).splitlines()

    output_lines: list[str] = []
    removed_lines: list[str] = []

    for line in source_lines:
        if should_exclude(line):
            removed_lines.append(line)
            continue

        output_lines.append(line)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        "\n".join(output_lines).rstrip() + "\n",
        encoding="utf-8",
    )

    print(f"Source requirements: {source_path}")
    print(f"CI requirements: {output_path}")
    print(f"Total source lines: {len(source_lines)}")
    print(f"Removed GPU package lines: {len(removed_lines)}")

    for removed_line in removed_lines:
        print(f"  Removed: {removed_line}")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a CPU-safe requirements file "
            "for ResearchEase CI."
        )
    )

    parser.add_argument(
        "--source",
        type=Path,
        default=Path("requirements.txt"),
        help="Source requirements file.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("requirements.ci.generated.txt"),
        help="Generated CI requirements file.",
    )

    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()

    create_ci_requirements(
        source_path=arguments.source,
        output_path=arguments.output,
    )


if __name__ == "__main__":
    main()