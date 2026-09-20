"""Regenerate the pinned requirements.txt.

The competition rules ask for a version-pinned requirements file. ``uv export``
writes the pins but not the extra package index that the CPU build of torch
comes from, so a plain ``pip install -r requirements.txt`` would fail to find
``torch==...+cpu``. This script adds that line back.

Usage::

    uv run python scripts/export_requirements.py
"""

from __future__ import annotations

import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
REQUIREMENTS = REPO_ROOT / "requirements.txt"
PYTORCH_CPU_INDEX = "https://download.pytorch.org/whl/cpu"

EXPORT_COMMAND = (
    "uv",
    "export",
    "--no-dev",
    "--no-hashes",
    "--no-emit-project",
    "--format",
    "requirements-txt",
)

HEADER = f"""# Regenerate with: uv run python scripts/export_requirements.py
#
# torch is pinned to the CPU build: development happens without a GPU, and
# Kaggle notebooks ship their own CUDA build that must not be replaced. To
# install a CUDA build instead, drop the two torch lines and follow
# https://pytorch.org/get-started/locally/
--extra-index-url {PYTORCH_CPU_INDEX}

"""


def main() -> int:
    exported = subprocess.run(
        EXPORT_COMMAND, cwd=REPO_ROOT, check=True, capture_output=True, text=True
    ).stdout
    # Drop uv's own header; it names a command that would lose the index line.
    body = "\n".join(line for line in exported.splitlines() if not line.startswith("#")).lstrip(
        "\n"
    )
    REQUIREMENTS.write_text(HEADER + body + "\n", encoding="utf-8")
    print(f"Wrote {REQUIREMENTS} ({len(body.splitlines())} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
