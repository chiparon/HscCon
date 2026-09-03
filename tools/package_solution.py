"""Build the competition submission archive with a stable root layout."""

from __future__ import annotations

import argparse
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOLUTION_PATH = REPOSITORY_ROOT / "solution.py"


def build_archive(output: Path) -> None:
    """Write ``solution.py`` as the sole root entry in *output*."""
    if not SOLUTION_PATH.is_file():
        raise FileNotFoundError(f"submission entry point not found: {SOLUTION_PATH}")

    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    # A fixed timestamp and explicit permissions make equal sources reproducible.
    entry = ZipInfo("solution.py", date_time=(1980, 1, 1, 0, 0, 0))
    entry.compress_type = ZIP_DEFLATED
    entry.external_attr = 0o644 << 16
    with ZipFile(output, "w") as archive:
        archive.writestr(entry, SOLUTION_PATH.read_bytes())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPOSITORY_ROOT / "solution.zip",
        help="archive path (default: repository-root solution.zip)",
    )
    args = parser.parse_args()
    build_archive(args.output)
    print(f"created {args.output.resolve()} containing solution.py")


if __name__ == "__main__":
    main()
