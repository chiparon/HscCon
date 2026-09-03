from pathlib import Path
from zipfile import ZipFile

from tools.package_solution import SOLUTION_PATH, build_archive


def test_submission_archive_contains_only_root_solution(tmp_path: Path) -> None:
    output = tmp_path / "solution.zip"

    build_archive(output)

    with ZipFile(output) as archive:
        assert archive.namelist() == ["solution.py"]
        assert archive.read("solution.py") == SOLUTION_PATH.read_bytes()
