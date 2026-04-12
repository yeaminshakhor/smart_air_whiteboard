from __future__ import annotations
import argparse
import os
import py_compile
from pathlib import Path
from typing import Iterable


EXCLUDE_PARTS = {".venv", ".venv313", "__pycache__", ".git"}


def iter_python_files(root: Path) -> Iterable[Path]:
    for path in root.rglob("*.py"):
        if any(part in EXCLUDE_PARTS for part in path.parts):
            continue
        yield path


def compile_workspace(root: Path) -> list[tuple[Path, str]]:
    failures: list[tuple[Path, str]] = []
    for py_file in iter_python_files(root):
        try:
            py_compile.compile(str(py_file), doraise=True)
        except py_compile.PyCompileError as exc:
            failures.append((py_file, str(exc)))
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description="Compile-check workspace Python files")
    parser.add_argument("--root", default=".", help="Workspace root path")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    failures = compile_workspace(root)

    import logging
    logger = logging.getLogger(__name__)
    if not failures:
        logger.info(f"OK: compiled all project Python files under {root}")
        return 0

    logger.error(f"FAIL: {len(failures)} file(s) failed to compile")
    for file_path, message in failures:
        rel = os.path.relpath(file_path, root)
        logger.error(f"--- {rel} ---\n{message}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
