"""Create a clean submission archive without secrets or runtime caches."""

from __future__ import annotations

import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT.parent / "课程资料问答智能体_提交版.zip"
EXCLUDED_PARTS = {
    ".env",
    "__pycache__",
    ".pytest_cache",
    "vector_store",
    "materials",
    "chat_history.sqlite3",
    "index_manifest.json",
}
EXCLUDED_SUFFIXES = {".pyc", ".pyo", ".log"}


def should_include(path: Path) -> bool:
    relative = path.relative_to(ROOT)
    if any(part in EXCLUDED_PARTS for part in relative.parts):
        return False
    return path.suffix.lower() not in EXCLUDED_SUFFIXES


def main() -> None:
    with zipfile.ZipFile(OUTPUT, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(ROOT.rglob("*")):
            if path.is_file() and should_include(path):
                archive.write(path, Path("agent_project") / path.relative_to(ROOT))
    print(f"已生成：{OUTPUT}")


if __name__ == "__main__":
    main()

