"""Build the portable source-and-web bundle from release inputs only."""

import argparse
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from zipfile import ZIP_DEFLATED, ZipFile

from start_workbench import FRONTEND, FRONTEND_CONFIG, ROOT, prepare_frontend

FIXED_FILES = (
    ".env.example",
    "requirements/core.lock.txt",
    "requirements/probes.txt",
    "tools/start_workbench.py",
    "tools/continue_expanded.py",
    "docs/WORKBENCH.md",
    "docs/THIRD_PARTY_NOTICES.md",
    "docs/experiments/QUERY_ABLATION_20261121.md",
    "docs/experiments/SPLIT_SIGNOFF_20261121.md",
    "docs/experiments/TERMINAL_DECISION_20261121.md",
    "dataverify/workbench/data/duplicate-amount.json",
    "dataverify/workbench/data/split-signoff.json",
    "dataverify/workbench/data/penguins-2009.json",
    "dataverify/workbench/data/palmer-penguins.csv",
    "dataverify/workbench/data/SOURCE.md",
)
START_HERE = """# DataVerify 工作台体验包

将 ZIP 完整解压到可写目录。在此目录打开终端，Windows 运行
`py -3.12 tools/start_workbench.py`，macOS / Linux 运行
`python3.12 tools/start_workbench.py`。首次需联网安装 Python 依赖。
终端显示 Uvicorn 启动后打开 http://127.0.0.1:8765/ 。按 Ctrl+C 结束。

包含 Python 源码、React 源码和已构建网页。示例与回放无需模型密钥。
先试 Palmer Penguins 的“统计有体重测量值的观测记录数”，结果应为 342。
更多操作见 [工作台指南](docs/WORKBENCH.md)。
研究实验、完整测试和项目历史见 [项目仓库](https://github.com/repeatant/task-verified-data-agent)。
"""


def public_files():
    for name in FIXED_FILES:
        yield ROOT / name
    yield from (ROOT / "dataverify").rglob("*.py")
    for name in FRONTEND_CONFIG:
        yield FRONTEND / name
    for ext in ("*.ts", "*.tsx", "*.css"):
        yield from (FRONTEND / "src").rglob(ext)
    yield FRONTEND / "dist/index.html"
    yield FRONTEND / "dist/.workbench-build"
    for ext in ("*.css", "*.js"):
        yield from (FRONTEND / "dist/assets").glob(ext)


def build_archive(destination: Path):
    prepare_frontend()
    modules = FRONTEND / "node_modules"
    react_dom = (modules / "react-dom").resolve()
    licenses = {
        "react": modules / "react/LICENSE",
        "react-dom": react_dom / "LICENSE",
        "scheduler": react_dom.parent / "scheduler/LICENSE",
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=destination.parent, suffix=".tmp", delete=False) as staged:
        tmp = Path(staged.name)
    try:
        with ZipFile(tmp, "w", compression=ZIP_DEFLATED) as archive:
            archive.writestr("README.md", START_HERE)
            for path in sorted(set(public_files())):
                if path.is_symlink() or not path.resolve().is_relative_to(ROOT):
                    raise ValueError(f"Unexpected link in release source: {path}")
                archive.write(path, path.relative_to(ROOT).as_posix())
            for name, license_path in licenses.items():
                archive.write(license_path, f"licenses/{name}.txt")
        os.replace(tmp, destination)
    finally:
        tmp.unlink(missing_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path, help="output .zip path")
    args = parser.parse_args()
    if args.destination.suffix.lower() != ".zip":
        parser.error("destination must have the .zip extension")
    build_archive(args.destination)
    print(args.destination.resolve())
