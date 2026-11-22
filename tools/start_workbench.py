"""Prepare a local workbench from a source checkout, then serve it on loopback."""

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FRONTEND = ROOT / "frontend"
ENV = ROOT / ".venv-workbench"
FRONTEND_CONFIG = (
    "index.html",
    "package.json",
    "pnpm-lock.yaml",
    "pnpm-workspace.yaml",
    "tsconfig.json",
    "vite.config.ts",
)


def fingerprint(files):
    digest = hashlib.sha256()
    for file in sorted(files):
        digest.update(file.relative_to(ROOT).as_posix().encode() + b"\0")
        digest.update(hashlib.sha256(file.read_bytes()).digest())
    return digest.hexdigest()


def matches(stamp, expected):
    return stamp.is_file() and stamp.read_text(encoding="ascii") == expected


def prepare_python():
    python = ENV / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    locks = [ROOT / "requirements" / name for name in ("core.lock.txt", "probes.txt")]
    stamp = ENV / ".workbench-requirements"
    version = f"{sys.version_info.major}.{sys.version_info.minor}"
    expected = version + ":" + fingerprint(locks)
    if not python.is_file():
        subprocess.run([sys.executable, "-m", "venv", str(ENV)], check=True)
    if not matches(stamp, expected):
        installed = subprocess.check_output(
            [str(python), "-c", "import sys; print('%d.%d' % sys.version_info[:2])"], text=True
        ).strip()
        if installed != version:
            raise SystemExit(f"{ENV} uses Python {installed}. Run this command with that version.")
        subprocess.run(
            [str(python), "-m", "pip", "install", "-r", str(locks[0]), "-r", str(locks[1])],
            cwd=ROOT,
            check=True,
        )
        stamp.write_text(expected, encoding="ascii")
    return python


def prepare_frontend():
    files = [FRONTEND / name for name in FRONTEND_CONFIG]
    for source_dir in ("src", "public"):
        files.extend(p for p in (FRONTEND / source_dir).rglob("*") if p.is_file())
    expected = fingerprint(files)
    dist = FRONTEND / "dist"
    stamp = dist / ".workbench-build"
    if (dist / "index.html").is_file() and matches(stamp, expected):
        return
    pnpm = shutil.which("pnpm.cmd" if os.name == "nt" else "pnpm")
    if pnpm:
        command = [pnpm]
    else:
        npm = shutil.which("npm.cmd" if os.name == "nt" else "npm")
        if not npm:
            raise SystemExit("Install Node.js 22.12+ with npm, or add pnpm to PATH, then retry.")
        command = [npm, "exec", "--yes", "--package=pnpm@10", "--", "pnpm"]
    subprocess.run([*command, "install", "--frozen-lockfile"], cwd=FRONTEND, check=True)
    subprocess.run([*command, "run", "build"], cwd=FRONTEND, check=True)
    stamp.write_text(expected, encoding="ascii")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--prepare-only", action="store_true", help="install and build, then exit")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port must be in 1..65535")
    if sys.version_info < (3, 12):
        parser.error("Use Python 3.12 or later (Windows: py -3.12 tools/start_workbench.py).")
    python = prepare_python()
    prepare_frontend()
    if args.prepare_only:
        print("Workbench environment and frontend are ready.")
        return 0
    print(f"Starting DataVerify at http://127.0.0.1:{args.port}/", flush=True)
    return subprocess.call(
        [str(python), "-m", "dataverify.workbench", "--port", str(args.port)], cwd=ROOT
    )


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except subprocess.CalledProcessError as error:
        raise SystemExit(error.returncode) from None
    except KeyboardInterrupt:
        raise SystemExit(130) from None
