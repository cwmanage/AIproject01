#!/usr/bin/env python3
"""Cross-platform one-click launcher for the Titanic AI coursework project.

What it does (idempotent — safe to run again and again):
    1. Resolve the project root (this file's parent directory)
    2. Locate the project virtual environment (.venv); create it if missing
       (uses `python -m venv`, falling back to `uv venv` if available)
    3. Install dependencies with the best available tool:
         uv sync  ->  uv pip install  ->  python -m pip install -r requirements.txt
       (requirements.txt is generated from pyproject.toml if missing)
    4. Train the models + regenerate charts if outputs/ is empty
       (python -m titanic.train — also validates that every dependency works)
    5. Start the FastAPI server (uvicorn) and print the URLs

Run it with the Python you want to use:
    python start.py                 # Windows / macOS / Linux (any platform)
    py -3 start.py                  # Windows, if `python` is not on PATH
On Windows you can also double-click `start.bat` (it simply calls this file).

Ctrl+C stops the server. On Windows the terminal window stays open afterwards.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
VENV_DIR = PROJECT_ROOT / ".venv"
REQ_FILE = PROJECT_ROOT / "requirements.txt"
PYPROJECT = PROJECT_ROOT / "pyproject.toml"

# Files that mark "project already set up" -> we can skip the heavy steps
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
METRICS_CSV = OUTPUTS_DIR / "csv" / "metrics_all_models.csv"

HOST = os.environ.get("TITANIC_HOST", "127.0.0.1")
PORT = int(os.environ.get("TITANIC_PORT", "8000"))


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

def log(msg: str) -> None:
    """Print a message that is visible in every OS console (no fancy escapes)."""
    print(f"[start.py] {msg}", flush=True)


def run(cmd: list[str], cwd: Path = PROJECT_ROOT) -> None:
    """Run a command and stream its output; abort on non-zero exit."""
    log("$ " + " ".join(str(c) for c in cmd))
    result = subprocess.run(cmd, cwd=str(cwd))
    if result.returncode != 0:
        log(f"FATAL: command failed with exit code {result.returncode} -> aborting.")
        sys.exit(result.returncode)


def get_python() -> Path:
    """Return the interpreter we use for env creation / pip / scripts."""
    # Prefer the venv's own interpreter if it already exists
    if (VENV_DIR / "Scripts" / "python.exe").exists():      # Windows layout
        return VENV_DIR / "Scripts" / "python.exe"
    if (VENV_DIR / "bin" / "python").exists():              # POSIX layout
        return VENV_DIR / "bin" / "python"
    # No venv yet -> use the interpreter that is running this script
    return Path(sys.executable)


def ensure_venv() -> Path:
    """Create the virtual environment if it does not exist yet.

    Returns the venv interpreter. When a .venv already exists (e.g. created
    earlier by uv with Python 3.12), it is reused as-is — no version checks
    are needed because the environment was already validated by training.
    """
    if get_python().parent.parent != VENV_DIR:  # not created yet
        # We are about to create the venv with the launcher's Python:
        # dependencies (numpy 2.5 / pandas 3.0) require Python 3.12+.
        if sys.version_info < (3, 12):
            log("FATAL: Python 3.12+ is required to CREATE the venv "
                "(numpy 2.5 / pandas 3.0 need it). Found " + sys.version.split()[0])
            sys.exit(1)
        log("Creating virtual environment (.venv) ...")
        venv_py = sys.executable
        if shutil.which("uv"):                  # uv is fastest when available
            run([shutil.which("uv"), "venv", str(VENV_DIR)])
        else:
            run([venv_py, "-m", "venv", str(VENV_DIR)])
    return get_python()


def ensure_requirements_file() -> None:
    """Generate requirements.txt from pyproject.toml if it is missing."""
    if REQ_FILE.exists():
        return
    if not PYPROJECT.exists():
        log("No pyproject.toml found -> nothing to install; skipping.")
        return
    log("Generating requirements.txt from pyproject.toml ...")
    try:
        # pandas/sklearn/matplotlib/jinja2 live under [project].dependencies
        import tomllib
        with open(PYPROJECT, "rb") as fh:
            deps = tomllib.load(fh).get("project", {}).get("dependencies", [])
        # also include optional-dependencies if any exist
        for group in tomllib.load(open(PYPROJECT, "rb")).get("project", {}).get(
            "optional-dependencies", {}
        ).values():
            deps.extend(group)
        REQ_FILE.write_text("\n".join(deps) + "\n", encoding="utf-8")
    except Exception as exc:  # pragma: no cover - defensive
        log(f"Could not parse pyproject.toml ({exc}); using static requirements.")
        REQ_FILE.write_text(
            "fastapi\nuvicorn\npandas\nnumpy\nscikit-learn\nmatplotlib\njoblib\njinja2\n",
            encoding="utf-8",
        )


def install_dependencies(python: Path) -> None:
    """Install dependencies with uv if possible, otherwise pip."""
    uv = shutil.which("uv")
    ensure_requirements_file()
    if uv and PYPROJECT.exists():
        log("Installing dependencies with `uv sync` ...")
        run([uv, "sync", "--project", str(PYPROJECT)])
        return
    # Fallback 1: pip install -r requirements.txt inside the venv
    if REQ_FILE.exists():
        log("Installing dependencies with pip (requirements.txt) ...")
        run([str(python), "-m", "pip", "install", "--upgrade", "pip"])
        run([str(python), "-m", "pip", "install", "-r", str(REQ_FILE)])
        return
    # Fallback 2: pip install the raw list
    log("Installing core dependencies with pip ...")
    run([str(python), "-m", "pip", "install",
         "fastapi", "uvicorn", "pandas", "numpy",
         "scikit-learn", "matplotlib", "joblib", "jinja2"])


def train_if_needed(python: Path) -> None:
    """Run the full pipeline (charts + models + CSVs) on the first launch."""
    if METRICS_CSV.exists():
        log("Training artifacts already present in outputs/ -> skipping training.")
        return
    log("No training artifacts found -> running `python -m titanic.train` ...")
    run([str(python), "-m", "titanic.train"])


def wait_for_server(url: str, timeout: float = 30.0) -> bool:
    """Poll the health endpoint until the server responds (or timeout)."""
    import urllib.request
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url + "/api/health", timeout=2) as resp:
                return resp.status == 200
        except Exception:
            time.sleep(0.8)
    return False


def serve(python: Path) -> None:
    """Start uvicorn and wait for it to come up."""
    log(f"Starting FastAPI server on http://{HOST}:{PORT} ...")
    cmd = [str(python), "-m", "uvicorn", "app:app",
           "--host", str(HOST), "--port", str(PORT)]
    log("$ " + " ".join(cmd))
    proc = subprocess.Popen(cmd, cwd=str(PROJECT_ROOT))
    if wait_for_server(f"http://{HOST}:{PORT}"):
        log("Server is UP. Press Ctrl+C to stop it.")
    else:
        log("Server did not answer yet — it may still be starting. Check the log above.")
    try:
        proc.wait()  # block until the user presses Ctrl+C
    except KeyboardInterrupt:
        log("Shutting down ...")
        proc.terminate()


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    log(f"Project root: {PROJECT_ROOT}")

    python = ensure_venv()
    install_dependencies(python)
    train_if_needed(python)
    serve(python)


if __name__ == "__main__":
    main()
