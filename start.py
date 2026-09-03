#!/usr/bin/env python3
"""泰坦尼克 AI 课程作业的跨平台一键启动器。

它做的事情(幂等——重复运行也安全)：
    1. 解析项目根目录(本文件所在目录)
    2. 找到项目虚拟环境(.venv)；不存在就创建
       (用 `python -m venv`，有 uv 则退回 `uv venv`)
    3. 用可用的最佳工具安装依赖：
         uv sync  ->  uv pip install  ->  python -m pip install -r requirements.txt
       (requirements.txt 缺失时会从 pyproject.toml 生成)
    4. 如果 outputs/ 为空则训练模型 + 重新生成图
       (python -m titanic.train —— 顺便验证所有依赖可用)
    5. 启动 FastAPI 服务(uvicorn)并打印访问地址

用你想用的 Python 运行它：
    python start.py                 # Windows / macOS / Linux (任何平台)
    py -3 start.py                  # Windows：如果 `python` 不在 PATH 里
Windows 上也可以双击 `start.bat`(它只是调用本文件)。

Ctrl+C 停止服务。Windows 上终端窗口之后会保持打开。
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
VENV_DIR = PROJECT_ROOT / ".venv"
REQ_FILE = PROJECT_ROOT / "requirements.txt"
PYPROJECT = PROJECT_ROOT / "pyproject.toml"

# 标记"项目已初始化"的文件 -> 可以跳过耗时的步骤
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
METRICS_CSV = OUTPUTS_DIR / "csv" / "metrics_all_models.csv"

HOST = os.environ.get("TITANIC_HOST", "127.0.0.1")
PORT = int(os.environ.get("TITANIC_PORT", "8000"))


# ---------------------------------------------------------------------------
# 小工具函数
# ---------------------------------------------------------------------------

def log(msg: str) -> None:
    """打印消息，任何操作系统的控制台都可见(不用花哨的转义符)。"""
    print(f"[start.py] {msg}", flush=True)


def run(cmd: list[str], cwd: Path = PROJECT_ROOT) -> None:
    """运行命令并流式输出；非零退出码则中止。"""
    log("$ " + " ".join(str(c) for c in cmd))
    result = subprocess.run(cmd, cwd=str(cwd))
    if result.returncode != 0:
        log(f"FATAL: command failed with exit code {result.returncode} -> aborting.")
        sys.exit(result.returncode)


def get_python() -> Path:
    """返回用于创建环境 / 装包 / 跑脚本的解释器。"""
    # 优先用虚拟环境自己的解释器(如果已存在)
    if (VENV_DIR / "Scripts" / "python.exe").exists():      # Windows 布局
        return VENV_DIR / "Scripts" / "python.exe"
    if (VENV_DIR / "bin" / "python").exists():              # POSIX 布局
        return VENV_DIR / "bin" / "python"
    # 还没有虚拟环境 -> 用正在运行本脚本的解释器
    return Path(sys.executable)


def ensure_venv() -> Path:
    """如果虚拟环境不存在则创建它。

    返回 venv 的解释器。如果 .venv 已存在(比如之前用 uv 建的
    Python 3.12)，原样复用——不需要检查版本，因为训练早就验证过
    这个环境是可用的。
    """
    if get_python().parent.parent != VENV_DIR:  # 还没创建
        # 我们即将用启动器的 Python 创建 venv：
        # 依赖(numpy 2.5 / pandas 3.0)要求 Python 3.12+。
        if sys.version_info < (3, 12):
            log("FATAL: Python 3.12+ is required to CREATE the venv "
                "(numpy 2.5 / pandas 3.0 need it). Found " + sys.version.split()[0])
            sys.exit(1)
        log("Creating virtual environment (.venv) ...")
        venv_py = sys.executable
        if shutil.which("uv"):                  # 有 uv 时它最快
            run([shutil.which("uv"), "venv", str(VENV_DIR)])
        else:
            run([venv_py, "-m", "venv", str(VENV_DIR)])
    return get_python()


def ensure_requirements_file() -> None:
    """如果 requirements.txt 缺失，则从 pyproject.toml 生成。"""
    if REQ_FILE.exists():
        return
    if not PYPROJECT.exists():
        log("No pyproject.toml found -> nothing to install; skipping.")
        return
    log("Generating requirements.txt from pyproject.toml ...")
    try:
        # pandas/sklearn/matplotlib/jinja2 都在 [project].dependencies 下
        import tomllib
        with open(PYPROJECT, "rb") as fh:
            deps = tomllib.load(fh).get("project", {}).get("dependencies", [])
        # 如果有 optional-dependencies 也一并包含
        for group in tomllib.load(open(PYPROJECT, "rb")).get("project", {}).get(
            "optional-dependencies", {}
        ).values():
            deps.extend(group)
        REQ_FILE.write_text("\n".join(deps) + "\n", encoding="utf-8")
    except Exception as exc:  # pragma: no cover - 防御性兜底
        log(f"Could not parse pyproject.toml ({exc}); using static requirements.")
        REQ_FILE.write_text(
            "fastapi\nuvicorn\npandas\nnumpy\nscikit-learn\nmatplotlib\njoblib\njinja2\n",
            encoding="utf-8",
        )


def install_dependencies(python: Path) -> None:
    """装依赖：能用 uv 就用 uv，否则退回 pip。"""
    uv = shutil.which("uv")
    ensure_requirements_file()
    if uv and PYPROJECT.exists():
        log("Installing dependencies with `uv sync` ...")
        run([uv, "sync", "--project", str(PYPROJECT)])
        return
    # 兜底 1：在 venv 里 pip install -r requirements.txt
    if REQ_FILE.exists():
        log("Installing dependencies with pip (requirements.txt) ...")
        run([str(python), "-m", "pip", "install", "--upgrade", "pip"])
        run([str(python), "-m", "pip", "install", "-r", str(REQ_FILE)])
        return
    # 兜底 2：pip 直接装核心依赖清单
    log("Installing core dependencies with pip ...")
    run([str(python), "-m", "pip", "install",
         "fastapi", "uvicorn", "pandas", "numpy",
         "scikit-learn", "matplotlib", "joblib", "jinja2"])


def train_if_needed(python: Path) -> None:
    """首次启动时跑完整流程(图 + 模型 + CSV)。"""
    if METRICS_CSV.exists():
        log("Training artifacts already present in outputs/ -> skipping training.")
        return
    log("No training artifacts found -> running `python -m titanic.train` ...")
    run([str(python), "-m", "titanic.train"])


def wait_for_server(url: str, timeout: float = 30.0) -> bool:
    """轮询健康检查端点直到服务响应(或超时)。"""
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
    """启动 uvicorn 并等它起来。"""
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
        proc.wait()  # 阻塞直到用户按 Ctrl+C
    except KeyboardInterrupt:
        log("Shutting down ...")
        proc.terminate()


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main() -> None:
    log(f"Project root: {PROJECT_ROOT}")

    python = ensure_venv()
    install_dependencies(python)
    train_if_needed(python)
    serve(python)


if __name__ == "__main__":
    main()
