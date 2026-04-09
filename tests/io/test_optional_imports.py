from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _run_python(script: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    return subprocess.run(
        [sys.executable, "-c", script], check=False, capture_output=True, text=True, env=env
    )


def _block_imports(*blocked: str) -> str:
    blocked_names = repr(blocked)
    return textwrap.dedent(
        f"""
        import builtins

        _real_import = builtins.__import__
        _blocked = {blocked_names}

        def _guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
            if any(name == target or name.startswith(target + ".") for target in _blocked):
                raise ModuleNotFoundError(f"No module named '{{name.split('.')[0]}}'")
            return _real_import(name, globals, locals, fromlist, level)

        builtins.__import__ = _guarded_import
        """
    )


def test_reader_and_adapter_imports_are_lazy() -> None:
    """Importing reader and adapter packages should not require optional ML deps."""
    proc = _run_python(
        _block_imports("torch", "torch_geometric", "streaming")
        + textwrap.dedent(
            """
            import dronalize.io
            import dronalize.io.adapters
            import dronalize.io.readers

            print("ok")
            """
        )
    )

    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "ok"


def test_pyg_adapter_install_hint() -> None:
    """The PyG adapter should explain which extra to install."""
    proc = _run_python(
        _block_imports("torch_geometric")
        + textwrap.dedent(
            """
            try:
                from dronalize.io.adapters import MDSHeteroDataset
            except ModuleNotFoundError as exc:
                print(exc)
            else:
                raise SystemExit(f"unexpected success: {MDSHeteroDataset}")
            """
        )
    )

    assert proc.returncode == 0, proc.stderr
    assert "pip install dronalize[pyg]" in proc.stdout
