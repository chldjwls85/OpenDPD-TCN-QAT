"""Tests for the command-line interfaces (python main.py and opendpd-cli)."""

import os
import subprocess
import sys

from conftest import MAIN_PY, REPO_ROOT


def run(cmd, **kwargs):
    env = dict(os.environ, MPLBACKEND="Agg")
    return subprocess.run(
        cmd, capture_output=True, text=True, timeout=120, env=env, **kwargs
    )


def test_main_py_help():
    result = run([sys.executable, str(MAIN_PY), "--help"])
    assert result.returncode == 0
    assert "--step" in result.stdout
    assert "--accelerator" in result.stdout


def test_cli_module_help():
    """python -m opendpd.cli mirrors the opendpd-cli console script."""
    result = run([sys.executable, "-m", "opendpd.cli", "--help"], cwd=REPO_ROOT)
    assert result.returncode == 0
    assert "--step" in result.stdout


def test_main_py_rejects_unknown_step():
    result = run([sys.executable, str(MAIN_PY), "--step", "not_a_step"])
    assert result.returncode != 0


def test_main_py_rejects_unknown_backbone():
    result = run(
        [sys.executable, str(MAIN_PY), "--step", "train_pa", "--PA_backbone", "bogus"]
    )
    assert result.returncode != 0
