"""Shared fixtures and helpers for the OpenDPD test suite.

All training smoke tests run the real pipeline end-to-end on the smallest
built-in dataset (DPA_200MHz) with tiny hyperparameters so that a full
train_pa -> train_dpd -> run_dpd -> plot chain completes in seconds on a
CPU-only machine.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
MAIN_PY = REPO_ROOT / "main.py"

# Tiny hyperparameters shared by all smoke tests: 1 epoch, short frames,
# large stride (fewer frames), small batches. Keeps each step under ~30 s
# on a 2-core GitHub Actions runner.
SMOKE_ARGS = [
    "--accelerator", "cpu",
    "--n_epochs", "1",
    "--frame_length", "50",
    "--frame_stride", "16",
    "--batch_size", "64",
    "--batch_size_eval", "256",
]

SMOKE_DATASET = "DPA_200MHz"


def run_main(workdir, step, *extra_args, dataset=SMOKE_DATASET, timeout=600):
    """Run ``python main.py --step <step> ...`` in ``workdir`` and return the
    completed process. Fails the test with the captured output if the exit
    code is nonzero."""
    cmd = [
        sys.executable,
        str(MAIN_PY),
        "--dataset_name", dataset,
        "--step", step,
        *SMOKE_ARGS,
        *extra_args,
    ]
    env = dict(os.environ, MPLBACKEND="Agg", KMP_DUPLICATE_LIB_OK="TRUE")
    result = subprocess.run(
        cmd,
        cwd=workdir,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if result.returncode != 0:
        pytest.fail(
            f"'{' '.join(cmd)}' exited with {result.returncode}\n"
            f"--- stdout (tail) ---\n{result.stdout[-3000:]}\n"
            f"--- stderr (tail) ---\n{result.stderr[-3000:]}"
        )
    return result


@pytest.fixture(scope="session")
def e2e_workdir(tmp_path_factory):
    """A working directory shared by the chained E2E smoke tests. Training
    artifacts (save/, log/, dpd_out/, plots/) are written here, keeping the
    repository clean."""
    return tmp_path_factory.mktemp("e2e")


@pytest.fixture(scope="session")
def pa_trained(e2e_workdir):
    """Run PA modeling once for the whole session; later stages reuse it."""
    run_main(e2e_workdir, "train_pa")
    return e2e_workdir


@pytest.fixture(scope="session")
def dpd_trained(pa_trained):
    """Run DPD learning on top of the trained PA model."""
    run_main(pa_trained, "train_dpd")
    return pa_trained
