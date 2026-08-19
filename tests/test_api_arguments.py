"""Regression tests for high-level API to argparse serialization."""
from __future__ import annotations

import sys
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import opendpd.api as api
from opendpd.api import _append_keyword_arguments


def test_store_true_and_explicit_artifacts_have_valid_argv():
    argv = ["opendpd", "--step", "train_dpd"]
    _append_keyword_arguments(argv, {
        "quant": True,
        "plot": False,
        "use_segments": False,
        "pa_checkpoint": "/artifacts/pa.pt",
        "pa_output_checkpoint": "/artifacts/trained-pa.pt",
        "dpd_output_checkpoint": "/artifacts/fp32-dpd.pt",
        "qat_output_checkpoint": "/artifacts/qat.pt",
        "n_bits_a": 12,
    })
    assert "--quant" in argv
    assert "True" not in argv
    assert "False" not in argv
    assert "--plot" not in argv
    assert "--use_segments" not in argv
    assert argv[argv.index("--pa_checkpoint") + 1] == "/artifacts/pa.pt"
    assert argv[argv.index("--pa_output_checkpoint") + 1] == "/artifacts/trained-pa.pt"
    assert argv[argv.index("--dpd_output_checkpoint") + 1] == "/artifacts/fp32-dpd.pt"
    assert argv[argv.index("--qat_output_checkpoint") + 1] == "/artifacts/qat.pt"
    assert argv[argv.index("--n_bits_a") + 1] == "12"


def test_train_dpd_api_emits_quant_and_explicit_artifact_flags():
    fake_project = SimpleNamespace(
        path_save_file_best="legacy.pt",
        path_log_file_best="train.csv",
        published_qat_checkpoint="published.pt",
    )
    original_argv = sys.argv
    try:
        with (
            mock.patch.object(api, "Project", return_value=fake_project),
            mock.patch.object(api.train_dpd_module, "main"),
        ):
            result = api.train_dpd(
                dataset_name="unit",
                quant=True,
                plot=False,
                pa_checkpoint="/artifacts/pa.pt",
                qat_output_checkpoint="/artifacts/qat.pt",
                n_bits_a=12,
                n_bits_w=12,
            )
            argv = list(sys.argv)
    finally:
        sys.argv = original_argv
    assert result["model_path"] == "published.pt"
    assert argv.count("--quant") == 1
    assert "True" not in argv and "False" not in argv
    assert argv[argv.index("--pa_checkpoint") + 1] == "/artifacts/pa.pt"
    assert argv[argv.index("--qat_output_checkpoint") + 1] == "/artifacts/qat.pt"


def test_create_dataset_writes_canonical_sample_rate_and_rejects_conflict():
    import pandas as pd

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        csv_path = root / "samples.csv"
        pd.DataFrame({
            "I_in": [0.0, 0.1], "Q_in": [0.0, -0.1],
            "I_out": [0.0, 0.2], "Q_out": [0.0, -0.2],
        }).to_csv(csv_path, index=False)
        dataset = Path(api.create_dataset(
            str(csv_path), str(root), "unit", input_signal_fs=800e6,
        ))
        spec = json.loads((dataset / "spec.json").read_text())
        assert spec["sample_rate_hz"] == 800e6
        assert spec["input_signal_fs"] == 800e6
        try:
            api.create_dataset(
                str(csv_path), str(root), "bad",
                sample_rate_hz=800e6, input_signal_fs=700e6,
            )
        except ValueError as error:
            assert "must match" in str(error)
        else:
            raise AssertionError("conflicting sample-rate aliases were accepted")


if __name__ == "__main__":
    test_store_true_and_explicit_artifacts_have_valid_argv()
    test_train_dpd_api_emits_quant_and_explicit_artifact_flags()
    test_create_dataset_writes_canonical_sample_rate_and_rejects_conflict()
    print("PASS test_store_true_and_explicit_artifacts_have_valid_argv")
