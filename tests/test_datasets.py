"""Integrity checks for the built-in datasets shipped with the repository.

A dataset is a directory under datasets/ containing a spec.json plus I/Q
sample CSVs — either paired train/val/test input/output files
(dataset_format: split_csv) or a single combined file (single_csv). These
tests catch corrupted or truncated data files and malformed specs before
they break training runs.
"""

import json

import numpy as np
import pandas as pd
import pytest

from conftest import REPO_ROOT

DATASETS_DIR = REPO_ROOT / "datasets"
SPLITS = ("train", "val", "test")


def load_spec(dataset):
    return json.loads((DATASETS_DIR / dataset / "spec.json").read_text())


def builtin_datasets():
    return sorted(
        d.parent.name for d in DATASETS_DIR.glob("*/spec.json") if d.is_file()
    )


def split_csv_datasets():
    return [d for d in builtin_datasets() if load_spec(d).get("dataset_format") == "split_csv"]


def single_csv_datasets():
    return [d for d in builtin_datasets() if load_spec(d).get("dataset_format") == "single_csv"]


def assert_finite_frame(frame, name, columns):
    assert list(frame.columns) == list(columns), (
        f"{name} must have exactly {list(columns)} columns, got {list(frame.columns)}"
    )
    assert len(frame) > 0, f"{name} is empty"
    assert np.isfinite(frame.to_numpy()).all(), f"{name} contains NaN/Inf"


@pytest.mark.parametrize("dataset", builtin_datasets())
def test_spec_is_valid_json(dataset):
    spec = load_spec(dataset)
    assert spec.get("dataset_format") in ("split_csv", "single_csv")
    assert "input_signal_fs" in spec, "spec.json missing input_signal_fs"
    assert float(spec["input_signal_fs"]) > 0


@pytest.mark.parametrize("split", SPLITS)
@pytest.mark.parametrize("dataset", split_csv_datasets())
def test_split_files_are_paired_iq_csv(dataset, split):
    dataset_dir = DATASETS_DIR / dataset
    input_file = dataset_dir / f"{split}_input.csv"
    output_file = dataset_dir / f"{split}_output.csv"
    assert input_file.is_file(), f"missing {input_file.name}"
    assert output_file.is_file(), f"missing {output_file.name}"

    x = pd.read_csv(input_file)
    y = pd.read_csv(output_file)
    assert_finite_frame(x, input_file.name, ["I", "Q"])
    assert_finite_frame(y, output_file.name, ["I", "Q"])
    assert len(x) == len(y), (
        f"{split} input ({len(x)} rows) and output ({len(y)} rows) differ"
    )


@pytest.mark.parametrize("dataset", single_csv_datasets())
def test_single_csv_dataset(dataset):
    spec = load_spec(dataset)
    data_file = DATASETS_DIR / dataset / spec["csv_filename"]
    assert data_file.is_file(), f"missing {data_file.name}"

    data = pd.read_csv(data_file)
    assert_finite_frame(data, data_file.name, ["I_in", "Q_in", "I_out", "Q_out"])

    split_indices = spec["split_indices"]
    assert 0 < split_indices["train_end"] < split_indices["val_end"] <= len(data), (
        "split_indices must be increasing and within the data length"
    )


def test_at_least_four_datasets_present():
    assert len(builtin_datasets()) >= 4, (
        f"expected the built-in datasets to ship with the repo, "
        f"found only {builtin_datasets()}"
    )
