"""Regression tests for the dataset bundle shipped with the OpenDPD fork."""

from __future__ import annotations

import csv
import gc
import json
from pathlib import Path
import sys
import unittest

import numpy as np


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from modules.data_collector import load_dataset


DATASETS_ROOT = REPO_ROOT / "datasets"
SPLIT_DATASETS = {
    "APA_200MHz": (58_980, 19_662, 19_662),
    "APA_200MHz_b": (58_980, 19_662, 19_662),
    "DPA_160MHz": (294_912, 98_304, 98_304),
    "DPA_200MHz": (23_040, 7_680, 7_680),
}
SINGLE_DATASETS = {
    "MyCustomPA": (58_982, 19_660, 19_662),
}
SPLIT_FILES = tuple(
    f"{split}_{role}.csv"
    for split in ("train", "val", "test")
    for role in ("input", "output")
)


class DatasetBundleTests(unittest.TestCase):
    def _read_spec(self, dataset_dir: Path, expected_format: str) -> dict:
        spec_path = dataset_dir / "spec.json"
        self.assertTrue(spec_path.is_file(), f"missing {spec_path}")
        with spec_path.open(encoding="utf-8") as handle:
            spec = json.load(handle)
        self.assertIsInstance(spec, dict)
        self.assertEqual(spec.get("dataset_format"), expected_format)
        return spec

    def _assert_csv_has_data(self, path: Path, expected_header: tuple[str, ...]) -> None:
        self.assertTrue(path.is_file(), f"missing {path}")
        self.assertGreater(path.stat().st_size, 0, f"empty {path}")
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            self.assertEqual(tuple(next(reader, ())), expected_header, str(path))
            first_data_row = next(reader, None)
        self.assertIsNotNone(first_data_row, f"no data rows in {path}")
        self.assertEqual(len(first_data_row), len(expected_header), str(path))

    def _assert_loaded_arrays(
        self,
        dataset_dir: Path,
        expected_lengths: tuple[int, int, int],
    ) -> None:
        arrays = load_dataset(dataset_path=str(dataset_dir))
        self.assertEqual(len(arrays), 6)
        for split_index, (split, expected_length) in enumerate(
            zip(("train", "val", "test"), expected_lengths)
        ):
            inputs = arrays[split_index * 2]
            outputs = arrays[split_index * 2 + 1]
            with self.subTest(dataset=dataset_dir.name, split=split):
                self.assertIsInstance(inputs, np.ndarray)
                self.assertIsInstance(outputs, np.ndarray)
                self.assertEqual(inputs.shape, (expected_length, 2))
                self.assertEqual(outputs.shape, (expected_length, 2))
                self.assertTrue(np.issubdtype(inputs.dtype, np.number))
                self.assertTrue(np.issubdtype(outputs.dtype, np.number))
                self.assertTrue(np.isfinite(inputs).all())
                self.assertTrue(np.isfinite(outputs).all())
        del arrays
        gc.collect()

    def test_split_csv_dataset_bundle_loads_one_dataset_at_a_time(self):
        for dataset_name, expected_lengths in SPLIT_DATASETS.items():
            with self.subTest(dataset=dataset_name):
                dataset_dir = DATASETS_ROOT / dataset_name
                self.assertTrue(dataset_dir.is_dir(), f"missing {dataset_dir}")
                self._read_spec(dataset_dir, "split_csv")
                for filename in SPLIT_FILES:
                    self._assert_csv_has_data(
                        dataset_dir / filename,
                        ("I", "Q"),
                    )
                self._assert_loaded_arrays(dataset_dir, expected_lengths)

    def test_single_csv_dataset_bundle_loads(self):
        for dataset_name, expected_lengths in SINGLE_DATASETS.items():
            with self.subTest(dataset=dataset_name):
                dataset_dir = DATASETS_ROOT / dataset_name
                self.assertTrue(dataset_dir.is_dir(), f"missing {dataset_dir}")
                spec = self._read_spec(dataset_dir, "single_csv")
                self.assertEqual(spec.get("csv_filename"), "data.csv")
                self._assert_csv_has_data(
                    dataset_dir / "data.csv",
                    ("I_in", "Q_in", "I_out", "Q_out"),
                )
                self._assert_loaded_arrays(dataset_dir, expected_lengths)

    def test_matlab_signal_generation_helper_is_present(self):
        helper = DATASETS_ROOT / "MATLAB" / "signal_generation" / "iterative_match.py"
        self.assertTrue(helper.is_file(), f"missing {helper}")


if __name__ == "__main__":
    unittest.main()
