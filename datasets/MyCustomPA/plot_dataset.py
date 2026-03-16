#!/usr/bin/env python
"""Quick visualization of the dataset. Run from the project root:

    python datasets/DATASET_NAME/plot_dataset.py

Generates waveform, PSD, constellation, AM/AM, AM/PM plots
saved directly in this folder.
"""
import os, sys
# Ensure project root is on sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from datasets.plot_utils import plot_dataset


if __name__ == '__main__':
    plot_dataset('MyCustomPA')
