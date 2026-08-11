[English](README_TCN_QAT.md) | [한국어](README_TCN_QAT_KO.md)

# Bundled PA Dataset Catalog

This directory contains the PA input/output fixtures distributed with the
OpenDPD-TCN-QAT frontend. The measured input and output rows are time-aligned
complex-baseband I/Q samples. The CSV values remain floating point at the data
boundary; a QAT model applies its configured fake quantization after loading.

`DPA_200MHz` is the dataset pinned by the checked-in DPD-Flow
configurations. The other datasets broaden frontend experiments and exercise
the alternate storage and demodulation paths; including them does not make them
qualified DPD-Flow PPA or DPD-quality baselines.

## Directory layout

```text
datasets/
├── APA_200MHz/          # Split CSV, CP-bearing LTE/OFDM, measurement A
├── APA_200MHz_b/        # Split CSV, CP-bearing LTE/OFDM, measurement B
├── DPA_160MHz/          # Split CSV, no-CP IFFT frames
├── DPA_200MHz/          # Split CSV, qualified DPD-Flow fixture
├── MyCustomPA/          # Single-CSV custom-dataset example
├── MATLAB/
│   └── signal_generation/
│       ├── iterative_match.py
│       └── test_iterative_match.py
├── demodulator.py       # Shared demodulator base classes and factory
└── plot_utils.py        # Shared diagnostic-plot generator
```

Every PA dataset directory has a `spec.json`, a dataset-specific `demod.py`,
and a `plot_dataset.py` wrapper. Diagnostic PNGs are reproducible outputs of
the plotting wrapper, not required inputs or part of the dataset format.

## Data formats and API paths

### Split CSV format

`APA_200MHz`, `APA_200MHz_b`, `DPA_160MHz`, and `DPA_200MHz` each use six
files:

```text
train_input.csv   train_output.csv
val_input.csv     val_output.csv
test_input.csv    test_output.csv
```

Every file has an `I,Q` header. An input file and its corresponding output file
have the same row count and are aligned sample by sample. The split files are
already materialized; consumers must not concatenate and randomly re-split
them when reproducing an experiment.

### Single CSV format

`MyCustomPA/data.csv` has the four columns
`I_in,Q_in,I_out,Q_out`. Its `spec.json` declares `dataset_format` as
`single_csv`, names `data.csv`, and records the ordered split boundaries. The
loader slices rows without shuffling: `[0:58982)` for training,
`[58982:78642)` for validation, and `[78642:98304)` for test.

### Public API paths

Run these examples from `frontend/OpenDPD-TCN-QAT/`. The public
`opendpd.load_dataset()` function takes a filesystem path and returns a
dictionary of NumPy arrays:

```python
import opendpd

data = opendpd.load_dataset("datasets/DPA_200MHz")
print(data["X_train"].shape, data["y_test"].shape)
```

In contrast, the training and inference APIs resolve bundled data by its
directory name:

```python
import opendpd

pa_result = opendpd.train_pa(
    dataset_name="DPA_200MHz",
    PA_backbone="gru",
    PA_hidden_size=23,
    n_epochs=100,
    accelerator="cuda",
)
```

The dataset-specific demodulator is selected through the shared factory:

```python
from datasets.demodulator import Demodulator

demodulator = Demodulator.from_dataset("APA_200MHz")
```

## Dataset catalog

| Entry | Format | Actual train / val / test samples | Signal and demodulator | Intended use |
|---|---|---:|---|---|
| `APA_200MHz` | `split_csv` | 58,980 / 19,662 / 19,662 | 5-carrier LTE TM3.1a, CP-aware `OFDMCPDemodulator` | APA measurement A; CP synchronization, equalization, and standard-OFDM experiments |
| `APA_200MHz_b` | `split_csv` | 58,980 / 19,662 / 19,662 | Same waveform class and demodulator as measurement A | Independent measurement B for comparison across captures |
| `DPA_160MHz` | `split_csv` | 294,912 / 98,304 / 98,304 | 4-carrier no-CP IFFT frames, `IFFTFrameDemodulator` | Large-frame, 1024QAM, 160 MHz DPA experiments |
| `DPA_200MHz` | `split_csv` | 23,040 / 7,680 / 7,680 | 10-carrier no-CP IFFT frames, `IFFTFrameDemodulator` | Checked-in DPD-Flow training, QAT, integer, and RTL comparison baseline |
| `MyCustomPA` | `single_csv` | 58,982 / 19,660 / 19,662 | Single-channel example, `IFFTFrameDemodulator` | Demonstrates the custom single-CSV import contract; not a qualified baseline |
| `MATLAB/signal_generation` | Python helper, no PA CSV split | Not applicable | Fixed 5-carrier MATLAB-reference signal matcher | Signal-generation matching research; not consumed by PA/DPD training |

The counts above are the physical CSV row counts. The four split datasets also
declare nominal `0.6/0.2/0.2` ratios in `spec.json`; the files themselves are
authoritative when integer rounding differs.

### APA_200MHz and APA_200MHz_b

Both APA entries contain 98,304 samples of a 5-carrier LTE TM3.1a waveform.
Each carrier occupies 20 MHz and carrier centers are spaced by 40 MHz. The
signal was generated at 491.52 MHz with 15 kHz subcarrier spacing and recorded
in the dataset metadata at 983.04 MHz, where the effective spacing is 30 kHz.
The occupied composite bandwidth is 200 MHz and the data modulation is
256QAM.

The two `spec.json` files use the same signal parameters:

| Field | Value |
|---|---:|
| `input_signal_fs` | 983.04 MHz |
| `bw_main_ch` | 200 MHz |
| `bw_sub_ch` | 40 MHz carrier spacing |
| `n_sub_ch` | 5 |
| `nperseg` | 19,662 samples |
| `ofdm_nfft` | 32,768 samples |
| `n_active` | 600 subcarriers per carrier |
| `cp_first` / `cp_other` | 2,560 / 2,304 samples |
| `scs` | 30 kHz effective |
| `papr_db` | 10.0 dB |

For APA data, `nperseg` is the PSD/evaluation segment length, not an OFDM FFT
frame. `OFDMCPDemodulator` isolates each carrier, finds symbol boundaries by
cyclic-prefix correlation, fine-tunes the FFT offset, extracts the active
subcarriers, and can equalize a PA output against the clean input reference.
The short validation and test splits do not each contain a complete
32,768-sample FFT symbol plus CP, so plotting code uses the complete joined
sequence when full-sequence APA constellation data is required.

`APA_200MHz_b` is a second capture of the same waveform class. It must remain a
separate dataset name; results from measurements A and B must not be merged or
presented as one split.

### DPA_160MHz

`DPA_160MHz` contains 491,520 samples from a 4-carrier, 160 MHz DPA waveform at
640 MHz sample rate. Each carrier occupies 40 MHz, the modulation recorded in
`spec.json` is 1024QAM, and each no-CP IFFT frame is 16,384 samples. From these
values, `IFFTFrameDemodulator` derives 1,024 active subcarriers per carrier.

The demodulator divides the signal into aligned 16,384-sample frames, applies
one FFT per frame, and directly selects bins around the four carrier centers.
This dataset is useful for stressing longer temporal segmentation, a larger
FFT frame, and higher-order modulation than the default DPA fixture.

### DPA_200MHz

`DPA_200MHz` contains 38,400 samples from a 10-carrier, 200 MHz DPA waveform at
800 MHz sample rate. Each carrier occupies 20 MHz, the modulation is 64QAM,
and the no-CP IFFT frame size is `nperseg=2560`. The demodulator consequently
derives 64 active subcarriers per carrier.

The six CSV splits and `spec.json` are the measured-data fixture referenced by
the checked-in H10/A12W12 and H13/A14W14 flow configurations. Qualified
segmented evaluation uses 2,560-sample boundaries, right-zero-pads an
incomplete final segment, and resets the integer TCN history and frozen PA
state at each boundary. A continuous-stream result has different state
semantics and must be labeled separately.

### MyCustomPA

`MyCustomPA` demonstrates the `single_csv` contract with 98,304 ordered rows.
Its example metadata describes one 200 MHz channel at 800 MHz sample rate with
`nperseg=2560`. It has no `modulation` field, so no modulation order should be
inferred from the directory name or sample values.

Create another dataset from a four-column measurement CSV with the public API:

```python
import opendpd

dataset_dir = opendpd.create_dataset(
    csv_path="/path/to/measurements.csv",
    output_dir="datasets",
    dataset_name="MyPA",
    dataset_format="single_csv",
    input_signal_fs=800e6,
    bw_main_ch=200e6,
    bw_sub_ch=200e6,
    n_sub_ch=1,
    nperseg=2560,
)
```

Use `dataset_format="split_csv"` to generate the six-file layout instead.
Before treating a new dataset as a baseline, pin its measurement provenance,
split boundaries, framing/reset protocol, signal metadata, and frozen PA
checkpoint.

### MATLAB signal-generation helper

`MATLAB/signal_generation/iterative_match.py` is a Python research helper for
matching a generated waveform to a MATLAB `.mat` reference. It is not a PA
measurement dataset, has no train/validation/test split, and is not loaded by
`opendpd.load_dataset()` or by the PA/DPD training APIs.

The helper is specialized to 98,304 samples of five 20 MHz carriers spanning
100 MHz at 491.52 MHz sample rate, with a 32,768-point FFT, 1,200 active
subcarriers, and a 2,304-sample normal CP. It performs an analytical warm
start and differential-evolution refinement, then checks NMSE, PSD mean
absolute error, per-channel power error, PAPR difference, CCDF deviation, and
EVM. `test_iterative_match.py` records unit and reference-signal expectations.

This directory is a source snapshot rather than a standalone supported CLI:
the helper imports companion `generate_signal.py` and `plot_comparison.py`
modules and defaults to a target `.mat` file, none of which are bundled here.
Supply those research inputs from their original environment before running
it. Its constants also describe a 100 MHz, 491.52 MHz-reference waveform and
must not be substituted for the 200 MHz APA dataset metadata above.

## Signal metadata and demodulation

`spec.json` is the machine-readable source for sample rate, bandwidth,
carrier count, segmentation, modulation, and format. Do not infer these values
from a directory name. `datasets.demodulator.Demodulator.from_dataset(name)`
loads that file and then imports `datasets.<name>.demod`.

The two implemented waveform families are:

- `IFFTFrameDemodulator`: aligned back-to-back IFFT frames without a cyclic
  prefix. `nperseg` must equal the generating IFFT size.
- `OFDMCPDemodulator`: standard CP-bearing OFDM. It uses `ofdm_nfft`,
  `cp_first`, `cp_other`, `scs`, and `n_active` from the APA specifications.

Demodulation affects constellation visualization and EVM interpretation. It
does not change the time-domain samples consumed by model training.

## Regenerating diagnostic plots

From `frontend/OpenDPD-TCN-QAT/`, run one dataset wrapper or all five:

```bash
python3 datasets/APA_200MHz/plot_dataset.py
python3 datasets/APA_200MHz_b/plot_dataset.py
python3 datasets/DPA_160MHz/plot_dataset.py
python3 datasets/DPA_200MHz/plot_dataset.py
python3 datasets/MyCustomPA/plot_dataset.py
```

Each wrapper can regenerate `waveform.png`, `psd.png`, `constellation.png`,
`amam.png`, and `ampm.png` beside the dataset. These PNGs are disposable
diagnostic outputs: their presence in a checkout is optional, and training,
evaluation, and RTL export must not depend on them. Plot generation reads but
does not modify the CSV measurements.
