[English](UPSTREAM_AND_LICENSE.md) | [한국어](UPSTREAM_AND_LICENSE_KO.md)

# Upstream and License

This repository is an official GitHub fork and modified work based on commit
`98c2622cfef9ef537a250e118d811a6ae7333f53` of
[lab-emi/OpenDPD](https://github.com/lab-emi/OpenDPD). It retains the full
Apache License 2.0 text and upstream author attributions. Publication citations
remain available in the
[README at the pinned upstream commit](https://github.com/lab-emi/OpenDPD/blob/98c2622cfef9ef537a250e118d811a6ae7333f53/README.md).

The principal modifications are:

- causal FExLite TCN and topology metadata;
- Conv1d full-I/O QAT and train-only calibration;
- explicit, atomic publication of PA, FP32 DPD, and QAT training artifacts;
- exact-zero DGRU PA stabilization;
- portable integer RTL export and verifier;
- full-test integer DPD plus frozen-PA evaluator; and
- checkpoint-independent regression tests.

Existing OpenDPD datasets and third-party tools remain subject to their
respective distribution terms. Training checkpoints are not included in this
frontend Python package or its wheel. The DPD-Flow monorepo root may contain a
small reference checkpoint as a separately tracked artifact under
`artifacts/checkpoints/` for reproducibility checks. Synthesis libraries and EDA
binaries are not included in the source distribution.

The GitHub fork parent, official upstream URL, and source commit above form the
provenance record. The `tcn-qat` branch carries the DPD-Flow modifications;
upstream changes are fetched from `lab-emi/OpenDPD` and merged into that branch
only on a dedicated sync branch. Every sync must preserve and rerun the local
QAT, integer-export, frozen-PA, dataset, and regression contracts before the
DPD-Flow submodule pointer is advanced.
