# DPD Benchmark Report: AI vs. Traditional Approaches

## 1. Overview

This report benchmarks Digital Pre-Distortion (DPD) algorithms across two categories:

- **AI-based (neural network)**: GRU and TRes-DeltaGRU, trained via stochastic gradient descent (SGD)
- **Traditional (polynomial)**: Memory Polynomial (MP) and Generalized Memory Polynomial (GMP), identified via QR decomposition (closed-form least squares)

All models are constrained to approximately 500 real-valued parameters for a fair comparison. Evaluation uses Adjacent Channel Leakage Ratio (ACLR) and Error Vector Magnitude (EVM), where more negative values indicate better performance.

## 2. Test Signals

| Property | APA_200MHz | DPA_160MHz |
|----------|-----------|-----------|
| Standard | LTE TM3.1a | OFDM |
| Configuration | 5-carrier x 40 MHz | 4-carrier x 40 MHz |
| Total Bandwidth | 200 MHz | 160 MHz |
| Modulation | 256-QAM | 1024-QAM |
| PAPR | 10.01 dB (at CCDF of 0.001%) | 10.38 dB |
| Sampling Rate | 983.04 MHz | 640 MHz |
| Dataset Size | 98,304 samples | 491,520 samples |
| Segment Size (nperseg) | 19,662 | 16,384 |
| Sub-channels | 5 | 4 |
| Dataset Split | 60% / 20% / 20% (train / val / test) | 60% / 20% / 20% (train / val / test) |

## 3. Power Amplifier Devices Under Test (DUT)

| Property | APA_200MHz | DPA_160MHz |
|----------|-----------|-----------|
| PA Type | GaN Doherty PA | 40 nm CMOS Digital PA (DPA) |
| Part / Technology | Ampleon AR211132 (evaluation board) | 40 nm CMOS |
| Carrier Frequency | 3.5 GHz | 2.4 GHz |
| Average Output Power | 41.2 dBm | 13.75 dBm |
| P1dB Compression Point | 46.5 dBm | - |
| P3dB Compression Point | 50 dBm | - |
| Flat Gain Bandwidth | 200 MHz | 160 MHz |
| Reference | Wu et al., "OpenDPDv2," arXiv:2507.06849 | Wu et al., "MP-DPD," IEEE IMS 2024, arXiv:2404.15364 |

The APA_200MHz dataset uses a high-power 3.5 GHz GaN Doherty PA designed for 5G base station applications. The DPA_160MHz dataset uses a low-power 2.4 GHz CMOS digital PA targeting Wi-Fi/IoT transmitter applications. Both PAs exhibit nonlinear distortion with memory effects, requiring DPD to meet spectral emission standards.

## 4. PA Surrogate Model

All DPD models are evaluated through the same trained GRU PA surrogate model per dataset, ensuring a fair comparison. The PA model is frozen during DPD training (indirect learning architecture).

| Dataset | PA Backbone | PA Hidden Size | PA Parameters | PA Training | Best PA NMSE |
|---------|-------------|----------------|---------------|-------------|-------------|
| APA_200MHz | GRU | 23 | 1,911 | 100 epochs, AdamW, lr=5e-4 | -43.52 dB |
| DPA_160MHz | GRU | 24 | 2,066 | 100 epochs, AdamW, lr=5e-4 | -38.43 dB |

## 5. DPD Algorithms

### 5.1 AI-Based Models (trained via SGD)

**GRU (Gated Recurrent Unit)**
- Standard PyTorch GRU cell followed by a linear output layer
- Input: I/Q samples (2D); Output: I/Q samples (2D)
- Configuration: hidden_size=11, num_layers=1
- Parameters: 519 real-valued

**TRes-DeltaGRU (Temporal Residual Delta GRU)**
- Delta GRU cell with TCN-based temporal residual skip connection (Wu et al., OpenDPDv2, arXiv:2507.06849)
- The TCN residual path decouples output dynamics from hidden-state sparsity, enabling aggressive temporal sparsity without linearization loss
- Configuration: hidden_size=10, num_layers=1, Conv1d(2,3,k=3,d=16) + Conv1d(3,2,k=1)
- Parameters: 524 real-valued

### 5.2 Traditional Models (identified via QR)

**MP (Memory Polynomial)**
- Diagonal subset of the Volterra series: basis functions `x(n-q) * |x(n-q)|^k`
- Configuration: K=5 nonlinearity orders, Q=50 memory depth
- Parameters: 250 complex coefficients = 500 real-valued
- Identification: one-shot least-squares via `numpy.linalg.lstsq` (SVD-based)

**GMP (Generalized Memory Polynomial)**
- Extended Memory Polynomial with lagging and leading cross-terms (Morgan et al., IEEE TSP, 2006)
- Aligned terms: `x(n-q) * |x(n-q)|^k` (Ka=5, La=15 -> 75 terms)
- Lagging cross-terms: `x(n-q) * |x(n-q-l)|^k` (Kb=4, Lb=15, Mb=2 -> 120 terms)
- Leading cross-terms: `x(n-q) * |x(n-q+l)|^k` (Kc=4, Lc=15, Mc=1 -> 60 terms)
- Parameters: 255 complex coefficients = 510 real-valued
- Identification: one-shot least-squares via `numpy.linalg.lstsq` (SVD-based)

Both traditional models use the Indirect Learning Architecture (ILA): build basis from normalized PA output, solve for postdistorter coefficients, then copy to predistorter.

## 6. Training Configuration

### SGD-Based Models (GRU, TRes-DeltaGRU)

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Learning Rate | 5e-4 |
| Epochs | 300 |
| Batch Size | 256 |
| Loss Function | MSE (L2) |
| LR Schedule | None |
| Gradient Clipping | 200 |
| Best Model Selection | Validation ACLR_AVG |
| Seed | 0 |

### QR-Based Models (MP, GMP)

| Parameter | Value |
|-----------|-------|
| Solver | `numpy.linalg.lstsq` (SVD/QR internally) |
| Training Iterations | 1 (closed-form) |
| Regularization | None |
| Training Data | Full training set |

## 7. Results

### 7.1 APA_200MHz (3.5 GHz GaN Doherty PA, 5-carrier LTE, 200 MHz BW, 256-QAM)

| Rank | Model | Parameters | Method | ACLR_AVG (dB) | EVM (dB) | NMSE (dB) |
|------|-------|-----------|--------|---------------|----------|-----------|
| 1 | **TRes-DeltaGRU** | 524 | SGD 300ep | **-53.35** | **-49.08** | -48.36 |
| 2 | GRU | 519 | SGD 300ep | -52.61 | -47.50 | -46.84 |
| 3 | MP | 500 | QR | -41.01 | -32.68 | -31.77 |
| 4 | GMP | 510 | QR | -38.80 | -38.53 | -30.57 |

### 7.2 DPA_160MHz (2.4 GHz CMOS Digital PA, 4-carrier, 160 MHz BW, 1024-QAM)

| Rank | Model | Parameters | Method | ACLR_AVG (dB) | EVM (dB) | NMSE (dB) |
|------|-------|-----------|--------|---------------|----------|-----------|
| 1 | **TRes-DeltaGRU** | 524 | SGD 300ep | **-56.81** | **-54.00** | -49.74 |
| 2 | GMP | 510 | QR | -54.02 | -51.08 | -45.92 |
| 3 | GRU | 519 | SGD 300ep | -51.93 | -49.97 | -44.83 |
| 4 | MP | 500 | QR | -51.29 | -50.26 | -45.29 |

## 8. Analysis

### AI vs. Traditional

- On **APA_200MHz** (wider bandwidth, more carriers), the AI models dominate: TRes-DeltaGRU leads the best traditional method (MP) by **12.3 dB in ACLR** and **10.6 dB in EVM**. The recurrent hidden state captures complex long-range memory effects that polynomial basis functions cannot.
- On **DPA_160MHz**, the gap narrows but AI still leads: TRes-DeltaGRU beats the best traditional method (GMP/QR) by **2.8 dB in ACLR** and **2.9 dB in EVM**. The traditional GMP/QR is competitive here, outperforming vanilla GRU.

### QR vs. SGD for Polynomial Models

The identification method matters enormously for polynomial models. Comparing GMP trained with QR vs. SGD (100 epochs, from earlier experiments):

| Dataset | GMP (QR) ACLR | GMP (SGD 100ep) ACLR | Improvement |
|---------|--------------|---------------------|------------|
| APA_200MHz | -38.80 dB | -29.67 dB | +9.1 dB |
| DPA_160MHz | -54.02 dB | -44.46 dB | +9.6 dB |

QR finds the global optimum in one shot; SGD with 100 epochs of Adam cannot reach it for this convex problem. This demonstrates that when the model is linear-in-parameters, closed-form solvers are strictly superior to iterative gradient descent.

### TRes-DeltaGRU vs. GRU

TRes-DeltaGRU consistently outperforms vanilla GRU across both datasets:

| Dataset | TRes-DeltaGRU ACLR | GRU ACLR | Gap |
|---------|-------------------|----------|-----|
| APA_200MHz | -53.35 dB | -52.61 dB | +0.7 dB |
| DPA_160MHz | -56.81 dB | -51.93 dB | +4.9 dB |

The TCN skip connection provides a direct residual path from input to output, decoupling the output from hidden-state dynamics. This is especially beneficial on DPA_160MHz where the improvement is nearly 5 dB.

## 9. Reproducing These Results

### SGD Models

```bash
# Train PA model (once per dataset)
python main.py --step train_pa --dataset_name APA_200MHz --PA_backbone gru --PA_hidden_size 23 --n_epochs 100
python main.py --step train_pa --dataset_name DPA_160MHz --PA_backbone gru --PA_hidden_size 24 --n_epochs 100

# Train GRU DPD
python main.py --step train_dpd --dataset_name APA_200MHz --PA_backbone gru --PA_hidden_size 23 --DPD_backbone gru --DPD_hidden_size 11 --n_epochs 300
python main.py --step train_dpd --dataset_name DPA_160MHz --PA_backbone gru --PA_hidden_size 24 --DPD_backbone gru --DPD_hidden_size 11 --n_epochs 300

# Train TRes-DeltaGRU DPD
python main.py --step train_dpd --dataset_name APA_200MHz --PA_backbone gru --PA_hidden_size 23 --DPD_backbone tres_deltagru --DPD_hidden_size 10 --n_epochs 300
python main.py --step train_dpd --dataset_name DPA_160MHz --PA_backbone gru --PA_hidden_size 24 --DPD_backbone tres_deltagru --DPD_hidden_size 10 --n_epochs 300
```

### QR Models

```bash
# MP (K=5, Q=50 -> 500 real params)
python benchmark_volterra_qr.py --dataset_name APA_200MHz --pa_hidden_size 23 --model mp --K 5 --Q 50
python benchmark_volterra_qr.py --dataset_name DPA_160MHz --pa_hidden_size 24 --model mp --K 5 --Q 50

# GMP (255 complex = 510 real params)
python benchmark_volterra_qr.py --dataset_name APA_200MHz --pa_hidden_size 23 --model gmp
python benchmark_volterra_qr.py --dataset_name DPA_160MHz --pa_hidden_size 24 --model gmp
```
