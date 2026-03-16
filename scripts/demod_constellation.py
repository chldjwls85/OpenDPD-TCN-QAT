"""APA_200MHz — per-carrier with filter, plot each symbol separately."""
import sys
sys.path.insert(0, '.')

import numpy as np
import pandas as pd
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

with open('datasets/APA_200MHz/spec.json') as f:
    spec = json.load(f)

fs = spec['input_signal_fs']
n_sub_ch = spec['n_sub_ch']
bw_sub_ch = spec['bw_sub_ch']

dfs = []
for split in ['train', 'val', 'test']:
    df = pd.read_csv(f'datasets/APA_200MHz/{split}_input.csv')
    dfs.append(df)
data = pd.concat(dfs, ignore_index=True)
sig = data.iloc[:, 0].values + 1j * data.iloc[:, 1].values
MEM = len(sig)
NFFT = 32768
CP_OTHER = 2304
DC_BIN = NFFT // 2
N_HALF = 600
center = MEM // 2
bp_half = int(round(bw_sub_ch / 2 * MEM / fs))

f_shifts = [(i - (n_sub_ch - 1) / 2) * bw_sub_ch for i in range(n_sub_ch)]
ch_labels = [f'{f/1e6:+.0f} MHz' for f in f_shifts]


def process_channel(sig, fs, f_shift, bp_half):
    """Freq shift → bandpass → CP sync → per-symbol fine-tune."""
    MEM = len(sig)
    n = np.arange(MEM)
    ctr = MEM // 2

    if f_shift != 0:
        shifted = sig * np.exp(-1j * 2 * np.pi * f_shift * n / fs)
    else:
        shifted = sig.copy()

    fd = np.fft.fftshift(np.fft.fft(shifted))
    filt = np.zeros(MEM, dtype=complex)
    filt[ctr - bp_half:ctr + bp_half + 1] = fd[ctr - bp_half:ctr + bp_half + 1]
    clean = np.fft.ifft(np.fft.fftshift(filt))

    search = MEM - NFFT - CP_OTHER
    corr = np.zeros(search)
    energy = np.zeros(search)
    for nn in range(search):
        s1 = clean[nn:nn + CP_OTHER]
        s2 = clean[nn + NFFT:nn + NFFT + CP_OTHER]
        corr[nn] = np.abs(np.sum(s1 * np.conj(s2)))
        energy[nn] = np.sqrt(np.sum(np.abs(s1)**2) * np.sum(np.abs(s2)**2) + 1e-30)
    corr_norm = corr / energy
    peaks, _ = find_peaks(corr_norm, height=0.5, distance=NFFT // 2)

    results = []
    for p in peaks:
        best_kurt, best_off = 999, 0
        for off in range(-15, 16):
            fft_s = p + CP_OTHER + off
            if fft_s < 0 or fft_s + NFFT > MEM:
                continue
            sym = clean[fft_s:fft_s + NFFT]
            fd_sym = np.fft.fftshift(np.fft.fft(sym))
            sc = np.concatenate([fd_sym[DC_BIN - N_HALF:DC_BIN],
                                 fd_sym[DC_BIN + 1:DC_BIN + N_HALF + 1]])
            sc_n = sc / np.sqrt(np.mean(np.abs(sc)**2) + 1e-30)
            r, im = sc_n.real, sc_n.imag
            kurt = (np.mean(r**4) / (np.mean(r**2)**2 + 1e-30)
                    + np.mean(im**4) / (np.mean(im**2)**2 + 1e-30))
            if kurt < best_kurt:
                best_kurt, best_off = kurt, off

        fft_s = p + CP_OTHER + best_off
        sym = clean[fft_s:fft_s + NFFT]
        fd_sym = np.fft.fftshift(np.fft.fft(sym))
        sc = np.concatenate([fd_sym[DC_BIN - N_HALF:DC_BIN],
                             fd_sym[DC_BIN + 1:DC_BIN + N_HALF + 1]])
        sc_n = sc / np.sqrt(np.mean(np.abs(sc)**2) + 1e-30)
        results.append((sc_n, corr_norm[p], best_off, best_kurt))

    return results


# ---- Process all channels ----
all_results = {}
for f_shift, label in zip(f_shifts, ch_labels):
    results = process_channel(sig, fs, f_shift, bp_half)
    all_results[label] = results
    for i, (sc, corr, off, kurt) in enumerate(results):
        print(f"Ch {label} sym{i}: corr={corr:.4f} off={off:+d} kurt={kurt:.4f}")

# ---- Plot: each symbol separately (5 ch × max 2 sym) ----
max_syms = max(len(r) for r in all_results.values())
fig, axes = plt.subplots(max_syms, 5, figsize=(25, 5 * max_syms), squeeze=False)

best_sc = []  # collect only the best symbol per channel

for col, (label, results) in enumerate(all_results.items()):
    best_kurt_ch = 999
    best_sc_ch = None
    for row, (sc, corr, off, kurt) in enumerate(results):
        ax = axes[row, col]
        ax.scatter(sc.real, sc.imag, s=6, alpha=0.5,
                   color='royalblue', edgecolors='none')
        ax.set_title(f'{label} sym{row}\ncorr={corr:.3f} off={off:+d} kurt={kurt:.3f}')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('I')
        ax.set_ylabel('Q')

        if kurt < best_kurt_ch:
            best_kurt_ch = kurt
            best_sc_ch = sc

    if best_sc_ch is not None:
        best_sc.append(best_sc_ch)

fig.suptitle('APA_200MHz Per-Channel Per-Symbol Constellation', fontsize=14)
fig.tight_layout()
fig.savefig('plots/APA_200MHz_per_symbol.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print("\nSaved per-symbol to plots/APA_200MHz_per_symbol.png")

# ---- Combined: best symbol per channel ----
all_best = np.concatenate(best_sc)
fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(all_best.real, all_best.imag, s=5, alpha=0.4,
           color='royalblue', edgecolors='none')
ax.set_xlabel('In-phase (I)', fontsize=12)
ax.set_ylabel('Quadrature (Q)', fontsize=12)
ax.set_title('APA_200MHz Input Constellation (256QAM)\n'
             'Best symbol per carrier, 5-carrier LTE × 40MHz', fontsize=13)
ax.grid(True, alpha=0.3)
ax.set_aspect('equal')
fig.tight_layout()
fig.savefig('plots/APA_200MHz_constellation.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print("Saved combined (best/ch) to plots/APA_200MHz_constellation.png")
