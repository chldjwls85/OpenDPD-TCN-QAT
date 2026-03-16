"""User-created custom PA dataset (single-channel).

IFFT-frame demodulation (nperseg=2560, no cyclic prefix).
"""

from datasets.demodulator import IFFTFrameDemodulator


class Demodulator(IFFTFrameDemodulator):
    pass
