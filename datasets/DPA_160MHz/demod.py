"""4-carrier 40MHz signal, DPA device, 160 MHz total bandwidth.

IFFT-frame demodulation (nperseg=16384, no cyclic prefix).
"""

from datasets.demodulator import IFFTFrameDemodulator


class Demodulator(IFFTFrameDemodulator):
    pass
