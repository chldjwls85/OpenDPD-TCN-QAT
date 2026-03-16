"""10-carrier LTE 20MHz, DPA device, 200 MHz total bandwidth.

IFFT-frame demodulation (nperseg=2560, no cyclic prefix).
"""

from datasets.demodulator import IFFTFrameDemodulator


class Demodulator(IFFTFrameDemodulator):
    pass
