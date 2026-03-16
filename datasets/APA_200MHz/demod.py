"""5-carrier LTE 20MHz (TM3.1a 256QAM), APA device.

CP-synced OFDM demodulation.  Generated at 491.52 MHz (SCS=15 kHz),
transmitted/captured at 983.04 MHz (effective SCS=30 kHz, BW=200 MHz).
"""

from datasets.demodulator import OFDMCPDemodulator


class Demodulator(OFDMCPDemodulator):
    pass
