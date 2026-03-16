"""5-carrier LTE 20MHz (TM3.1a 256QAM), APA device, measurement B.

CP-synced OFDM demodulation.  Same signal type as APA_200MHz.
"""

from datasets.demodulator import OFDMCPDemodulator


class Demodulator(OFDMCPDemodulator):
    pass
