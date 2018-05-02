from .signal import Signal
import numpy as np
import pandas as pd

'''
Mean reversion signal, being 1 - price/average.
'''


class MeanReversion(Signal):
    def __init__(self, data, ewma_length):
        Signal.__init__(self, data.copy())
        self.decay_rate = 1 - np.exp(np.log(.5) / (ewma_length * 2))

    def _compute(self):
        self.data['ewma'] = self.data.groupby(
            level=1).close.apply(lambda x: x.ewm(alpha=self.decay_rate).mean())

        self.signal = 1 - self.data.close / \
            self.data.ewma.groupby(level=1).shift()
