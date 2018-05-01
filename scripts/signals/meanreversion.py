from .Signal import Signal
import numpy as np

'''
Mean reversion signal, being 1 - price/average.
'''


class MeanReversion(Signal):
    def __init__(self, data, ewma_lenght):
        Signal.__init__(self, data)
        self.decay_rate = 1 - np.exp(np.log(.5) / (ewma_length * 2))

    def compute(self, ):
        self.data['ewma'] = self.data.groupby(
            level=1).close.ewm(alpha=self.decay_rate).mean()

        self.signal = 1 - self.data.close / \
            self.data.ewma.groupby(level=1).shift

        self._computed()
