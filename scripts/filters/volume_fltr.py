from .filter import Filter
import numpy as np


class VolumeFilter(Filter):
    def __init__(self, data, nb_stocks, last_days=10):
        Filter.__init__(self, data)
        self.nb_stocks = nb_stocks
        self.last_days = last_days

    def _compute(self):
        self.data['past_volume'] = self.data.groupby(level=1).volume.rolling(
            self.last_days, min_periods=1).mean().fillna(0)
        self.filter = self.data.past_volume.groupby(level=0).apply(
            lambda x: np.argsort(x) < self.nb_stocks)
