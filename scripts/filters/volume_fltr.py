from .filter import Filter
import numpy as np


class VolumeFilter(Filter):
    def __init__(self, data, nb_stocks, last_days=10):
        Filter.__init__(self, data)
        self.nb_stocks = nb_stocks
        self.last_days = last_days

    def _compute(self):
        past_volume = self.data.groupby(level=1).adj_volume.rolling(self.last_days, min_periods=1).mean().fillna(0)
        past_volume.index = past_volume.index.droplevel(0)
        self.data['past_volume'] = past_volume
        self.filter = self.data.past_volume.groupby(level=0).apply(lambda x: np.argsort(-x) < self.nb_stocks)
