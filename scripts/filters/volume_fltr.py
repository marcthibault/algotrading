from .filter import Filter
import numpy as np
import pandas as pd


class VolumeFilter(Filter):
    def __init__(self, data, nb_stocks, last_days=10):
        Filter.__init__(self, data)
        self.nb_stocks = nb_stocks
        self.last_days = last_days

    def _compute(self):
        vol_dolls = self.data.adj_volume*self.data.adj_close
        past_volume = vol_dolls.groupby(level=1).rolling(self.last_days, min_periods=1).mean().fillna(0)
        past_volume.index = past_volume.index.droplevel(0)
        self.data['past_volume'] = vol_dolls
        def f(x):
            y = np.argsort(-x)
            res = np.zeros(len(x))
            res[y[:self.nb_stocks]] = 1

            return pd.Series(res, index=x.index)

        self.filter = self.data.past_volume.groupby(level=0).apply(f)
