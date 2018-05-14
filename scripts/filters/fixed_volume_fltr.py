from .filter import Filter
import numpy as np
import pandas as pd


class FixedVolumeFilter(Filter):
    def __init__(self, data, nb_stocks, evaluation_day):
        Filter.__init__(self, data)
        self.nb_stocks = nb_stocks
        self.evaluation_day = evaluation_day

    def _compute(self):
        volumefilter = np.argsort(-self.data.adj_volume.loc[self.evaluation_day, :].values * \
                                  self.data.adj_close.loc[self.evaluation_day, :].values)
        volumefilter = volumefilter[:self.nb_stocks]
        kept_tickers = self.data.index.get_level_values(1).unique()[volumefilter]

        self.data["filter"] = 0
        self.data["filter"].loc[:, kept_tickers] = 1
        
        self.filter = self.data['filter']
