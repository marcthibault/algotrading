from .filter import Filter
import numpy as np
import pandas as pd


class SliceFilter(Filter):
    def __init__(self, data, evaluation_day, begin, end):
        Filter.__init__(self, data)
        self.evaluation_day = evaluation_day
        self.begin = begin
        self.end = end

    def _compute(self):
        self.data['filter'] = 0
        volumefilter = np.argsort(-self.data.adj_volume.loc[self.evaluation_day, :].values * \
                                  self.data.adj_close.loc[self.evaluation_day, :].values)
        volumefilter = volumefilter[self.begin:self.end]
        kept_tickers = list(self.data.index.get_level_values(1).unique()[volumefilter])
        li_to_remove = ['X', 'MON', 'CHK', 'BAC', 'FCX']
        for tick in li_to_remove:
            if tick in kept_tickers:
                kept_tickers.remove(tick)
        self.data.loc[(slice(None), kept_tickers), 'filter'] = 1

        self.filter = self.data['filter']
