from .filter import Filter
import numpy as np
import pandas as pd


class FixedVolumeFilter(Filter):
    def __init__(self, data, nb_stocks, evaluation_day, n_past=100):
        Filter.__init__(self, data)
        self.nb_stocks = nb_stocks
        self.evaluation_day = evaluation_day
        self.n_past = n_past

    def _compute(self):
        volumefilter = np.argsort(-self.data.adj_volume.loc[self.evaluation_day, :].values * \
                                  self.data.adj_close.loc[self.evaluation_day, :].values)
        volumefilter = volumefilter[:self.nb_stocks]
        kept_tickers = self.data.index.get_level_values(1).unique()[volumefilter]

        ## Case dates don't appear in the groupby, we have to unstack 
        ## To keep the original calendar, we have to do the rolling version as well
        past_data_0 = self.data.adj_close.groupby(level=1).rolling(self.n_past, min_periods=1).count()
        past_data_0.index = past_data_0.index.droplevel(0)
        past_data_0 = past_data_0.sort_index(0)
        past_data = self.data.adj_close.unstack().rolling(self.n_past, min_periods=1).count().stack().sort_index(0)

        past_data = past_data_0.reset_index().merge(past_data.reset_index(), on=['date', 'ticker'], how='inner').set_index(['date', 'ticker'])[0]

        self.data["filter"] = 0
        self.data["filter"].loc[:, kept_tickers] = 1
        
        self.data["filter"][past_data < 0.98 * self.n_past] = 0
        self.data["filter"][self.data.past_perf_1d.isnull()] = 0
        self.data["filter"][self.data.past_perf_1d == 0.0] = 0
        self.data["filter"][self.data["adj_volume"]==0.0] = 0

        self.filter = self.data['filter']
