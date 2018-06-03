from .signal import Signal
import numpy as np
import pandas as pd

'''
Beta residuals of stock returns.
'''


class BetaResidual(Signal):
    def __init__(self, data, rolling_window=252):
        Signal.__init__(self, data)
        self.rw = rolling_window

    def _compute(self):
        # Compute the market perf
        self.data['perf'] = self.data.past_perf_1d
        rw = self.rw
        # idx_perf = self.data.loc[self.data['filter'] == 1].groupby(
        #     'date').past_perf_1d.mean().rename('idx_perf')
        #
        # idx_future_perf = self.data.loc[self.data['filter'] == 1].groupby(
        #     'date').future_perf_1d.mean().rename('idx_future_perf')

        # Variance by stock to compute volatility factor
        temp_var_stock = self.data.loc[self.data.future_filter == 1].groupby(level=1, group_keys=False)[
            'perf'].rolling(rw).var()
        temp_var_stock = temp_var_stock.reset_index(
            level=[0], drop=True).sort_index().rename('stock_var')
        temp_var_stock = temp_var_stock.loc[self.data['filter'] == 1].dropna()

        vol_perc_inf = temp_var_stock.dropna().groupby(level=0).apply(lambda x: x < np.percentile(x, 10))
        vol_perc_sup = temp_var_stock.dropna().groupby(level=0).apply(lambda x: x > np.percentile(x, 90))
        vol_factor = (self.data.loc[vol_perc_sup.loc[vol_perc_sup].index].perf.groupby(level=0).mean() -
                      self.data.loc[vol_perc_inf.loc[vol_perc_inf].index].perf.groupby(level=0).mean()).rename('idx_perf')

        self.data = self.data.join(vol_factor)
        self.data['idx_future_perf'] = self.data.idx_perf.groupby(level=1).shift(-1)

        # Data we need to compute var
        self.data['future_filter'] = self.data.groupby(level=1).rolling(rw)['filter'].apply(
            lambda x: np.sum(x) > 0).reset_index(level=0, drop=True).fillna(False)
        self.data.loc[:, 'future_filter'] = self.data.future_filter.groupby(level=1).shift(-rw)

        # Variance of the index
        temp_var = self.data.loc[self.data.future_filter == 1].groupby(level=1, group_keys=False)[
            'idx_perf'].rolling(rw).var()
        temp_var = temp_var.reset_index(level=[0], drop=True).sort_index().rename('var')

        # covariance stock / index
        temp_cov = self.data.loc[self.data.future_filter == 1].groupby(
            level=1, group_keys=False)[['perf', 'idx_perf']].rolling(rw).cov()
        temp_cov = temp_cov.reset_index(level=[0], drop=True).sort_index().loc(
            axis=0)[:, :, 'perf'].reset_index(level=2, drop=True).idx_perf.rename('cov')

        # Computing beta
        self.data['beta'] = temp_cov / temp_var
        self.signal = self.data.future_perf_1d - self.data.beta * self.data.idx_future_perf
