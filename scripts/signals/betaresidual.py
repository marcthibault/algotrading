from .signal import Signal
import numpy as np
import pandas as pd

'''
Beta residuals of stock returns.
'''


class BetaResidual(Signal):
    def __init__(self, data, past, rolling_window=252):
        Signal.__init__(self, data)
        self.past = past
        self.rw = rolling_window

    def _compute(self):
        if self.past:
            self.data['perf'] = self.data.past_perf_1d
            idx_perf = self.data.loc[self.data['filter'] == 1].groupby(
                'date').past_perf_1d.mean().rename('idx_perf')
        else:
            self.data['perf'] = self.data.future_perf_1d
            idx_perf = self.data.loc[self.data['filter'] == 1].groupby(
                'date').future_perf_1d.mean().rename('idx_perf')
        self.data = self.data.join(idx_perf)
        rw = self.rw

        self.data['future_filter'] = self.data.groupby(level=1).rolling(rw)['filter'].apply(
            lambda x: np.sum(x) > 0).reset_index(level=0, drop=True).fillna(False)
        self.data.loc[:, 'future_filter'] = self.data.future_filter.groupby(level=1).shift(-rw)

        temp_cov = self.data.loc[self.data.future_filter == 1].groupby(
            level=1, group_keys=False)[['perf', 'idx_perf']].rolling(rw).cov()
        temp_cov = temp_cov.reset_index(level=[0], drop=True).sort_index().loc(
            axis=0)[:, :, 'perf'].reset_index(level=2, drop=True).idx_perf.rename('cov')

        temp_var = self.data.loc[self.data.future_filter == 1].groupby(level=1, group_keys=False)[
            'idx_perf'].rolling(rw).var()
        temp_var = temp_var.reset_index(level=[0], drop=True).sort_index().rename('var')

        self.data['beta'] = temp_cov / temp_var
        self.signal = self.data.perf - self.data.beta * self.data.idx_perf
