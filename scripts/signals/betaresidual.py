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
            data['perf'] = data.past_perf_1d
            data['idx_perf'] = data.loc[data['filter'] == 1].groupby(
                'date').past_perf_1d.transform(np.mean)
        else:
            data['perf'] = data.future_perf_1d
            data['idx_perf'] = data.loc[data['filter'] == 1].groupby(
                'date').future_perf_1d.transform(np.mean)

        rw = self.rw

        temp_cov = data.groupby(level=1, group_keys=False)[['perf', 'idx_perf']].rolling(rw).cov()
        temp_cov = temp_cov.reset_index(level=[0], drop=True).sort_index().loc(
            axis=0)[:, :, 'perf'].reset_index(level=2, drop=True).idx_perf.rename('cov')

        temp_var = data.groupby(level=1, group_keys=False)['idx_perf'].rolling(rw).var()
        temp_var = temp_var.reset_index(level=[0], drop=True).sort_index().rename('var')

        data['beta'] = temp_cov / temp_beta
        self.signal = data.perf - data.beta * data.idx_perf
