from .signal import Signal
import numpy as np
import pandas as pd

'''
Beta residuals of stock returns.
'''


class FactorResidual(Signal):
    def __init__(self, data, past, gamma = 10, rolling_window=252):
        Signal.__init__(self, data)
        self.past = past
        self.rw = rolling_window

    def _compute(self):
        if self.past:
            self.data['perf'] = self.data.past_perf_1d

        else:
            self.data['perf'] = self.data.future_perf_1d

        rw = self.rw

        def fit(x_mat, gamma, nb_f):
            T = x_mat.shape[1]
            N = x_mat.shape[0]
            x_mat_bar = np.mean(x_mat, axis=0)
            matrix_temp = 1/T*np.dot(x_mat.T, x_mat) + gamma*np.outer(x_mat_bar, x_mat_bar)
            values, vectors = np.linalg.eigh(matrix_temp)
            print(values[np.argsort(-values)][:10])

            lmbd = vectors[:, np.argsort(-values)[:nb_f]]
            print(lmbd)
            f_mat = np.sqrt(N) * np.dot(x_mat, lmbd)
            res = x_mat - np.dot(f_mat, lmbd.T)
            return res

        results = pd.DataFrame()
        for t in self.data.index.get_level_values(0).unique()[250:]:
            temp = self.data.loc[t-pd.Timedelta('365 days'):t]
            stocks = temp.loc[temp['filter'] == 1].loc[t].index.get_level_values(0).unique()
            temp = temp.loc(axis=0)[:, stocks].perf.unstack().fillna(0)
            print(temp.values)
            print(fit(temp.values, -1, 3))
            break

        self.signal = self.data.perf.rename('signal')
