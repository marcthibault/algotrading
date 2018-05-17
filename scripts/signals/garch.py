from .signal import Signal
import numpy as np

# import matplotlib.pyplot as plt
from scipy.linalg import sqrtm
# from numba import jit
from arch import arch_model
from tqdm import tqdm
import itertools
from scipy.stats import norm

from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor


class CCC_GARCH(Signal):
    def __init__(self, data, n_past, n_fit):
        Signal.__init__(self, data)
        self.mu = None
        self.w = None
        self.alpha = None
        self.beta = None
        self.R = None

        ## Used for smoothing
        self.initial_sigma = None
        self.good_index = []

        self.n_fit = n_fit
        self.n_past = n_past

    def set_parameters(self, mu, w, alpha, beta, R):
        self.mu = mu.reshape((mu.shape[0], 1))
        self.w = w.reshape((mu.shape[0], 1))
        self.alpha = alpha.reshape((mu.shape[0], 1))
        self.beta = beta.reshape((mu.shape[0], 1))
        self.R = R

    def generateGARCH(self, N):
        if self.mu is None:
            raise Exception("You must set the value of the parameters before, with function set_parameters")
        else:
            n = self.mu.shape[0]

        r = np.zeros((n, N + 1))
        e = np.random.randn(n, N + 1)
        eps = np.zeros((n, N + 1))
        sigma = np.zeros((n, N + 1))
        r[:, 0:1] = self.mu
        sigma[:, 0:1] = self.w / (1 - self.alpha - self.beta)

        for i in range(1, N + 1):
            sigma[:, i:i + 1] = self.w + self.alpha * eps[:, i - 1:i] ** 2 + self.beta * sigma[:, i - 1:i]
            sig = np.diag(np.sqrt(sigma[:, i]))
            H = np.matmul(np.matmul(sig, self.R), sig)
            eps[:, i:i + 1] = np.matmul(sqrtm(H), e[:, i:i + 1])
            r[:, i:i + 1] = self.mu + eps[:, i:i + 1]

        return r

    def _compute(self, start_date=None, end_date=None, trailing_sigma=1, refit=1000):
        '''
        trailing_sigma : number of days to use to smoothe the last variance
        refit : number of days between two refits
        '''
        if start_date is None:
            start_date = self.data.index.get_level_values(0)[0]
        if end_date is None:
            end_date = self.data.index.get_level_values(0)[-1]

        self.data = self.data.loc[start_date:end_date]
        self.data = self.data.ffill().bfill()

        idx = self.data.index.get_level_values(0).unique()
        companies = self.data[self.data.loc[:, "filter"] == 1].index.get_level_values(1).unique().values
        self.data["signal"] = np.nan
        for k, date in enumerate(tqdm(idx[self.n_past:-self.n_fit])):
            print(date)
            past_date = idx[k]
            future_date = idx[k + self.n_fit + self.n_past]
            df1 = self.data.loc[past_date:future_date]
            temp_filter = (df1.loc[future_date, "filter"] == 1)
            index_temp = temp_filter.loc[temp_filter.values].index.values
            removed = set(companies) - set(index_temp)
            for tick in removed:
                print('WARNING : {} was removed from the trading'.format(tick))

            list_removed = []
            list_companies_removed = []
            for tick in removed:
                tick_idx = np.where(companies == tick)[0][0]
                list_companies_removed.append(tick_idx)
                list_removed.append(self.good_index.index(tick_idx))

            for tick in removed:
                self.data['filter'].loc[future_date:, tick] = 0.0

            if list_companies_removed != []:
                temp_companies_correct = [i for i in range(len(companies)) if i not in list_companies_removed]
                companies = companies[temp_companies_correct]
                temp_index_correct = [i for i in range(len(self.good_index)) if i not in list_removed]

            for i in list_removed[::-1]:
                del [self.good_index[i]]
                for b in range(i, len(self.good_index)):
                    self.good_index[b] -= 1

            if list_removed != []:
                self.R = self.R[temp_index_correct, :][:, temp_index_correct]
                self.mu = self.mu[temp_index_correct]
                self.alpha = self.alpha[temp_index_correct]
                self.beta = self.beta[temp_index_correct]
                self.w = self.w[temp_index_correct]
                self.initial_sigma = self.initial_sigma[temp_index_correct]

            temp_prices = df1["adj_close"].unstack().loc[:, index_temp]

            p = temp_prices.values.T
            r = np.log(p[:, 1:]) - np.log(p[:, :-1])
            ## We want to refit every "refit" date. no_refit is false when we want to refit
            signal = self.computeOneSignal(r, no_refit=(k % refit))

            self.data.loc[(future_date, list(index_temp)), "signal"] = signal

        self._computed = True
        self.signal = self.data.loc[:, "signal"]

    def _compute_dict(self, start_date=None, end_date=None, trailing_sigma=1, refit=1000):
        '''
        trailing_sigma : number of days to use to smoothe the last variance
        refit : number of days between two refits
        '''
        if start_date is None:
            start_date = self.data.index.get_level_values(0)[0]
        if end_date is None:
            end_date = self.data.index.get_level_values(0)[-1]

        self.data = self.data.loc[start_date:end_date]
        self.data = self.data.ffill().bfill()

        calendar = self.data.index.get_level_values(0).unique()
        previous_companies = self.data[self.data.loc[:, "filter"] == 1].index.get_level_values(1).unique().values
        self.data["signal"] = np.nan

        fitter = {ticker: GarchFitter() for ticker in all_companies}
        for k, fit_date in enumerate(tqdm(calendar[self.n_past:-self.n_fit])):
            start_fit_date = calendar[k]
            current_date = calendar[k + self.n_fit + self.n_past]
            print(current_date)

            df1 = self.data.loc[start_fit_date:current_date]
            temp_filter = (df1.loc[current_date, "filter"] == 1)
            current_companies = set(temp_filter.loc[temp_filter.values].index.values)

            temp_prices = df1["adj_close"].unstack().loc[:, current_companies]
            p = temp_prices.values.T
            r = np.log(p[:, 1:]) - np.log(p[:, :-1])
            r_past = r[:, :self.n_past]
            r_future = r[:, self.n_past:]

            # Every refit days, we refit all our models: single series and correlation matrix.
            if (k % refit) == 0:
                for ticker_index, ticker in enumerate(current_companies):
                    fitter[ticker].fit(r_past[ticker_index])

                residuals = np.array([fitter[ticker].residuals for ticker in current_companies])
                self.R = np.corrcoef(residuals)
                self.mu = np.array([fitter[ticker].mu for ticker in current_companies])  # TODO Nx1

            # Case where we lost or won a company
            if previous_companies != current_companies:
                pass
                # TODO

            # self.R = self.R[temp_index_correct, :][:, temp_index_correct]
            self.initial_sigma = np.array(
                [fitter[ticker]._getLastVariance(r_past[index_ticker]) for index_ticker, ticker in
                 enumerate(current_companies)])

            ## We want to refit every "refit" date. no_refit is false when we want to refit
            signal = self.computeOneSignal(r_future)
            self.data.loc[(current_date, list(current_companies)), "signal"] = signal

            previous_companies = current_companies

        self._computed = True
        self.signal = self.data.loc[:, "signal"]

    def computeOneSignal(self, r_future):
        """
        trailing_sigma : number of days to use to smoothe the last variance
        refit : if true, refit the GARCH model
        """
        n_stocks = r_future.shape[0]
        moved = np.sum(r_future, axis=1)
        signal = np.zeros(n_stocks)

        for ticker_index in range(n_stocks):
            try:
                forwarddistribution_mu, forwarddistribution_sigma = self._computeProbabilityDistributionGaussian(
                    ticker_index,
                    r_future)
                signal[ticker_index] = self.computeQuantile(forwarddistribution_mu, forwarddistribution_sigma, moved[ticker_index])
            except:
                print("Error in _computeProbabilityDistributionGaussian({})".format(ticker_index))
                signal[ticker_index] = np.nan

        return signal

    def _estimate(self, r):
        mu = np.zeros((r.shape[0], 1))
        w = np.zeros((r.shape[0], 1))
        alpha = np.zeros((r.shape[0], 1))
        beta = np.zeros((r.shape[0], 1))
        count = 0
        for i in range(r.shape[0]):
            am = arch_model(r[i], {'disp': False})
            if self.mu is not None and i in self.good_index:
                res = am.fit(disp="off", show_warning=False, starting_values=np.array(
                    [self.mu[count, 0], self.w[count, 0], self.alpha[count, 0], self.beta[count, 0]]))
                count += 1
            else:
                res = am.fit(disp="off", show_warning=False)
            mu[i, 0], w[i, 0], alpha[i, 0], beta[i, 0] = res.params.values

        mu = np.nan_to_num(mu)
        w = np.nan_to_num(w)
        alpha = np.nan_to_num(alpha)
        beta = np.nan_to_num(beta)

        eps, sigma = CCC_GARCH._recoverVariables(r, mu, w, alpha, beta)

        residuals = np.zeros(sigma.shape)
        residuals[sigma != 0.] = eps[sigma != 0.] / np.sqrt(sigma[sigma != 0.])

        R = np.corrcoef(residuals)
        self.set_parameters(mu, w, alpha, beta, R)

    @staticmethod
    # @jit(nopython=True)
    def _recoverVariables(r, mu, w, alpha, beta, initial_sigma=None):
        eps = r - mu
        sigma = np.zeros_like(r)
        alphabeta = alpha + beta
        alphabeta = np.minimum(alphabeta, 1 - 1e-4)
        if initial_sigma is None:
            initial_sigma = w / (1 - alphabeta)

        initial_sigma[np.isinf(initial_sigma)[:, 0]] = w[np.isinf(initial_sigma)[:, 0]]
        initial_sigma = np.maximum(np.minimum(initial_sigma, 1.), 0.)

        sigma[:, 0:1] = initial_sigma
        for i in range(1, r.shape[1]):
            sigma[:, i:i + 1] = w + alpha * eps[:, i - 1:i] ** 2 + beta * sigma[:, i - 1:i]

        return eps, sigma

    def _computeProbabilityDistributionGaussian(self, index, r_future):
        """
        Compute probability distribution of variable labeled index
        knowing the evolution of the others.
        index : index of the dependant variable in the array
        r : past returns on which the model was fitted
        rr : next returns for every variables in the model
        N : number of discretization points
        max_workers : number of workers to use in the multithreading pool
        """

        n_step = r_future.shape[1]
        m = r_future.shape[0]
        R_other = np.delete(r_future, index, axis=0)
        H = np.dot(np.diag(np.sqrt(self.initial_sigma[:, 0])),
                   np.dot(self.R, np.diag(np.sqrt(self.initial_sigma[:, 0]))))
        idx = np.concatenate((np.linspace(0, index - 1, index), np.linspace(index + 1, m - 1, m - index - 1))).astype(
            np.int32)
        H_other = H[idx][:, idx]
        H_inv = np.linalg.inv(H_other)
        rho = H[index:index + 1, idx]
        sigma1 = np.sqrt(self.initial_sigma[index, 0] - np.dot(rho, np.dot(H_inv, rho.T))[0, 0])
        mumu = self.mu[idx]

        mu1 = self.mu[index, 0] + np.dot(rho, np.dot(H_inv, (np.mean(R_other, axis=1) - mumu)))[0, 0]
        return n_step * mu1, np.sqrt(n_step) * sigma1

    def _getLastVariance(self, r):
        eps, sigma = CCC_GARCH._recoverVariables(r, self.mu, self.w, self.alpha, self.beta)
        return sigma[:, -1:]

    @staticmethod
    def computeQuantile(mu, sigma, value):
        return norm.cdf((mu - value) / sigma)
