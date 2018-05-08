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

    def _compute(self, start_date=None, end_date=None):
        if start_date is None:
            start_date = self.data.index.get_level_values(0)[0]
        if end_date is None:
            end_date = self.data.index.get_level_values(0)[-1]

        self.data = self.data.loc[start_date:end_date]
        self.data = self.data.ffill().bfill()

        idx = self.data.index.get_level_values(0).unique()
        self.data["signal"] = 0.0
        for k, date in enumerate(tqdm(idx[self.n_past:-self.n_fit])):
            print(date)
            past_date = idx[k]
            future_date = idx[k + self.n_fit + self.n_past]
            df1 = self.data.loc[past_date:future_date]
            temp_filter = (df1.loc[future_date, "filter"] == 1)
            index_temp = temp_filter.loc[temp_filter.values].index.values
            temp_prices = df1["adj_close"].unstack().loc[:, index_temp]

            p = temp_prices.values.T
            r = np.log(p[:, 1:]) - np.log(p[:, :-1])
            signal = self.computeOneSignal(r, self.n_past, self.n_past, self.n_fit)

            self.data.loc[(future_date, list(index_temp)), "signal"] = signal

        self._computed = True
        self.signal = self.data.loc[:, "signal"]

    def computeOneSignal(self, r, current_index, n_fit, n_predict, N=10000):
        if current_index < n_fit:
            n_fit = current_index

        r_past = r[:, (current_index - n_fit):current_index]
        r_future = r[:, current_index:(current_index + n_predict)]
        n = r_past.shape[0]
        moved = np.prod(1 + r_future, axis=1) - 1
        signal = np.zeros(n)
        self._estimate(r_past)
        for i in range(n):
            try:
                forwarddistribution_mu, forwarddistribution_sigma = self._computeProbabilityDistributionGaussian2(i,
                                                                                                                  r_past,
                                                                                                                  r_future,
                                                                                                                  N=N)
                signal[i] = self.computeQuantile2(forwarddistribution_mu, forwarddistribution_sigma, moved[i])
            except:
                signal[i] = 0.0
        return signal

    def _estimate(self, r):
        mu = np.zeros((r.shape[0], 1))
        w = np.zeros((r.shape[0], 1))
        alpha = np.zeros((r.shape[0], 1))
        beta = np.zeros((r.shape[0], 1))

        for i in range(r.shape[0]):
            am = arch_model(r[i])
            res = am.fit(disp="off")
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

    def _computeProbabilityDistributionGaussian(self, index, r, rr, N=1000):
        """
        Compute probability distribution of variable labeled index
        knowing the evolution of the others.
        index : index of the dependant variable in the array
        r : past returns on which the model was fitted
        rr : next returns for every variables in the model
        N : number of discretisation points
        max_workers : number of workers to use in the multithreading pool
        """
        initial_sigma = self._getLastVariance(r)

        n_step = rr.shape[1]
        m = r.shape[0]
        R_other = np.delete(rr, index, axis=0)
        H = np.dot(np.diag(np.sqrt(initial_sigma[:, 0])), np.dot(self.R, np.diag(np.sqrt(initial_sigma[:, 0]))))
        idx = np.concatenate((np.linspace(0, index - 1, index), np.linspace(index + 1, m - 1, m - index - 1))).astype(
            np.int32)
        H_other = H[idx][:, idx]
        H_inv = np.linalg.inv(H_other)
        rho = H[index:index + 1, idx]
        sigma1 = np.sqrt(initial_sigma[index, 0] - np.dot(rho, np.dot(H_inv, rho.T))[0, 0])
        mumu = self.mu[idx]
        distrib = np.zeros((N, n_step))
        for i in range(n_step):
            mu1 = self.mu[index, 0] + np.dot(rho,
                                             np.dot(H_inv, (R_other[:, i:i + 1] - mumu)))[0, 0]
            distrib[:, i] = np.random.normal(mu1, sigma1, N)

        distrib = np.product(1 + distrib, axis=1) - 1
        distrib = np.histogram(distrib, bins=int(np.sqrt(N)), density=True)
        log = np.zeros((2, len(distrib[0])))
        log[0] = distrib[1][1:]
        log[1] = distrib[0]
        log[1] = log[1] * (distrib[1][1:] - distrib[1][:-1])
        return log

    def _computeProbabilityDistributionGaussian2(self, index, r, rr, N=1000):
        """
        Compute probability distribution of variable labeled index
        knowing the evolution of the others.
        index : index of the dependant variable in the array
        r : past returns on which the model was fitted
        rr : next returns for every variables in the model
        N : number of discretization points
        max_workers : number of workers to use in the multithreading pool
        """
        initial_sigma = self._getLastVariance(r)

        n_step = rr.shape[1]
        m = r.shape[0]
        R_other = np.delete(rr, index, axis=0)
        H = np.dot(np.diag(np.sqrt(initial_sigma[:, 0])), np.dot(self.R, np.diag(np.sqrt(initial_sigma[:, 0]))))
        idx = np.concatenate((np.linspace(0, index - 1, index), np.linspace(index + 1, m - 1, m - index - 1))).astype(
            np.int32)
        H_other = H[idx][:, idx]
        H_inv = np.linalg.inv(H_other)
        rho = H[index:index + 1, idx]
        sigma1 = np.sqrt(initial_sigma[index, 0] - np.dot(rho, np.dot(H_inv, rho.T))[0, 0])
        mumu = self.mu[idx]

        mu1 = self.mu[index, 0] + np.dot(rho, np.dot(H_inv, (np.mean(R_other, axis=1) - mumu)))[0, 0]
        return n_step * mu1, np.sqrt(n_step) * sigma1

    def _getLastVariance(self, r):
        eps, sigma = CCC_GARCH._recoverVariables(r, self.mu, self.w, self.alpha, self.beta)
        return sigma[:, -1:]

    @staticmethod
    def computeQuantile(proba, value):
        p = 0
        i = 0
        while i < proba.shape[1] and proba[0, i] < value:
            p += proba[1, i]
            i += 1

        return p - 0.5

    @staticmethod
    def computeQuantile2(mu, sigma, value):
        return norm.cdf((mu - value) / sigma)
