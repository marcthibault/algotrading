import numpy as np
from scipy.stats import norm
from tqdm import tqdm

from scripts.signals.garch_fitter import GarchFitter
from .signal import Signal


class CCC_GARCH(Signal):
    def __init__(self, data, n_past, n_fit):
        Signal.__init__(self, data)
        self.mu = None
        self.R = None

        ## Used for smoothing
        self.initial_sigma = None
        self.good_index = []

        self.n_fit = n_fit
        self.n_past = n_past

    def _compute(self, start_date=None, end_date=None, refit=1000):
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
        all_companies = set(self.data.index.get_level_values(1).unique().values)
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
                conditional_mu, conditional_sigma = self._computeProbabilityDistributionGaussian(ticker_index,
                                                                                                 r_future)
                signal[ticker_index] = self.computeQuantile(conditional_mu,
                                                            conditional_sigma,
                                                            moved[ticker_index])
            except Exception as e:
                print(e, "Error in _computeProbabilityDistributionGaussian({})".format(ticker_index))
                signal[ticker_index] = np.nan

        return signal

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

    @staticmethod
    def computeQuantile(mu, sigma, value):
        return norm.cdf((mu - value) / sigma)
