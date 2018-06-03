import numpy as np
from tqdm import tqdm, tqdm_notebook

from scripts.signals.garch_fitter import GarchFitter, GARCHConvergenceException
from .signal import Signal


class CrossSectionCorrelation(Signal):
    def __init__(self, data, n_past, n_fit, refit):
        Signal.__init__(self, data)
        self.mu = None

        self.R = None
        self.R_inverses = {}

        self.n_fit = n_fit
        self.n_past = n_past
        self.refit = refit

    def set_correlation_matrix(self, R):
        self.R = R
        n_stocks = R.shape[0]
        R_inv = np.linalg.pinv(R, rcond=1e-3)
        for i in range(n_stocks):
            idx = np.concatenate((np.arange(0, i), np.arange(i + 1, n_stocks))).astype(np.int32)
            U = np.zeros([n_stocks, 2])
            U[i, 0] = 1
            U[idx, 1] = R[idx, i]

            V = np.zeros([2, n_stocks])
            V[1, i] = 1
            V[0, idx] = R[i, idx]

            R_other_inv_SM = (R_inv -
                              np.dot(np.dot(R_inv,
                                            np.dot(-U,
                                                   np.linalg.inv(np.eye(2) +
                                                                 np.dot(V,
                                                                        np.dot(R_inv,
                                                                               -U))))),
                                     np.dot(V,
                                            R_inv)))
            R_other_inv_SM = R_other_inv_SM[idx, :][:, idx]

            self.R_inverses[i] = R_other_inv_SM

    def _compute(self, start_date=None, end_date=None):
        """
        trailing_sigma : number of days to use to smoothe the last variance
        refit : number of days between two refits
        """
        if start_date is None:
            start_date = self.data.index.get_level_values(0)[0]
        if end_date is None:
            end_date = self.data.index.get_level_values(0)[-1]

        self.data = self.data.loc[start_date:end_date]

        calendar = self.data.index.get_level_values(0).unique()
        all_companies = set(self.data.index.get_level_values(1).unique().values)

        previous_companies = sorted(
            self.data[self.data.loc[:, "filter"] == 1].index.get_level_values(1).unique().values)
        self.data["signal"] = np.nan
        self.data["signalReverted"] = np.nan

        self.fitter = {ticker: GarchFitter(self.n_past) for ticker in all_companies}
        for k, fit_date in enumerate(tqdm_notebook(calendar[self.n_past:-self.n_fit])):
            start_fit_date = calendar[k]
            current_date = calendar[k + self.n_fit + self.n_past]
            print(current_date)

            df1 = self.data.loc[start_fit_date:current_date]
            temp_filter = (df1.loc[current_date, "filter"] == 1)
            current_companies = sorted(temp_filter.loc[temp_filter.values].index.values)

            temp_prices = df1["adj_close"].unstack().loc[:, current_companies]
            temp_prices = temp_prices.ffill().bfill()
            p = temp_prices.values.T
            r = np.log(p[:, 1:]) - np.log(p[:, :-1])
            r_past = r[:, :self.n_past]
            r_future = r[:, self.n_past:]

            # Every refit days, we refit all our models: single series and correlation matrix.
            if (k % self.refit) == 0:
                print("Starting new series fit.")
                for ticker_index, ticker in enumerate(current_companies):
                    self.fitter[ticker].fit(r_past[ticker_index])

                residuals = np.concatenate([np.reshape(self.fitter[ticker].get_residuals(r_past[ticker_index]),
                                                       newshape=[1, -1])
                                            for ticker_index, ticker in enumerate(current_companies)],
                                           axis=0)
                self.residuals = residuals
                self.set_correlation_matrix(np.corrcoef(residuals))
                if np.min(np.linalg.eigvals(self.R)) < 1e-5:
                    print("Minimum eigenvalue of R is {}".format(np.min(np.linalg.eigvals(self.R))))
                self.mu = np.reshape([self.fitter[ticker].mu for ticker in current_companies],
                                     newshape=[len(current_companies), 1])

            # Case where we lost or won a company
            if set(previous_companies) != set(current_companies):
                added_set = set(current_companies) - set(previous_companies)
                removed_set = set(previous_companies) - set(current_companies)
                if added_set is not {}:
                    print("Added companies : {}".format(added_set))
                if removed_set is not {}:
                    print("Removed companies : {}".format(set(previous_companies) - set(current_companies)))

                for ticker in added_set:
                    ticker_index = current_companies.index(ticker)
                    self.fitter[ticker].fit(r_past[ticker_index])

                residuals = np.concatenate([np.reshape(self.fitter[ticker].get_residuals(r_past[ticker_index]),
                                                       newshape=[1, -1])
                                            for ticker_index, ticker in enumerate(current_companies)],
                                           axis=0)
                self.residuals = residuals
                self.set_correlation_matrix(np.corrcoef(residuals))
                if np.min(np.linalg.eigvals(self.R)) < 1e-5:
                    print("Minimum eigenvalue of R is {}".format(np.min(np.linalg.eigvals(self.R))))
                self.mu = np.reshape([self.fitter[ticker].mu for ticker in current_companies],
                                     newshape=[len(current_companies), 1])

            self.initial_sigma = np.reshape([self.fitter[ticker].getLastVariance(r_past[index_ticker])
                                             for index_ticker, ticker in enumerate(current_companies)],
                                            newshape=[len(current_companies), 1])

            signalMean, signalReverted = self.computeOneSignal(r_future, current_companies)
            self.data.loc[(current_date, list(current_companies)), "signal"] = signalMean
            self.data.loc[(current_date, list(current_companies)), "signalReverted"] = signalReverted

            previous_companies = current_companies

        self._computed = True
        self.signal = self.data.loc[:, "signal"]
        self.signalReverted = self.data.loc[:, "signalReverted"]

    def computeOneSignal(self, r_future, current_companies):
        """
        trailing_sigma : number of days to use to smooth the last variance
        refit : if true, refit the GARCH model
        """
        n_stocks = r_future.shape[0]
        moved = np.sum(r_future, axis=1)
        signalMean = np.zeros(n_stocks)
        signalReverted = np.zeros(n_stocks)

        for ticker_index, ticker in enumerate(current_companies):
            conditional_mu, conditional_sigma = self._computeProbabilityDistributionGaussian(ticker_index,
                                                                                             r_future)
            
            signalMean[ticker_index], signalReverted[ticker_index] = self.computeSignal(conditional_mu,
                                                          conditional_sigma,
                                                          moved[ticker_index])

        return signalMean, signalReverted

    def _computeProbabilityDistributionGaussian(self, index, r_future):
        """
        Compute probability distribution of variable labeled index
        knowing the evolution of the others.
        index : index of the dependant variable in the array
        r_future : past returns on which the model was fitted
        """
        n_step = r_future.shape[1]
        n_stock = r_future.shape[0]
        r_other = np.delete(r_future, index, axis=0)

        # V2
        idx = np.concatenate((np.arange(0, index), np.arange(index + 1, n_stock))).astype(np.int32)
        sigmas_inv = 1 / np.sqrt(self.initial_sigma[idx, 0:1])

        H_inv = sigmas_inv * self.R_inverses[index] * sigmas_inv.T

        # V1
        # H = np.dot(np.diag(np.sqrt(self.initial_sigma[:, 0])),
        #            np.dot(self.R, np.diag(np.sqrt(self.initial_sigma[:, 0]))))
        # idx = np.concatenate((np.arange(0, index), np.arange(index + 1, n_stock))).astype(np.int32)
        # H_other = H[idx][:, idx]
        # H_inv = np.linalg.pinv(H_other)

        rho = self.R[index:index + 1, idx] * np.sqrt(self.initial_sigma[index, 0] * self.initial_sigma[idx, 0])
        sigma1 = np.sqrt(self.initial_sigma[index, 0] - np.dot(rho, np.dot(H_inv, rho.T))[0, 0])
        mumu = self.mu[idx]

        mu1 = self.mu[index, 0] + np.dot(rho, np.dot(H_inv, (np.mean(r_other, axis=1) - mumu)))[0, 0]
        return n_step * mu1, np.sqrt(n_step) * sigma1

    @staticmethod
    def computeSignal(mu, sigma, value, alpha=1.0):
        epsilon = value - mu
        return mu, mu - epsilon
