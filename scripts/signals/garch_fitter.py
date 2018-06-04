import numpy as np
from arch import arch_model
from scipy.stats import norm
import warnings
from .series_fitter import SeriesFitter


class GARCHConvergenceException(Exception):
    def __init__(self, message):
        super().__init__(message)


class GarchFitter(SeriesFitter):
    def __init__(self, n_past):
        self.n_past = n_past
        self.garch_fitted = False
        self.mu = 0
        self.w = 0
        self.alpha = 0
        self.beta = 0

        self.last_sigma = 0

    def fit(self, r):
        assert r.shape == (self.n_past,)

        if True:
            am = arch_model(r, {'disp': False})
            if self.garch_fitted and False:
                res = am.fit(disp="off", show_warning=False,
                             starting_values=np.array([self.mu, self.w, self.alpha, self.beta]))
            else:
                res = am.fit(disp="off", show_warning=False)

            mu, w, alpha, beta = res.params.values

            mu = np.nan_to_num(mu)
            w = np.nan_to_num(w)
            alpha = np.nan_to_num(alpha)
            beta = np.nan_to_num(beta)
        else:
            alpha = 1
            beta = 1

        if alpha + beta > 1 - 1e-8 or w < 1e-8:
            # warnings.warn("GARCH Convergence issue: alpha + beta = 1 - {}. Switching to simple mode.".format(
            #     - (1 - alpha - beta)))
            # print("Warning: GARCH Convergence issue: alpha + beta = 1 - {}. Switching to simple mode.".format(
            #     (1 - alpha - beta)))
            self.mu = np.mean(r)
            self.sigma = np.var(r)
            self.garch_fitted = False
            self.last_sigma = self.sigma
        else:
            self.garch_fitted = True
            self.mu, self.w, self.alpha, self.beta = mu, w, alpha, beta
            self.last_sigma = res.conditional_volatility[-1] ** 2

    def get_residuals(self, r):
        assert r.shape == (self.n_past,)
        eps, sigma = self._recoverVariables(r)

        if not self.garch_fitted:
            return eps / np.sqrt(sigma)

        residuals = np.zeros(eps.shape)
        residuals[sigma != 0.] = eps[sigma != 0.] / np.sqrt(sigma[sigma != 0.])
        return residuals

    def _recoverVariables(self, r, initial_sigma=None):
        if not self.garch_fitted:
            eps = r - self.mu
            sigma = self.sigma
            return eps, sigma

        eps = r - self.mu
        sigma = np.zeros_like(r)

        alphabeta = self.alpha + self.beta
        alphabeta = np.minimum(alphabeta, 1 - 1e-4)
        if initial_sigma is None:
            initial_sigma = self.w / (1 - alphabeta)

        initial_sigma = np.maximum(np.minimum(initial_sigma, 1.), 0.)

        sigma[0] = initial_sigma
        for i in range(1, self.n_past):
            sigma[i:i + 1] = self.w + self.alpha * eps[i - 1:i] ** 2 + self.beta * sigma[i - 1:i]

        return eps, sigma

    # def getLastVariance(self, r):
    #     if not self.garch_fitted:
    #         return self.sigma

    #     assert r.shape == (self.n_past,)
    #     _, sigma = self._recoverVariables(r)
    #     return sigma[-1]

    def getLastVariance(self, r):
        if self.garch_fitted:
            last_eps = (r - self.mu)[-1]
            self.last_sigma = self.w + self.alpha * last_eps ** 2 + self.beta * self.last_sigma
        
        return self.last_sigma

    @staticmethod
    def computeQuantile(mu, sigma, value):
        return norm.cdf((mu - value) / 4.0 / sigma)
