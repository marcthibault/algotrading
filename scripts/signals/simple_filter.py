import numpy as np
from scipy.stats import norm
from .series_fitter import SeriesFitter


class SimpleFitter(SeriesFitter):
    def __init__(self, n_past):
        self.n_past = n_past
        self.fitted = False

        self.mu = 0
        self.sigma = 0

    def fit(self, r):
        assert r.shape == (self.n_past,)

        self.mu = np.mean(r)
        self.sigma = np.std(r)

        self.fitted = True

    def get_residuals(self, r):
        eps, sigma = self._recoverVariables(r)

        residuals = eps / np.sqrt(sigma)
        return residuals

    def _recoverVariables(self, r):
        eps = r - self.mu
        sigma = self.sigma
        return eps, sigma

    def _getLastVariance(self, r):
        _, sigma = self._recoverVariables(r)
        return sigma[:, -1:]

    @staticmethod
    def computeQuantile(mu, sigma, value):
        return norm.cdf((mu - value) / 4.0 / sigma)
