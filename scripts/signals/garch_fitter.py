import numpy as np
from arch import arch_model
from scipy.linalg import sqrtm
from scipy.stats import norm
from .series_fitter import SeriesFitter


class GARCHConvergenceException(Exception):
    def __init__(self, message):
        super().__init__(message)


class GarchFitter(SeriesFitter):
    def __init__(self, n_past):
        self.n_past = n_past
        self.fitted = False

        self.mu = 0
        self.w = 0
        self.alpha = 0
        self.beta = 0

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

    def fit(self, r):
        assert r.shape == (self.n_past,)

        am = arch_model(r, {'disp': False})
        if self.fitted:
            res = am.fit(disp="off", show_warning=False,
                         starting_values=np.array([self.mu, self.w, self.alpha, self.beta]))
        else:
            res = am.fit(disp="off", show_warning=False)

        mu, w, alpha, beta = res.params.values

        mu = np.nan_to_num(mu)
        w = np.nan_to_num(w)
        alpha = np.nan_to_num(alpha)
        beta = np.nan_to_num(beta)

        if 1 - self.alpha - self.beta < 1e-8:
            raise GARCHConvergenceException("alpha + beta = 1 - {}".format(- (1 - self.alpha - self.beta)))

        self.fitted = True
        self.mu, self.w, self.alpha, self.beta = mu, w, alpha, beta

    def get_residuals(self, r):
        assert r.shape == (self.n_past,)
        eps, sigma = self._recoverVariables(r)

        residuals = np.zeros(sigma.shape)
        residuals[sigma != 0.] = eps[sigma != 0.] / np.sqrt(sigma[sigma != 0.])
        return residuals

    def _recoverVariables(self, r, initial_sigma=None):
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

    def _getLastVariance(self, r):
        assert r.shape == (self.n_past,)
        _, sigma = self._recoverVariables(r)
        return sigma[:, -1:]

    @staticmethod
    def computeQuantile(mu, sigma, value):
        return norm.cdf((mu - value) / 4.0 / sigma)
