import numpy as np
from arch import arch_model
from scipy.linalg import sqrtm
from scipy.stats import norm


class SeriesFitter(object):
    def __init__(self):
        pass

    def fit(self, r):
        pass

    def get_residuals(self, r):
        pass

    def _recoverVariables(self, r):
        pass

    def _getLastVariance(self, r):
        pass

    @staticmethod
    def computeQuantile(mu, sigma, value):
        pass
