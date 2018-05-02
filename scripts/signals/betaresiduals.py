from .signal import Signal
import numpy as np
import pandas as pd

'''
Beta residuals of stock returns.
'''


class MeanReversion(Signal):
    def __init__(self, data, rolling_window = 252):
        Signal.__init__(self, data)
        self.rolling_window = rolling_window

    def _compute(self):
        data['']
