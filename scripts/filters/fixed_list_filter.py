from .filter import Filter
import numpy as np
import pandas as pd


class FixedListFilter(Filter):
    def __init__(self, data, list_stocks):
        Filter.__init__(self, data)
        self.list_stocks = list_stocks

    def _compute(self):
        self.data['filter'] = 0
        self.data.loc[(slice(None), self.list_stocks), 'filter'] = 1

        self.filter = self.data['filter']
