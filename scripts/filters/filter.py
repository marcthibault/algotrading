import pandas as pd


class Filter:
    def __init__(self, data):
        self.data = data.copy()
        self.filter = None
        self._computed = False

    def get_filter(self):
        if ~self._computed:
            self._compute()
            self._computed = True
        return self.filter
