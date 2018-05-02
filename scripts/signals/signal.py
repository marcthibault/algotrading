import pandas as pd

'''
Basic signal computation class.
'''


class Signal:
    def __init__(self, data):
        self.data = data
        self._computed = False

    def get_signal(self):
        if ~self._computed:
            self._compute()
            self._computed = True
            self.signal = self.signal.rename('signal')
        return self.signal

    def compute(self):
        if ~self._computed:
            self._compute()
            self._computed = True
            self.signal = self.signal.rename('signal')

    def save(self, file):
        if self._computed:
            self.signal.to_csv("../results/" + file + ".csv")
            print(">> Results saved in " + file + ".")
        else:
            print(">> You must compute the results before saving them.")
