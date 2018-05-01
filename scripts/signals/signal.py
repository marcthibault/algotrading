import pandas as pd

'''
Basic signal computation class.
'''


class Signal:
    def __init__(self, data):
        self.data = data
        self.computed = False

    def compute(self, ):
        self.signal = pd.Series()
        self._computed()

    def _computed(self, ):
        self.computed = True

    def save(self, file):
        if self.computed():
            self.signal.to_csv("../results/" + file + ".csv")
            print(">> Results saved in " + file + ".")
        else:
            print(">> You must compute the results before saving them.")
