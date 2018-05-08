from .filter import Filter
import numpy as np


class FixedVolumeFilter(Filter):
    def __init__(self, data, nb_stocks, evaluation_day):
        Filter.__init__(self, data)
        self.nb_stocks = nb_stocks
        self.evaluation_day = evaluation_day

    def _compute(self):
        volumefilter = np.argsort(-self.data.adj_volume.loc[self.evaluation_day, :]) < self.nb_stocks
        volumefilter.index = volumefilter.index.droplevel(0)
        volumefilter.name = 'filter'
        self.data = self.data.join(volumefilter, how='inner')
        self.filter = self.data['filter']

