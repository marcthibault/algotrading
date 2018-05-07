import numpy as np
import pandas as pd

from scripts.filters.volume_fltr import VolumeFilter
from scripts.signals.garch import CCC_GARCH

raw_data = pd.read_csv('../data/data_f2009.csv', parse_dates=['date']).set_index(['date', 'ticker'])
data = raw_data.sort_index()

# %% filter data
nb_stocks = 500
data['filter'] = VolumeFilter(data, nb_stocks).get_filter()

# %% create signal
mr_signal = CCC_GARCH(data, n_past=30, n_fit=1)
data['signal'] = mr_signal.get_signal()

# %% compute perfs
data_perf = data.loc[data['filter'] == 1].dropna(subset=['signal'])
data_perf['position'] = data_perf.signal.groupby(level=0).apply(lambda x: x - np.mean(x))
data_perf['perf'] = data_perf.position * data_perf.future_perf_1d
data_perf['to'] = data_perf.position.groupby(level=1).diff().fillna(0).abs()
data_perf['not'] = data_perf.position.fillna(0).abs()

# %% perfs
print('Sharpe: ' + "{0:.3f}".format(16 * data_perf.perf.groupby(
    level=0).sum().mean() / data_perf.perf.groupby(level=0).sum().std()))
print('rbt: ' + "{0:.3f}".format(100 * data_perf.perf.groupby(
    level=0).sum().sum() / data_perf.to.groupby(level=0).sum().sum()) + "%")
print('holding: ' + "{0:.2f}".format(2 * data_perf['not'].groupby(
    level=0).sum().sum() / data_perf.to.groupby(level=0).sum().sum()))
data_perf.perf.groupby('date').sum().cumsum().plot(figsize=(12, 7))

data_perf.to_csv("output.csv")