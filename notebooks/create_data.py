import pandas as pd
%matplotlib inline
data_close = pd.read_csv('data/adjusted_prices.csv', parse_dates=['date'])
data_close = pd.DataFrame(data_close.set_index('date').stack().rename('close'))
data_close = data_close.reset_index(level=1).rename(
    columns={'level_1': 'ticker'}).reset_index().set_index(['date', 'ticker'])

data_volume = pd.read_csv('data/adjusted_volumes.csv', parse_dates=['date'])
data_volume = pd.DataFrame(data_volume.set_index('date').stack().rename('volume'))
data_volume = data_volume.reset_index(level=1).rename(
    columns={'level_1': 'ticker'}).reset_index().set_index(['date', 'ticker'])

data = pd.merge(data_volume, data_close, left_index = True, right_index=True)
data['future_perf_1d'] = data.close.groupby(level=1).shift(-1) / data.close - 1
data['past_perf_1d'] = data.close / data.close.groupby(level=1).shift(1) - 1

data.loc["20090105":].reset_index().to_csv('data/data_f2009.csv', index=False)
