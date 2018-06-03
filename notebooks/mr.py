%load_ext autoreload
%autoreload 2

import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
%matplotlib inline

from scripts.filters.fixed_volume_fltr import FixedVolumeFilter
from scripts.filters.volume_fltr import VolumeFilter
from scripts.signals.meanreversion import MeanReversion
from scripts.signals.factoresiduals import FactorResidual
from scripts.signals.betaresidual import BetaResidual

# %% load data
raw_data = pd.read_csv('data/data_f2009.csv', parse_dates=['date']).set_index(['date', 'ticker'])
data = raw_data.sort_index()

# %% filter data
nb_stocks = 500
data['filter'] = VolumeFilter(data, nb_stocks, 100).get_filter()

max_returns = 0.1
data.loc[data.future_perf_1d.abs()>max_returns, 'filter'] = 0
data.loc[data.past_perf_1d.abs()>max_returns, 'filter'] = 0

# %% beta residuals
beta_residuals = BetaResidual(data, 252)
data['residual'] = beta_residuals.get_signal()

# %% create signal
mr_signal = MeanReversion(data, 4)
data['signal'] = mr_signal.get_signal()
data.loc[data.residual.abs()>0.1, 'signal'] = np.nan

# %% compute perfs
data_perf = data.loc[data['filter'] == 1].dropna(subset=['signal']).loc["20100105":]
data_perf['position'] = data_perf.signal.groupby(level=0).apply(lambda x: x - np.mean(x))#.clip(-1, 1)
data_perf['perf'] = data_perf.position * data_perf.residual
data_perf['to'] = data_perf.position.groupby(level=1).diff().fillna(0).abs()
data_perf['not'] = data_perf.position.fillna(0).abs()

# data_perf.to_csv("data_{}_{}_{}_{}_{}.csv".format(n_past, n_fit, nb_stocks, trailing_sigma, refit))

# %% perfs
print('Sharpe: ' + "{0:.3f}".format(16 * data_perf.perf.groupby(
    level=0).sum().mean() / data_perf.perf.groupby(level=0).sum().std()))
print('rbt: ' + "{0:.3f}".format(100 * data_perf.perf.groupby(
    level=0).sum().sum() / data_perf.to.groupby(level=0).sum().sum()) + "%")
print('holding: ' + "{0:.2f}".format(2 * data_perf['not'].groupby(
    level=0).sum().sum() / data_perf.to.groupby(level=0).sum().sum()))
data_perf.perf.groupby('date').sum().cumsum().plot(figsize=(12, 7))

data.loc[data.residual>0.25].residual.abs().sort_values(ascending=False).iloc[1:].plot()




# %% Fit multiscaling coef
data['future_perf_2d'] = data.adj_close.groupby(level=1).shift(-2) / data.adj_close - 1
data['future_perf_5d'] = data.adj_close.groupby(level=1).shift(-5) / data.adj_close - 1
data['future_perf_10d'] = data.adj_close.groupby(level=1).shift(-10) / data.adj_close - 1

As = []
Bs = []
ticks = []
for tick in data.loc[data['filter']==1].loc["20100105"].index.get_level_values(1).unique():
    s = data.loc(axis=0)[:, tick].reset_index(level=1, drop=True).loc["20100101":]#.future_perf_1d
    qs = np.arange(0.1, 1, 0.1)
    kq = np.array([(np.mean(s.future_perf_1d.abs()**i)) for i in qs])

    to_fit = np.log(np.array([(np.mean(s.future_perf_10d.abs()**i)) for i in qs])/kq)/(qs*np.log(10))
    r = np.polyfit(qs, to_fit, 1)
    As += [r[1]]
    Bs += [r[0]]
    ticks += [tick]

all_corrs = data.loc[data['filter']==1].loc["20100105":].future_perf_1d.unstack().corr()

s_mean_rho = all_corrs.mean().rename('rho')
s_b = pd.Series(Bs, index=ticks).rename('b')
s_a = pd.Series(As, index=ticks).rename('a')
s_mean_rho = s_mean_rho.loc[s_a.index]
s_mean_rho.shape
aaa = pd.concat([s_mean_rho, s_a, s_b], axis=1).dropna()
aaa.plot(x='rho', y='b', kind='scatter')

aaa.corr()

# %% beta residuals
dd = data.copy()
dd['perf'] = dd.future_perf_1d
idx_perf = dd.loc[dd['filter'] == 1].groupby('date').future_perf_1d.mean().rename('idx_perf')
dd = dd.join(idx_perf)
rw = 252

dd['future_filter'] = dd.groupby(level=1).rolling(rw)['filter'].apply(lambda x:np.sum(x)>0).reset_index(level=0, drop=True).fillna(False)
dd.loc[:, 'future_filter'] = dd.future_filter.groupby(level=1).shift(-rw)
temp_cov = dd.loc[dd.future_filter==1].groupby(level=1, group_keys=False)[['perf', 'idx_perf']].rolling(rw).cov()

temp_cov = temp_cov.reset_index(level=[0], drop=True).sort_index().loc(
    axis=0)[:, :, 'perf'].reset_index(level=2, drop=True).idx_perf.rename('cov')

temp_var = dd.loc[dd.future_filter==1].groupby(level=1, group_keys=False)['idx_perf'].rolling(rw).var()
temp_var = temp_var.reset_index(level=[0], drop=True).sort_index().rename('var')

dd['beta'] = temp_cov / temp_var
data['residuals'] = dd.perf - dd.beta * dd.idx_perf

# %% Test factor Computation
factor_residuals = FactorResidual(data, False)
tttt = factor_residuals.get_signal()

data['future_filter'] = data.groupby(level=1).rolling(252)['filter'].apply(
    lambda x: np.sum(x) > 0).reset_index(level=0, drop=True).fillna(False)
data.loc[:, 'future_filter'] = data.future_filter.groupby(level=1).shift(-252)
temp_var = data.loc[data.future_filter == 1].groupby(level=1, group_keys=False)[
    'future_perf_1d'].rolling(252).var()
temp_var = temp_var.reset_index(level=[0], drop=True).sort_index().rename('var')
temp_var = temp_var.loc[data['filter']==1].dropna()
aa = temp_var.dropna().groupby(level=0).apply(lambda x:x<np.percentile(x, 10))
bb = temp_var.dropna().groupby(level=0).apply(lambda x:x>np.percentile(x, 90))
cc = (data.loc[bb.loc[bb].index]).past_perf_1d.groupby(level=0).mean()-(data.loc[aa.loc[aa].index]).past_perf_1d.groupby(level=0).mean()
#data.loc(axis=0)["20130401":, 'TAM']#.sort_values(by='past_perf_1d', ascending=False)#.reset_index(level=1, drop=True).past_perf_1d.plot()


cc.cumsum().plot()
