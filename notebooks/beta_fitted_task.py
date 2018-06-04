import numpy as np
import pandas as pd
import sys

from scripts.filters import FixedVolumeFilter, SliceFilter
from scripts.filters.volume_fltr import VolumeFilter
from scripts.signals.cross_section_correlation import CrossSectionCorrelation

import argparse

def compute(ARGS):
    input_file = ARGS.input
    print(input_file)
    n_past = 100
    n_fit = 5
    nb_stocks = 100
    trailing_sigma = 1
    refit = 100
    begin = 0
    end = 10
    end_date = pd.to_datetime('2018-01-01')

    with open(input_file) as f:
        for line in f:
            if line.startswith('n_past'):
                n_past = int(line.split(' = ')[-1])
            if line.startswith('n_fit'):
                n_fit = int(line.split(' = ')[-1])
            if line.startswith('nb_stocks'):
                nb_stocks = int(line.split(' = ')[-1])
            if line.startswith('trailing_sigma'):
                trailing_sigma = int(line.split(' = ')[-1])
            if line.startswith('refit'):
                refit = int(line.split(' = ')[-1])
            if line.startswith('end '):
                end = int(line.split(' = ')[-1])
            if line.startswith('begin'):
                begin = int(line.split(' = ')[-1])
            if line.startswith('end_date'):
                end_date = pd.to_datetime(str(line.split(' = ')[-1]))
    
    raw_data = pd.read_csv('../data/data_f2009_withbeta.csv', parse_dates=['date']).set_index(['date', 'ticker'])
    data = raw_data.sort_index()

    data['spy_future_perf'] = data['spy_past_perf'].groupby(level=1).shift(-1)
    data['future_perf_1d_beta'] = data['future_perf_1d'] - data['beta_market'] * data['spy_future_perf']
    data['past_perf_1d_beta'] = data['past_perf_1d'] - data['beta_market'] * data['spy_past_perf']

    # %% filter data
    # nb_stocks = 100
    data['filter'] = SliceFilter(data, "2009-01-05", begin, end).get_filter()
    # %% create signal
    max_returns = 0.1
    data.loc[data.future_perf_1d_beta.abs()>max_returns, 'filter'] = 0
    
    jump_data =data.past_perf_1d_beta.groupby(level=1).rolling(5).sum()
    jump_data.index = jump_data.index.droplevel(0)
    jump_data = jump_data.sort_index(level=0)
    data.loc[jump_data > 0.5, 'filter'] = 0

    # data["filter"].loc(axis=0)[:, 'WBA'] = 0
    # data["filter"].loc(axis=0)[:, 'BTU'] = 0
    # data["filter"].loc(axis=0)[:, 'MCD'] = 0
    # data["filter"].loc(axis=0)[:, 'PM'] = 0
    # data["filter"].loc(axis=0)[:, 'DTV'] = 0

    mr_signal = CrossSectionCorrelation(data, n_past, n_fit, refit)
    data['signal'], data['signalReverted'] = mr_signal.get_signal({'start_date':pd.to_datetime('2010-03-01')})

    # %% compute perfs
    data_perf = data.loc[data['filter'] == 1].dropna(subset=['signal'])
    data_perf['position'] = data_perf.signalReverted.groupby(level=0).apply(lambda x: x - np.mean(x))
    # data_perf["position"] = data_perf["position"].clip(-0.25, 0.25)
    # data_perf['perf'] = data_perf.position * data_perf.future_perf_1d
    data_perf['perf'] = data_perf.position * data_perf.future_perf_1d_beta
    data_perf['to'] = data_perf.position.groupby(level=1).diff().fillna(0).abs()
    data_perf['not'] = data_perf.position.fillna(0).abs()

    data_perf = data_perf.loc(axis=0)[:'2018-01-01', :]

    data_perf.to_csv("data_{}_{}_{}_{}_{}.csv".format(n_past, n_fit, nb_stocks, trailing_sigma, refit))

    # %% perfs
    print('Sharpe: ' + "{0:.3f}".format(16 * data_perf.perf.groupby(
        level=0).sum().mean() / data_perf.perf.groupby(level=0).sum().std()))
    print('rbt: ' + "{0:.3f}".format(100 * data_perf.perf.groupby(
        level=0).sum().sum() / data_perf.to.groupby(level=0).sum().sum()) + "%")
    print('holding: ' + "{0:.2f}".format(2 * data_perf['not'].groupby(
        level=0).sum().sum() / data_perf.to.groupby(level=0).sum().sum()))
    # data_perf.perf.groupby('date').sum().cumsum().plot(figsize=(12, 7))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute signal')
    subparsers = parser.add_subparsers()

    command_parser = subparsers.add_parser('run', help='')
    command_parser.add_argument('-i', '--input', type=str,
                                default='input.in', help="Input file")

    command_parser.set_defaults(func=compute)

    ARGS = parser.parse_args()

    if ARGS.func is None:
        parser.print_help()
        sys.exit(1)
    else:
        ARGS.func(ARGS)