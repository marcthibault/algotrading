import numpy as np
import pandas as pd
import sys

from scripts.filters.fixed_volume_fltr import FixedVolumeFilter
from scripts.filters.volume_fltr import VolumeFilter
from scripts.signals.garch import CCC_GARCH

import argparse

def compute(ARGS):
    input_file = ARGS.input
    print(input_file)
    n_past = 100
    n_fit = 5
    nb_stocks = 100
    trailing_sigma = 1
    refit = 100
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
            if line.startswith('end_date'):
                end_date = pd.to_datetime(str(line.split(' = ')[-1]))
    
    raw_data = pd.read_csv('../data/data_f2009.csv', parse_dates=['date']).set_index(['date', 'ticker'])
    data = raw_data.sort_index()

    # %% filter data
    # nb_stocks = 100
    data['filter'] = FixedVolumeFilter(data, nb_stocks, "2009-01-05").get_filter()

    # %% create signal
    mr_signal = CCC_GARCH(data, n_past=n_past, n_fit=n_fit)
    data['signal'] = mr_signal.get_signal({'end_date':end_date, 'trailing_sigma': trailing_sigma, 'refit':refit})

    # %% compute perfs
    data_perf = data.loc[data['filter'] == 1].dropna(subset=['signal'])
    data_perf['position'] = data_perf.signal.groupby(level=0).apply(lambda x: x - np.mean(x))
    data_perf['perf'] = data_perf.position * data_perf.future_perf_1d
    data_perf['to'] = data_perf.position.groupby(level=1).diff().fillna(0).abs()
    data_perf['not'] = data_perf.position.fillna(0).abs()

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