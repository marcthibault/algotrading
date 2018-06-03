import numpy as np
import pandas as pd


class Analyzer():

    def __init__(self, signal):

        self.signal = signal


    def analyze(self):

        tickers = self.signal.index.get_level_values(1).unique()

        results = pd.DataFrame(index=tickers)
        results['Perc_buy'] = (self.signal['position'] > 0).groupby('ticker').mean()

        for ticker in tickers:
            data = self.signal.loc(axis=0)[:, ticker]
            data["perf_2d"] = data["future_perf_1d"] + data["future_perf_1d"].shift(-1)
            data["perf_3d"] = data["perf_2d"] + data["future_perf_1d"].shift(-2)
            data["perf_4d"] = data["perf_3d"] + data["future_perf_1d"].shift(-3)
            data["perf_5d"] = data["perf_4d"] + data["future_perf_1d"].shift(-4)
            data["perf_6d"] = data["perf_5d"] + data["future_perf_1d"].shift(-5)

            results.loc[ticker, "Perf_1d_buy"] = data["future_perf_1d"][data["position"] > 0].mean()
            results.loc[ticker, "Perf_1d_sell"] = data["future_perf_1d"][data["position"] < 0].mean()

            results.loc[ticker, "Freq_up_1d_buy"] = (data["future_perf_1d"][data["position"] > 0] > 0).mean()
            results.loc[ticker, "Freq_up_1d_sell"] = (data["future_perf_1d"][data["position"] < 0] > 0).mean()

            results.loc[ticker, "Perf_2d_buy"] = data["perf_2d"][data["position"] > 0].mean()
            results.loc[ticker, "Perf_2d_sell"] = data["perf_2d"][data["position"] < 0].mean()

            results.loc[ticker, "Freq_up_2d_buy"] = (data["perf_2d"][data["position"] > 0] > 0).mean()
            results.loc[ticker, "Freq_up_2d_sell"] = (data["perf_2d"][data["position"] < 0] > 0).mean()

            results.loc[ticker, "Perf_3d_buy"] = data["perf_3d"][data["position"] > 0].mean()
            results.loc[ticker, "Perf_3d_sell"] = data["perf_3d"][data["position"] < 0].mean()

            results.loc[ticker, "Freq_up_3d_buy"] = (data["perf_3d"][data["position"] > 0] > 0).mean()
            results.loc[ticker, "Freq_up_3d_sell"] = (data["perf_3d"][data["position"] < 0] > 0).mean()

            results.loc[ticker, "Perf_4d_buy"] = data["perf_4d"][data["position"] > 0].mean()
            results.loc[ticker, "Perf_4d_sell"] = data["perf_4d"][data["position"] < 0].mean()

            results.loc[ticker, "Freq_up_4d_buy"] = (data["perf_4d"][data["position"] > 0] > 0).mean()
            results.loc[ticker, "Freq_up_4d_sell"] = (data["perf_4d"][data["position"] < 0] > 0).mean()

            results.loc[ticker, "Perf_5d_buy"] = data["perf_5d"][data["position"] > 0].mean()
            results.loc[ticker, "Perf_5d_sell"] = data["perf_5d"][data["position"] < 0].mean()

            results.loc[ticker, "Freq_up_5d_buy"] = (data["perf_5d"][data["position"] > 0] > 0).mean()
            results.loc[ticker, "Freq_up_5d_sell"] = (data["perf_5d"][data["position"] < 0] > 0).mean()

            results.loc[ticker, "Perf_6d_buy"] = data["perf_6d"][data["position"] > 0].mean()
            results.loc[ticker, "Perf_6d_sell"] = data["perf_6d"][data["position"] < 0].mean()

            results.loc[ticker, "Freq_up_6d_buy"] = (data["perf_6d"][data["position"] > 0] > 0).mean()
            results.loc[ticker, "Freq_up_6d_sell"] = (data["perf_6d"][data["position"] < 0] > 0).mean()

        return results