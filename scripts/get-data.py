import pandas as pd
import quandl
from tqdm import tqdm
import requests

quandl.ApiConfig.api_key = 'baSZrmyEH9Vv-rFW83U6'

tickers_list_df = pd.read_csv("data/WIKI-datasets-codes.csv", header=None)
tickers_list = [ticker.split("/")[-1] for ticker in tickers_list_df.loc[:, 0].values]

data_list = []
for iticker, ticker in enumerate(tqdm(tickers_list)):
    try:
        data_list.append(quandl.get_table('WIKI/PRICES', ticker=ticker, paginate=True))
    except (quandl.QuandlError, requests.exceptions.ConnectionError) as e:
        print(e, "\nStopped downloading at index {}. \nDumped the data in 'data/WIKI-data-all.csv'".format(iticker))
        break

data_df = pd.concat(data_list).set_index(["date", "ticker"]).sort_index()
data_df.to_csv("data/WIKI-data-all.csv")
