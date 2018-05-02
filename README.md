# Algo trading class project

Mean reversion using residuals from GARCH-like methods.

## Data

The data contains:
- adjusted close price
- adjusted daily volume
- past returns one day (close to close)
- future returns one day (close to close)

Let's say we trade at close for now. This hypothesis is there to avoid taking into account execution issues right now.
The dataframe has a two-level index : date and stocks (named 'date' and 'ticker').


## Filters

The filters are here to choose the stocks we want to trade at each timestep. Basically, our main
dataframe have a column indicating if a stock is traded or not at this timestep. We want to be able
to access data of a stock even if it is not traded at a time, because we might want to compute
later rolling values for instance. The filtering appear before the signal computation, so that the
signal can have the information, which might be needed in some cases.

The first simplest case is the volume filter, choosing the stocks the most traded the last X days.
We might also think of filtering a given industry, or given other factors (market cap,
volatility...).
