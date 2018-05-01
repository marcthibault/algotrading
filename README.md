# Algo trading class project
Mean reversion using residuals from GARCH-like methods.

## Data
The data used is daily close of US stocks (adjusted). Let's say we trade at
close for now.
The dataframe has a two-level index : date and stocks (name 'date' and 'ticker').

We filter stocks by their volume. Let's say we take the 500 biggest stocks on the
last 10 days (2 weeks).  
