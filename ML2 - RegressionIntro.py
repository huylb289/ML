import pandas as pd
import quandl

apikeyQuandl = 'xZEUj1LcWi8MxLy6rz_V'

df = quandl.get("WIKI/GOOGL", authtoken="xZEUj1LcWi8MxLy6rz_V")
df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close'] * 100.0 # Different from High and Low
df['CHANGE_PCT'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0 # Different from Open and Close

df = df[['Adj. Close', 'HL_PCT', 'CHANGE_PCT', 'Adj. Volume']]
##print(df.head())

forecast_col = 'Adj. Close'


