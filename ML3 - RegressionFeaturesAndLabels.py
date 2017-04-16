import pandas as pd
import quandl
import math

apikeyQuandl = 'xZEUj1LcWi8MxLy6rz_V'

df = quandl.get("WIKI/GOOGL", authtoken="xZEUj1LcWi8MxLy6rz_V")
df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close'] * 100.0 # Different from High and Low
df['CHANGE_PCT'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0 # Different from Open and Close

df = df[['Adj. Close', 'HL_PCT', 'CHANGE_PCT', 'Adj. Volume']]

forecast_col = 'Adj. Close'
df.fillna(-99999, inplace=True) # in machine learning, we need to get rip of the nan

##ceil(...)
##    ceil(x)
##    
##    Return the ceiling of x as an Integral.
##    This is the smallest integer >= x.

##Simon Chan8 months ago
##This is how I understood sentdex logic,
##he is taking 0.01 or 1% of the length of all
##the rows within the dataframe. Each row in
##the dataFrame is representative of a day in
##the life of the stock. So if the stock has been trading for 365 days,
##there will be 365 rows in the dataFrame. 1% of 365 is 3.65 days
##which is then rounded up by the math.ceil function to 4 days.
##The 4 days will be the forecast _out variable which is the variable
##that used to shift the Adj.Close price column in the dataFame up by 4.
##In other words, if you were standing at day 1 of the stock when
##it was first traded, the prediction or the 'label' from his algorithm
##would tell you that at day 4, your stock will be valued at the amount
##of the close as taken on day 4 from actual data.
##This isn't totally useful information since you can look at
##the Adj.Close column on day 4 to get back to the label info on day 1.
##This is really all done to build a training set so that the machine can
##learn from the trend.﻿

forecast_out = int(math.ceil(0.01*len(df))) # round everything up

##http://stackoverflow.com/questions/20095673/python-shift-column-in-pandas-dataframe-up-by-one
df['label'] = df[forecast_col].shift(-forecast_out)
df.dropna(inplace=True)
print(df.tail())
