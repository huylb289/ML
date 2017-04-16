import pandas as pd
import quandl
import math
import numpy as np
from sklearn import preprocessing, svm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
##http://stackoverflow.com/questions/41238769/warning-messages-when-using-python

apikeyQuandl = 'xZEUj1LcWi8MxLy6rz_V'

df = quandl.get("WIKI/GOOGL", authtoken="xZEUj1LcWi8MxLy6rz_V")
df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close'] * 100.0 # Different from High and Low
df['CHANGE_PCT'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0 # Different from Open and Close

df = df[['Adj. Close', 'HL_PCT', 'CHANGE_PCT', 'Adj. Volume']]

forecast_col = 'Adj. Close'
df.fillna(-99999, inplace=True) # in machine learning, we need to get rip of the nan

forecast_out = int(math.ceil(0.01*len(df))) # round everything up (~30 days shift)
print(forecast_out)

##http://stackoverflow.com/questions/20095673/python-shift-column-in-pandas-dataframe-up-by-one
df['label'] = df[forecast_col].shift(-forecast_out)
df.dropna(inplace=True)

X = np.array(df.drop(['label'], 1))
y = np.array(df['label'])
X = preprocessing.scale(X) # like normalization data
y = np.array(df['label'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

clf = LinearRegression()
##clf = svm.SVR(kernel='poly')
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)

print(accuracy)
