import pandas as pd
import quandl
import datetime
import math
import numpy as np
from sklearn import preprocessing, svm
from sklearn.model_selection import cross_validate
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')


df = quandl.get("WIKI/GOOGL")
df.columns = df.columns.to_series().apply(lambda x: x.strip())
df = df[['Adj. Open', "Adj. High", "Adj. Low", "Adj. Close", "Adj. Volume", ]]
df['HL_PCT'] = (df["Adj. High"]-df["Adj. Close"]) / df["Adj. Close"]*100
df['PCT_chag'] = (df["Adj. Close"]-df["Adj. Open"]) / df["Adj. Close"]*100
df = df[["Adj. Close", "HL_PCT", "PCT_chag"]]
forcast_col = "Adj. Close"
df.fillna(-99999, inplace=True)
# taking random number of days till which we want to predict stock price int(math.ceil(0.01*len(df)))
forcast_out = 800

# shifting label value by forcast out number
df['label'] = df[forcast_col].shift(-forcast_out)

# taking attributes (which are all cloumns expect label (y))
X = np.array(df.drop(['label'], 1))

# scaling to save time
X = preprocessing.scale(X)
X = X[:-forcast_out]

# values for prediction
X_lately = X[-forcast_out:]

df.dropna(inplace=True)
y = np.array(df['label'])

# dividing data into training and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
clf = LinearRegression(n_jobs=-1)
clf.fit(X_train, y_train)
acc = clf.score(X_test, y_test)
# predicting values
forcast_set = clf.predict(X_lately)

df["Forecast"] = np.nan

# taking dates in last_date
last_date = df.iloc[-1].name
# conv of human date to epic
last_unix = last_date.timestamp()

one_day = 86400
next_unix = last_unix+one_day

for i in forcast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)]+[i]

df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('date')
plt.ylabel('price')
plt.show()
