from matplotlib import style
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn import preprocessing;
from sklearn import model_selection;
from sklearn import linear_model;

import csv
import datetime
import math
import matplotlib
matplotlib.use("MacOSX")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas_datareader.data as web


dates = []
prices = []
start = datetime.datetime(2018, 9, 1)
end = datetime.datetime(2019, 8, 31)


def get_data_from_yahoo():
    return web.DataReader("AAPL", 'yahoo', start, end)


def save_to_csv(df, filename='data/AAPL.csv'):
    df.to_csv(filename)


def get_data(filename):
    with open(filename, 'r') as csvfile:
        csv_file_feader = csv.reader(csvfile)
        next(csv_file_feader)
        for row in csv_file_feader:
            dates.append(int(row[0].split('-')[0]))
            prices.append(float(row[1]))
    return dates, prices


def predict_prices(dates, prices, x):
    dates = np.reshape(dates, (len(dates), 1))

    svr_lin = SVR(kernel ='linear', C=1e3, gamma='scale')
    svr_poly = SVR(kernel='poly', C=1e3, degree=2, gamma='scale')
    svr_rbf = SVR(kernel='rbf', C=1e3, gamma='scale')
    svr_lin.fit(dates, prices)
    svr_poly.fit(dates, prices)
    svr_rbf.fit(dates, prices)

    plt.scatter(dates, prices, color='black', label='Data')
    plt.plot(dates, svr_rbf.predict(dates), color='red', label='RBF model')
    plt.plot(dates, svr_lin.predict(dates), color='green', label='Linear model')
    plt.plot(dates, svr_poly.predict(dates), color='blue', label='Polynomial model')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('Support Vector Regression')
    plt.legend
    plt.show()
    return svr_rbf.predict(x)[0], svr_lin.predict(x)[0], svr_poly.predict(x)[0]


def rolling_mean(df):
    print('Rolling mean:')
    close_px = df['Adj Close']
    mavg = close_px.rolling(window=100).mean()
    print(mavg)

    # Adjusting the style of matplotlib
    style.use('ggplot')
    # moving average for stock closing price
    close_px.plot(label='AAPL')
    mavg.plot(label='mavg')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend
    plt.show()
    plt.close()


def diviation(df):
    # Return Deviation — to determine risk and return
    close_px = df['Adj Close']
    rets = close_px / close_px.shift(1) - 1
    rets.plot(label='return')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend
    plt.show()
    plt.close()


def correlation_competitors():
    # Competitors Stocks
    dfcomp = web.DataReader(['AAPL', 'GE', 'GOOG', 'IBM', 'MSFT'], 'yahoo', start=start, end=end)['Adj Close']
    # Correlation Analysis — Does one competitor affect others?
    retscomp = dfcomp.pct_change()
    corr = retscomp.corr()
    # APPLE, GE distributions
    plt.scatter(retscomp.AAPL, retscomp.GE)
    plt.xlabel('Returns AAPL')
    plt.ylabel('Returns GE')
    plt.legend
    plt.show()
    plt.close()


def kde():
    dfcomp = web.DataReader(['AAPL', 'GE', 'GOOG', 'IBM', 'MSFT'], 'yahoo', start=start, end=end)['Adj Close']
    retscomp = dfcomp.pct_change()
    corr = retscomp.corr()
    # kernel density estimation(KDE).
    pd.plotting.scatter_matrix(retscomp, diagonal='kde', figsize=(10, 10));
    plt.show()
    plt.close()
    # heat maps to visualize the correlation ranges among the competing stocks.
    plt.imshow(corr, cmap='hot', interpolation='none')
    plt.colorbar()
    plt.xticks(range(len(corr)), corr.columns)
    plt.yticks(range(len(corr)), corr.columns);
    plt.legend
    plt.show()
    plt.close()


def stock_return_risk():
    dfcomp = web.DataReader(['AAPL', 'GE', 'GOOG', 'IBM', 'MSFT'], 'yahoo', start=start, end=end)['Adj Close']
    retscomp = dfcomp.pct_change()
    plt.scatter(retscomp.mean(), retscomp.std())
    plt.xlabel('Expected returns')
    plt.ylabel('Risk')
    for label, x, y in zip(retscomp.columns, retscomp.mean(), retscomp.std()):
        plt.annotate(
            label,
            xy=(x, y), xytext=(20, -20),
            textcoords='offset points', ha='right', va='bottom',
            bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    plt.legend
    plt.show()
    plt.close()


def show_plot(dates,prices):
    linear_mod = linear_model.LinearRegression()
    dates = np.reshape(dates, (len(dates), 1))
    prices = np.reshape(prices, (len(prices), 1))
    linear_mod.fit(dates, prices)
    plt.scatter(dates, prices, color='yellow')
    plt.plot(dates, linear_mod.predict(dates), color='blue', linewidth=3)
    plt.show()


def predict_price(dates, prices, x):
    linear_mod = linear_model.LinearRegression()
    dates = np.reshape(dates, (len(dates), 1))
    prices = np.reshape(prices, (len(prices), 1))
    linear_mod.fit(dates, prices)
    predicted_price = linear_mod.predict(x)
    return predicted_price[0][0], linear_mod.coef_[0][0], linear_mod.intercept_[0]


def prepare_data(df,forecast_col,forecast_out,test_size):
    label = df[forecast_col].shift(-forecast_out);#creating new column called label with the last 5 rows are nan
    X = np.array(df[[forecast_col]]); #creating the feature array
    X = preprocessing.scale(X) #processing the feature array
    X_lately = X[-forecast_out:] #creating the column i want to use later in the predicting method
    X = X[:-forecast_out] # X that will contain the training and testing
    label.dropna(inplace=True); #dropping na values
    y = np.array(label)  # assigning Y
    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, y, test_size=test_size) #cross validation

    response = [X_train,X_test , Y_train, Y_test , X_lately];
    return response;


def linear_regression(X_train, X_test, Y_train, Y_test , X_lately):
    # Linear regression
    clfreg = linear_model.LinearRegression(n_jobs=-1)
    # training the linear regression model
    clfreg.fit(X_train, Y_train)
    # testing the linear regression model
    score = clfreg.score(X_test, Y_test);
    print('Linear regression:')
    forcast = clfreg.predict(X_lately);  # set that will contain the forecasted data
    confidencereg = clfreg.score(X_test, Y_test)
    return forcast, confidencereg


def quadratic_regression2(X_train, X_test, Y_train, Y_test , X_lately):
    # Quadratic Regression
    print('Quadratic Regression')
    clfpoly2 = make_pipeline(PolynomialFeatures(2), Ridge())
    clfpoly2.fit(X_train, Y_train)
    score = clfpoly2.score(X_test, Y_test);  # testing the linear regression model
    forecast = clfpoly2.predict(X_lately);  # set that will contain the forecasted data
    confidencepoly2 = clfpoly2.score(X_test, Y_test)
    return forecast, confidencepoly2


def quadratic_regression3(X_train, X_test, Y_train, Y_test , X_lately):
    # Quadratic Regression
    print('Quadratic Regression')
    # Quadratic Regression
    clfpoly3 = make_pipeline(PolynomialFeatures(3), Ridge())
    clfpoly3.fit(X_train, Y_train)
    score = clfpoly3.score(X_test, Y_test);  # testing the linear regression model
    forecast = clfpoly3.predict(X_lately);  # set that will contain the forecasted data
    confidencepoly3 = clfpoly3.score(X_test, Y_test)
    return forecast, confidencepoly3


def knn_regression(X_train, X_test, Y_train, Y_test , X_lately):
    # KNN Regression
    print('KNN Regression:')
    clfknn = KNeighborsRegressor(n_neighbors=2)
    clfknn.fit(X_train, Y_train)
    score = clfknn.score(X_test, Y_test);  # testing the linear regression model
    forecast = clfknn.predict(X_lately);  # set that will contain the forecasted data
    confidenceknn = clfknn.score(X_test, Y_test)
    return forecast, confidenceknn


def plot(df, forecast, title):
    dfreg = df.loc[:, ['Adj Close', 'Volume']]
    dfreg['Forecast'] = np.nan

    last_date = dfreg.iloc[-1].name
    last_unix = last_date
    next_unix = last_unix + datetime.timedelta(days=1)
    for i in forecast:
        next_date = next_unix
        next_unix += datetime.timedelta(days=1)
        dfreg.loc[next_date] = [np.nan for _ in range(len(dfreg.columns) - 1)] + [i]
    dfreg['Adj Close'].tail(500).plot()
    dfreg['Forecast'].tail(500).plot()
    plt.title(title)
    plt.legend(loc=4)
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.show()


df = get_data_from_yahoo()

# choosing which column to forecast
forecast_col = 'Close'
# how far to forecast
forecast_out = 5
# the size of my test set
test_size = 0.2;

# calling the method were the cross validation and data preperation is in
X_train, X_test, Y_train, Y_test , X_lately = \
    prepare_data(df,forecast_col,forecast_out,test_size);


forecast, confidence = linear_regression(X_train, X_test, Y_train, Y_test , X_lately)
plot(df, forecast, 'Linear Regression')
print('Confidence: {0}'.format(confidence))

forecast, confidence = quadratic_regression2(X_train, X_test, Y_train, Y_test , X_lately)
plot(df, forecast, 'Quadratic Regression 2')
print('Confidence: {0}'.format(confidence))

forecast, confidence = quadratic_regression3(X_train, X_test, Y_train, Y_test , X_lately)
plot(df, forecast, 'Quadratic Regression 3')
print('Confidence: {0}'.format(confidence))

forecast, confidence = knn_regression(X_train, X_test, Y_train, Y_test , X_lately)
plot(df, forecast, 'KNN Regression')
print('Confidence: {0}'.format(confidence))
