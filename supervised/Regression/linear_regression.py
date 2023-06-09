# -*- coding: utf-8 -*-
# LINEAR REGRESSION.ipynb

from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data = np.random.randint(0, 101, (4, 3))
data

df = pd.DataFrame(data)
df

dataa = pd.read_csv('.\data\california_housing_train.csv')
dataa.head()


dff = dataa.iloc[0]
dataa.drop(dataa.index[-1], axis=0, inplace=True)
dataa

dataa[(dataa['total_rooms'] > 5000) & (dataa['latitude'] > 30)]


def lasttwodigits(perm):
    return (str(perm)[-2:])


dataa['households'].apply(lasttwodigits)

# data2 = np.vectorize(lasttwodigits)(dataa['households'],dataa['latitude'])

dataa.sort_values('population')

dataa.corr()

dataa['population'].idxmax()

dataa.iloc[978]

print(dataa.groupby('median_house_value').mean().transpose())

dataa['total'] = dataa['latitude'] + dataa['longitude']
dataa

X = dataa['total_bedrooms']
y = dataa['median_house_value']
np.polyfit(X, y, deg=1)

potential_spend = np.linspace(100, 5000, 500)
predected_val = 2.25468672e+01*potential_spend + 1.93797456e+05
plt.plot(potential_spend, predected_val)

dataa.drop('total', axis=1)

X = dataa.iloc[:, 0:-3]
y = dataa.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

"""Confusion matrics is used for classification task and R squared is used for Regression task so we use R squared here"""

y_pred = model.predict(X_test)

"""making liner relation into polynomial to better understadn the relaiton bteween the variable"""

poly_converter = PolynomialFeatures(degree=2, include_bias=True)

poly_feature = poly_converter.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(
    poly_feature, y, test_size=0.30, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

"""Confusion matrics is used for classification task and R squared is used for Regression task so we use R squared here

Calculating error
"""

MAE = mean_absolute_error(y_test, y_pred)
MSE = mean_squared_error(y_test, y_pred)
RMSE = np.sqrt(MSE)
print(MAE, MSE, RMSE)

"""Testing which degree of polynomial fits best in our model"""

for i in range(1, 10):
    poly_converter = PolynomialFeatures(degree=i, include_bias=False)
    poly_feature = poly_converter.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        poly_feature, y, test_size=0.30, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

dataa.corr()['median_house_value']

plt.scatter(dataa['median_house_value'], dataa['median_income'])
