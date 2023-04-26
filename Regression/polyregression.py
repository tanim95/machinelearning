# -*- coding: utf-8 -*-


# Original file is located at
#     https://colab.research.google.com/drive/1OTl9bHMaITHdvI-YImzSYVAkK5E6JGwT


from sklearn.metrics import mean_absolute_error, mean_squared_error
from joblib import dump, load
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LassoCV
from sklearn.metrics import r2_score
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv(".\data\Advertising.csv")

df.head()

X = df.drop('sales', axis=1)

y = df['sales']

# Polynomial Regression

polynomial_converter = PolynomialFeatures(degree=2, include_bias=False)
poly_features = polynomial_converter.fit_transform(X)

X.iloc[0]

poly_features[0]

poly_features[0][:3]

poly_features[0][:3]**2

X_train, X_test, y_train, y_test = train_test_split(
    poly_features, y, test_size=0.3, random_state=42)

"""#Data Regularisation(L2)"""

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_train[0]

ridgeCv = RidgeCV(alphas=(0.1, 1, 3, 10))
ridgeCv.fit(X_train, y_train)
ridgeCv.alpha_

ridge_model = Ridge(alpha=0.1)
ridge_model.fit(X_train, y_train)

prediction = ridge_model.predict(X_test)
prediction

r2_score(y_test, prediction)

"""#REGULARISATION L1"""

lasso_model = LassoCV(eps=0.001, n_alphas=100, cv=5)
lasso_model.fit(X_train, y_train)
y__pred = lasso_model.predict(X_test)
lasso_model.alpha_

r2_score(y_test, y__pred)

model = LinearRegression(fit_intercept=True)
model.fit(X_train, y_train)

test_predictions = model.predict(X_test)
test_predictions


MAE = mean_absolute_error(y_test, test_predictions)
MSE = mean_squared_error(y_test, test_predictions)
RMSE = np.sqrt(MSE)


df['sales'].mean()

train_rmse_errors = []
test_rmse_errors = []

for i in range(1, 10):

    polynomial_converter = PolynomialFeatures(degree=i, include_bias=False)
    poly_features = polynomial_converter.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        poly_features, y, test_size=0.3, random_state=101)
    model = LinearRegression(fit_intercept=True)
    model.fit(X_train, y_train)
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    train_RMSE = np.sqrt(mean_squared_error(y_train, train_pred))
    test_RMSE = np.sqrt(mean_squared_error(y_test, test_pred))

    train_rmse_errors.append(train_RMSE)
    test_rmse_errors.append(test_RMSE)
train_rmse_errors

test_rmse_errors

plt.plot(range(1, 6), train_rmse_errors[:5], label='TRAIN')
plt.plot(range(1, 6), test_rmse_errors[:5], label='TEST')
plt.xlabel("DEGREE")
plt.ylabel("RMSE")
plt.legend()

plt.plot(range(1, 10), train_rmse_errors, label='TRAIN')
plt.plot(range(1, 10), test_rmse_errors, label='TEST')
plt.xlabel("Polynomial Complexity")
plt.ylabel("RMSE")
plt.legend()
plt.show()

final_poly_converter = PolynomialFeatures(degree=3, include_bias=False)

final_model = LinearRegression()
featured_data = final_poly_converter.fit_transform(X)
final_model.fit(featured_data, y)


# name are given as i wish
dump(final_model, 'sales_poly_model.joblib')
dump(final_poly_converter, 'poly_converter.joblib')

loaded_poly = load('poly_converter.joblib')
loaded_model = load('sales_poly_model.joblib')

campaign = [[149, 22, 12]]

campaign_poly = loaded_poly.transform(campaign)

campaign_poly

final_model.predict(campaign_poly)
