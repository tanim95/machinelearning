# -*- coding: utf-8 -*-

# Original file is located at
#     https://colab.research.google.com/drive/1gJUfagz1-kLsxo2CWUmmxsc2O2B0vMgB


from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from joblib import dump, load
import scipy as sp
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv(".\data\Advertising.csv")

df.head()

"""  # Multiple Features (N-Dimensional)"""

# Relationships between features
sns.pairplot(df, diag_kind='kde')


X = df.drop('sales', axis=1)
y = df['sales']


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)


model = LinearRegression()

model.fit(X_train, y_train)

# X_test

# We only pass in test features
# The model predicts its own y hat
# We can then compare these results to the true y test label value
test_predictions = model.predict(X_test)

test_predictions


MAE = mean_absolute_error(y_test, test_predictions)
MSE = mean_squared_error(y_test, test_predictions)
RMSE = np.sqrt(MSE)
# print(RMSE)
# print(r2_score(y_test,test_predictions))
df['sales'].mean()

quartet = pd.read_csv('./data/anscombes_quartet1.csv')

# y = 3.00 + 0.500x
quartet['pred_y'] = 3 + 0.5 * quartet['x']
quartet['residual'] = quartet['y'] - quartet['pred_y']

sns.scatterplot(data=quartet, x='x', y='y')
sns.lineplot(data=quartet, x='x', y='pred_y', color='red')
plt.vlines(quartet['x'], quartet['y'], quartet['y']-quartet['residual'])
plt.show()
sns.kdeplot(quartet['residual'])
plt.show()
sns.scatterplot(data=quartet, x='y', y='residual')
plt.axhline(y=0, color='r', linestyle='--')
plt.show()
"""---"""

quartet = pd.read_csv('./data/anscombes_quartet2.csv')

quartet.columns = ['x', 'y']

# y = 3.00 + 0.500x
quartet['pred_y'] = 3 + 0.5 * quartet['x']
quartet['residual'] = quartet['y'] - quartet['pred_y']

sns.scatterplot(data=quartet, x='x', y='y')
sns.lineplot(data=quartet, x='x', y='pred_y', color='red')
plt.vlines(quartet['x'], quartet['y'], quartet['y']-quartet['residual'])

sns.kdeplot(quartet['residual'])

sns.scatterplot(data=quartet, x='y', y='residual')
plt.axhline(y=0, color='r', linestyle='--')

quartet = pd.read_csv('./data/anscombes_quartet4.csv')

quartet

# y = 3.00 + 0.500x
quartet['pred_y'] = 3 + 0.5 * quartet['x']

quartet['residual'] = quartet['y'] - quartet['pred_y']

sns.scatterplot(data=quartet, x='x', y='y')
sns.lineplot(data=quartet, x='x', y='pred_y', color='red')
plt.vlines(quartet['x'], quartet['y'], quartet['y']-quartet['residual'])

sns.kdeplot(quartet['residual'])

sns.scatterplot(data=quartet, x='y', y='residual')
plt.axhline(y=0, color='r', linestyle='--')

# Plotting Residuals

test_predictions = model.predict(X_test)

test_res = y_test - test_predictions

sns.scatterplot(x=y_test, y=test_res)
plt.axhline(y=0, color='r', linestyle='--')

len(test_res)

sns.displot(test_res, bins=25, kde=True)


# Retraining Model on Full Data


final_model = LinearRegression()

final_model.fit(X, y)

y_hat = final_model.predict(X)

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(16, 6))

axes[0].plot(df['TV'], df['sales'], 'o')
axes[0].plot(df['TV'], y_hat, 'o', color='red')
axes[0].set_ylabel("Sales")
axes[0].set_title("TV Spend")

axes[1].plot(df['radio'], df['sales'], 'o')
axes[1].plot(df['radio'], y_hat, 'o', color='red')
axes[1].set_title("Radio Spend")
axes[1].set_ylabel("Sales")

axes[2].plot(df['newspaper'], df['sales'], 'o')
axes[2].plot(df['radio'], y_hat, 'o', color='red')
axes[2].set_title("Newspaper Spend")
axes[2].set_ylabel("Sales")
plt.tight_layout()


residuals = y_hat - y

sns.scatterplot(x=y, y=residuals)
plt.axhline(y=0, color='r', linestyle='--')

"""### Coefficients"""

final_model.coef_

coeff_df = pd.DataFrame(final_model.coef_, X.columns, columns=['Coefficient'])
coeff_df

df.corr()

campaign = [[149, 22, 12]]

final_model.predict(campaign)

"""-----

Saving and Loading a Model
"""


dump(final_model, 'sales_model.joblib')

loaded_model = load('sales_model.joblib')

loaded_model.predict(campaign)
