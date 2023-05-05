# -*- coding: utf-8 -*-


# https://colab.research.google.com/drive/18aGhbwZM68LXhZC5X2D1CA4bLqW7xe4F
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('/content/Advertising.csv')
df.head()

X = df.drop('sales', axis=1)
y = df['sales']

X_train, X_other, y_train, y_other = train_test_split(
    X, y, test_size=0.3, random_state=42)
X_eval, X_test, y_eval, y_test = train_test_split(
    X_other, y_other, test_size=0.5, random_state=101)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_eval = scaler.transform(X_eval)

ridge_model_1 = Ridge(alpha=100)
ridge_model_1.fit(X_train, y_train)
y_pred = ridge_model_1.predict(X_eval)

np.sqrt(mean_squared_error(y_eval, y_pred))
r2_score(y_eval, y_pred)
# mean_squared_error(y_eval,y_pred)

ridge_model_2 = Ridge(alpha=1)
ridge_model_2.fit(X_train, y_train)
y_pred = ridge_model_2.predict(X_eval)

np.sqrt(mean_squared_error(y_eval, y_pred))
# r2_score(y_eval,y_pred)
mean_squared_error(y_eval, y_pred)

y_final_pred = ridge_model_2.predict(X_test)
r2_score(y_eval, y_pred)

"""#Cross Validation"""

score = cross_val_score(ridge_model_2, X_train, y_train,
                        cv=5, scoring='neg_mean_squared_error')
score

abs(score.mean())

score = cross_validate(ridge_model_2, X_train, y_train, cv=5, scoring=(
    'r2', 'neg_mean_squared_error'), return_train_score=True)

score

scores = pd.DataFrame(score)
scores

scores.mean()

y_final_pred = ridge_model_2.predict(X_test)
r2_score(y_eval, y_pred)

"""

#Grid Search

"""

elasticnet_model = ElasticNet()
param = {'alpha': [0.1, 1, 5, 10, 50, 100],
         'l1_ratio': [0.1, 0.5, 0.7, 0.99, 1]}

grid = GridSearchCV(estimator=elasticnet_model, param_grid=param, scoring=(
    'r2', 'neg_mean_squared_error'), refit='r2', cv=7, verbose=1)

# if we use multiple scoring matrics such as r2,neg_mean_squared_error we need to choose refit either false or any of the scoring matrics and if  'refit = false it will not provide any best_estimator,it must be set to true or any scoring matrics to get the best estimator param"""

grid.fit(X_train, y_train)

grid.best_estimator_

grid.best_params_

# pd.DataFrame(grid.cv_results_)

y_pred = grid.predict(X_test)
r2_score(y_test, y_pred)
