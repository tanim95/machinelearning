# -*- coding: utf-8 -*-
"""
Original file is located at
    https://colab.research.google.com/drive/1hjn-w8q6VlYWrKLUU3GUs-BxteO8YtjD

same data that are used in adaboosting method,description is written in adaboost file
"""

from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('./data/mushrooms.csv')
df

X = pd.get_dummies(df.drop('class', axis=1), drop_first=True)
y = df['class']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)

model = GradientBoostingClassifier()

#  learning rate controls the step size, learning rate of 0.1 means that each additional tree will have one-tenth the impact of the previous tree
param = {'n_estimators': [50, 100], 'learning_rate': [
    0.1, 0.05, 0.4], 'max_depth': [3, 5, 10]}
grid_model = GridSearchCV(model, param)

grid_model.fit(X_train, y_train)

grid_model.best_params_

y_pred = grid_model.predict(X_test)

accuracy_score(y_test, y_pred)

important_features = grid_model.best_estimator_.feature_importances_
imp_features = X.columns[important_features > 0.1]
imp_features
