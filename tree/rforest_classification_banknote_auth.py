# -*- coding: utf-8 -*-
"""

Original file is located at
    https://colab.research.google.com/drive/1BmFX9AlnV-G4ommGLN6EYz7GJYlzL4Ee
"""

from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('./data/data_banknote_authentication.csv')
df

sns.pairplot(data=df, hue='Class')

sns.countplot(x='Class', data=df)

X = df.drop('Class', axis=1)
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42)


n_estimators = [10, 50, 100, 200]
bootstrap = [True, False]
max_features = [2, 3, 4]
oob_score = [True, False]

param_grid = {'n_estimators': n_estimators, 'bootstrap': bootstrap,
              'max_features': max_features, 'oob_score': oob_score}

rfc = RandomForestClassifier()
rfc_model = GridSearchCV(rfc, param_grid)

rfc_model.fit(X_train, y_train)

# rfc_model.best_estimator_.get_params()
rfc_model.best_params_

rfc_final = RandomForestClassifier(
    bootstrap=True, max_features=2, n_estimators=50, oob_score=False)

rfc_model.fit(X_train, y_train)

y_pred = rfc_model.predict(X_test)

accuracy_score(y_test, y_pred)

confusion_matrix(y_test, y_pred)

errors = []
miss_classification = []
for n in range(1, 150):
    rfc = RandomForestClassifier(n_estimators=n, max_features=2)
    rfc.fit(X_train, y_train)
    predicion = rfc.predict(X_test)
    err = 1 - accuracy_score(y_test, predicion)
    n_missed = np.sum(y_test != predicion)
    errors.append(err)
    miss_classification.append(n_missed)

plt.plot(range(1, 150), errors)

plt.plot(range(1, 150), miss_classification)
