# -*- coding: utf-8 -*-


# KNN is particularly useful when the dataset is small and the number of features is not too large


from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn import neighbors
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('./data/gene_expression.csv')
print(df.head())

sns.scatterplot(data=df, x='Gene One', y='Gene Two',
                hue='Cancer Present', alpha=0.7)
plt.xlim(4, 8)
plt.ylim(4, 8)

X = df.drop('Cancer Present', axis=1)
y = df['Cancer Present']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=101)

scaler = StandardScaler()
scaled_X_train = scaler.fit_transform(X_train)
scaled_X_test = scaler.transform(X_test)

knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(scaled_X_train, y_train)

y_pred = knn_model.predict(scaled_X_test)

accuracy_score(y_test, y_pred)

confusion_matrix(y_test, y_pred)

test_error = []
for k in range(1, 30):
    knn_model_2 = KNeighborsClassifier(n_neighbors=k)
    knn_model_2.fit(scaled_X_train, y_train)
    y_pred_2 = knn_model_2.predict(scaled_X_test)
    error_score = 1 - accuracy_score(y_test, y_pred_2)
    test_error.append(error_score)
test_error

"""#Incresing accuracy usuing elbow method to choose the best k value"""

plt.plot(range(1, 30), test_error)
plt.xlabel('K=VALUE')
plt.ylabel('ERROR RATE')

"""#increasing accuracy usuing pipeline to choose the best perameters

pipeline does the scaling and train the model and choose the best parameters in a encapsulate way so less chance of information leakage and always fit the train data as it pipeline will scale them
"""
pipe = Pipeline([('scaler', StandardScaler()),
                ('knn', KNeighborsClassifier())])

k_values = list(range(1, 20))
param = {'knn__n_neighbors': k_values, }

grid_model = GridSearchCV(pipe, param_grid=param, cv=10, scoring='accuracy')

grid_model.fit(X_train, y_train)

# grid_model.best_params_
grid_model.best_estimator_.get_params()

y_pred = grid_model.predict(X_test)

accuracy_score(y_test, y_pred)

print(classification_report(y_test, y_pred))
