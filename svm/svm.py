# -*- coding: utf-8 -*-

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('./data/mouse_viral_study.csv')
df

sns.scatterplot(data=df, x='Med_1_mL', y='Med_2_mL', hue='Virus Present')

"""#SVM for Classification task

"""

X = df.drop('Virus Present', axis=1)
y = df['Virus Present']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=101)

scaler = StandardScaler()
scaled_X_train = scaler.fit_transform(X_train)
scaled_X_test = scaler.transform(X_test)

svm_model = SVC(kernel='linear', C=100)
svm_model.fit(X_train, y_train)


def plot_svm_boundary(model, X, y):

    X = X.values
    y = y.values

    # Scatter Plot
    plt.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap='seismic')

    # plot the decision function
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # create grid to evaluate model
    xx = np.linspace(xlim[0], xlim[1], 30)
    yy = np.linspace(ylim[0], ylim[1], 30)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = model.decision_function(xy).reshape(XX.shape)

    # plot decision boundary and margins
    ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
               linestyles=['--', '-', '--'])
    # plot support vectors
    ax.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1], s=100,
               linewidth=1, facecolors='none', edgecolors='k')
    plt.show()


plot_svm_boundary(svm_model, X_train, y_train)

svm_model = SVC(kernel='linear', C=0.1)
svm_model.fit(X_train, y_train)
plot_svm_boundary(svm_model, X_train, y_train)

"""if we use gamma = 'auto' it will pick up more noise and cause overfit..the default value is 'scale' but we can choose gamma value as we like i.g 0.1,5,10 the higher the valu the more it will capture noise. the formula for gamma =( 1 / no of feature) ,in our data no of features is 2"""

svm_model = SVC(kernel='rbf', C=10, gamma='auto')
svm_model.fit(X_train, y_train)
plot_svm_boundary(svm_model, X_train, y_train)

svm_model = SVC(kernel='rbf', C=0.5, gamma='scale')
svm_model.fit(X_train, y_train)
plot_svm_boundary(svm_model, X_train, y_train)

svm = SVC()
param = {'C': [1, 0.1, 10], 'kernel': [
    'linear', 'rbf', 'poly'], 'degree': [2, 3, 4, 5]}

grid = GridSearchCV(svm, param_grid=param)

grid.fit(X_train, y_train)

grid.best_estimator_.get_params()

y_pred = grid.predict(X_test)


accuracy_score(y_test, y_pred)

print(classification_report(y_test, y_pred))

"""#SVM for Regression task
we have data for a concreate slump and we need to predict its strength after 28 days later without having to wait 28 days
"""

df2 = pd.read_csv('./data/cement_slump.csv')
df2

# sns.pairplot(data = df2,hue = 'Compressive Strength (28-day)(Mpa)')

sns.heatmap(df2.corr(), annot=True,)

df2.columns

X = df2.drop('Compressive Strength (28-day)(Mpa)', axis=1)
y = df2['Compressive Strength (28-day)(Mpa)']

X_train2, X_test2, y_train2, y_test2 = train_test_split(
    X, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
scaled_X_train2 = scaler.fit_transform(X_train2)
scaled_X_test2 = scaler.transform(X_test2)

svr_model = SVR()
# help(SVR)

param2 = {'C': [1, 0.1, 10], 'kernel': ['linear', 'rbf', 'poly'],
          'degree': [2, 3, 4, 5], 'epsilon': [0, 0.1, 0.5, 2]}
grid2 = GridSearchCV(svr_model, param_grid=param2)

grid2.fit(X_train2, y_train2)

grid2.best_estimator_.get_params()

y_pred2 = grid2.predict(X_test2)


np.sqrt(mean_squared_error(y_test2, y_pred2))

r2_score(y_test2, y_pred2)
