
# [Fraud in Wine](https://en.wikipedia.org/wiki/Wine_fraud)

# Wine fraud relates to the commercial aspects of wine. The most prevalent type of fraud is one where wines are adulterated,
#  usually with the addition of cheaper products (e.g. juices) and sometimes with harmful chemicals and sweeteners
#  (compensating for color or flavor).

# Counterfeiting and the relabelling of inferior and cheaper wines to more expensive
#  brands is another common type of wine fraud.

# ## Project Goals

# A distribution company that was recently a victim of fraud has completed an audit of various samples of wine
#  through the use of chemical analysis on samples. The distribution company specializes in exporting extremely high quality,
#  expensive wines, but was defrauded by a supplier who was attempting to pass off cheap, low quality wine as higher grade wine.
#  The distribution company has hired you to attempt to create a machine learning model that can help detect low quality
# (a.k.a "fraud") wine samples. They want to know if it is even possible to detect such a difference.

# Data Source: *P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis. Modeling wine preferences by data mining from physicochemical properties.
# In Decision Support Systems, Elsevier, 47(4):547-553, 2009.*

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('./data/wine_fraud.csv')
df

df.columns

df['quality'].unique()

sns.countplot(x='quality', data=df)

sns.countplot(x='type', data=df, hue='quality')
# plt.ylim(0,500)

reds = df[df['type'] == 'red']
whites = df[df['type'] == 'white']
red_fraud = reds[reds['quality'] == 'Fraud']
white_fraud = whites[whites['quality'] == 'Fraud']

"""Calculating the percentage of fraud red and fraud white wine"""

r = (len(red_fraud) / len(reds)) * 100
w = (len(white_fraud) / len(whites)) * 100
r, w

df['Fraud'] = df['quality'].map({'Legit': 0, 'Fraud': 1})
df.corr()['Fraud'].sort_values()

df2 = pd.get_dummies(df[['quality', 'type']], drop_first=True)

df3 = df.iloc[:, 0:11]
new_df = pd.concat([df3, df2], axis=1)
new_df

plt.figure(figsize=(9, 9), dpi=100)
sns.heatmap(new_df.corr(), annot=True)

X = new_df.drop('quality_Legit', axis=1)
y = new_df['quality_Legit']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
scaled_X_train = scaler.fit_transform(X_train)
scaled_X_test = scaler.transform(X_test)

"""In SVC(class_weight) class weight parameter is used to balnce weight as here fraud wine value are very less so it will automatically add more weight to the fraud value to regularise """

svc_model = SVC(class_weight='balanced')

param = {'C': [1, 0.1, 10], 'kernel': [
    'linear', 'rbf', 'poly'], 'degree': [2, 3, 4, 5]}
grid = GridSearchCV(estimator=svc_model, param_grid=param)

grid.fit(scaled_X_train, y_train)

grid.best_estimator_.get_params()

y_pred = grid.predict(scaled_X_test)

print(classification_report(y_test, y_pred)), accuracy_score(y_test, y_pred)

confusion_matrix(y_test, y_pred)
