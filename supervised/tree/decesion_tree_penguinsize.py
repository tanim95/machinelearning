# -*- coding: utf-8 -*-


from sklearn.tree import plot_tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('./data/penguins_size.csv')
print(df)

df['species'].unique()

df.isnull().sum()

df = df.dropna()

df['sex'].unique()

df[df['sex'] == '.']

df[df['species'] == 'Gentoo'].groupby('sex').describe().transpose()

df.at[336, 'sex'] = 'FEMALE'

df.loc[336]

X = pd.get_dummies(df.drop('species', axis=True), drop_first=True)

y = df['species']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)


tree_model = DecisionTreeClassifier()
tree_model.fit(X_train, y_train)

tree_model.feature_importances_

pd.DataFrame(index=X.columns, data=tree_model.feature_importances_)


plt.figure(figsize=(10, 8), dpi=200)
plot_tree(tree_model, feature_names=X.columns, filled=True)
