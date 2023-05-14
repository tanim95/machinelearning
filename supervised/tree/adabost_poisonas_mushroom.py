# -*- coding: utf-8 -*-
"""adabost_poisonas_mushroom.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Nxe6YU3yP3ZTGr0oqQ-viRYID5drjwda

The goal is to identify safe mushrooms also give some necessary gudlines to identify poisonous mushroom as ada boost uses stump which depends on one feature at a time from this idea we can give some guidelines for which mushroom is poisonous and which not!

#Mushroom Hunting: Edible or Poisonous?

Data Source: https://archive.ics.uci.edu/ml/datasets/Mushroom


This data set includes descriptions of hypothetical samples corresponding to 23 species of gilled mushrooms in the Agaricus and Lepiota Family (pp. 500-525). Each species is identified as definitely edible, definitely poisonous, or of unknown edibility and not recommended. This latter class was combined with the poisonous one. The Guide clearly states that there is no simple rule for determining the edibility of a mushroom; no rule like ``leaflets three, let it be'' for Poisonous Oak and Ivy.


Attribute Information:

1. cap-shape: bell=b,conical=c,convex=x,flat=f, knobbed=k,sunken=s
2. cap-surface: fibrous=f,grooves=g,scaly=y,smooth=s
3. cap-color: brown=n,buff=b,cinnamon=c,gray=g,green=r, pink=p,purple=u,red=e,white=w,yellow=y
4. bruises?: bruises=t,no=f
5. odor: almond=a,anise=l,creosote=c,fishy=y,foul=f, musty=m,none=n,pungent=p,spicy=s
6. gill-attachment: attached=a,descending=d,free=f,notched=n
7. gill-spacing: close=c,crowded=w,distant=d
8. gill-size: broad=b,narrow=n
9. gill-color: black=k,brown=n,buff=b,chocolate=h,gray=g, green=r,orange=o,pink=p,purple=u,red=e, white=w,yellow=y
10. stalk-shape: enlarging=e,tapering=t
11. stalk-root: bulbous=b,club=c,cup=u,equal=e, rhizomorphs=z,rooted=r,missing=?
12. stalk-surface-above-ring: fibrous=f,scaly=y,silky=k,smooth=s
13. stalk-surface-below-ring: fibrous=f,scaly=y,silky=k,smooth=s
14. stalk-color-above-ring: brown=n,buff=b,cinnamon=c,gray=g,orange=o, pink=p,red=e,white=w,yellow=y
15. stalk-color-below-ring: brown=n,buff=b,cinnamon=c,gray=g,orange=o, pink=p,red=e,white=w,yellow=y
16. veil-type: partial=p,universal=u
17. veil-color: brown=n,orange=o,white=w,yellow=y
18. ring-number: none=n,one=o,two=t
19. ring-type: cobwebby=c,evanescent=e,flaring=f,large=l, none=n,pendant=p,sheathing=s,zone=z
20. spore-print-color: black=k,brown=n,buff=b,chocolate=h,green=r, orange=o,purple=u,white=w,yellow=y
21. population: abundant=a,clustered=c,numerous=n, scattered=s,several=v,solitary=y
22. habitat: grasses=g,leaves=l,meadows=m,paths=p, urban=u,waste=w,woods=d
"""

from sklearn.metrics import accuracy_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('./data/mushrooms.csv')
print(df.head())
sns.countplot(data=df, x='class')

unique_d = df.describe().transpose().reset_index().sort_values('unique')
unique_d

sns.barplot(data=unique_d, x='index', y='unique')
# adding semicolone end of the line hides warning or text
plt.xticks(rotation=90)

X = pd.get_dummies(df.drop('class', axis=1), drop_first=True)
y = df['class']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# n_estimators param indicates number of stump
model = AdaBoostClassifier(n_estimators=15)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy_score(y_test, y_pred)

errors = []
for n in range(1, 50):
    amodel = AdaBoostClassifier(n_estimators=n)
    amodel.fit(X_train, y_train)
    predicion = amodel.predict(X_test)
    err = 1 - accuracy_score(y_test, predicion)
    errors.append(err)

plt.plot(errors)

# it represents an array of number ranging from 0 to 1 as model thinks which feature is importan and which not.most important feature is 1
model.feature_importances_

imprtant_features = pd.DataFrame(
    data=model.feature_importances_, index=X.columns, columns=['Importance'])
imprtant_features_final = imprtant_features[imprtant_features['Importance'] > 0]
imprtant_features_final

# to see which feature is the model thinks most important to identify poisonous vs ediable
model.feature_importances_.argmax()

important_features = X.columns[model.feature_importances_ > 0.1]
important_features

X.columns[28]

"""It means if you had to check one thing to identify if the mushroom is ediable you can go with 'gill-size' or 'odor',and it says naroow gills are usually poisonous"""

sns.countplot(data=df, x='gill-size', hue='class')

sns.countplot(data=df, x='odor', hue='class')