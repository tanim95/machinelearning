# -*- coding: utf-8 -*-
"""pca_tumor_analysis.ipynb

Original file is located at
    https://colab.research.google.com/drive/1AgWl0BROLkcK3HFEXTBH1Iwc4ebJU33x

#Breast cancer wisconsin (diagnostic) dataset
--------------------------------------------

**Data Set Characteristics:**

    :Number of Instances: 569

    :Number of Attributes: 30 numeric, predictive attributes and the class

    :Attribute Information:
        - radius (mean of distances from center to points on the perimeter)
        - texture (standard deviation of gray-scale values)
        - perimeter
        - area
        - smoothness (local variation in radius lengths)
        - compactness (perimeter^2 / area - 1.0)
        - concavity (severity of concave portions of the contour)
        - concave points (number of concave portions of the contour)
        - symmetry
        - fractal dimension ("coastline approximation" - 1)

        The mean, standard error, and "worst" or largest (mean of the three
        worst/largest values) of these features were computed for each image,
        resulting in 30 features.  For instance, field 0 is Mean Radius, field
        10 is Radius SE, field 20 is Worst Radius.

        - class:
                - WDBC-Malignant
                - WDBC-Benign
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('/data/cancer_tumor_data.csv')
df

df.drop(['id', 'Unnamed: 32'], axis=1, inplace=True)
df = pd.get_dummies(df, drop_first=True)
df

df.describe()

scale = StandardScaler()
scaled_X = scale.fit_transform(df)

# sns.heatmap(df.corr())

# here rowvar = false,meanign row varience is false as we just want to know the varience accoross the column
covarience_matrix = np.cov(scaled_X, rowvar=False)
covarience_matrix.shape

eigen_values, eigen_vectors = np.linalg.eig(covarience_matrix)

num_pc = 2
np.argsort(eigen_values)
# here the first member in the result array is 20,which is the index of lowest eigen value so we need to sort it,
# as argsort gives as the order of index to arrange value in ascending value

# [::-1] this line is for reversing value,a short cut
np.argsort(eigen_values)[::-1]

sorted_key = np.argsort(eigen_values)[::-1][:num_pc]

# now lets choose max eigen value and vector again form those previous multiple value
eigen_values, eigen_vectors = eigen_values[sorted_key], eigen_vectors[:, sorted_key]

# now lets project the original data to eigen vectors which will be our Principle component,we use vectors dot product for projection
principle_component = np.dot(scaled_X, eigen_vectors)
principle_component

plt.figure(figsize=(8, 6), dpi=150)
plt.scatter(principle_component[:, 0], principle_component[:, 1])

cancer_dictionary = load_breast_cancer()
cancer_dictionary.keys()

print(cancer_dictionary['DESCR'])

# as our data set is a part of the "Wisconsin (Diagnostic) datasets" dataset ,so now we test
# our principle componet based on the target column of the Wisconsin data
plt.figure(figsize=(8, 6), dpi=150)
plt.scatter(principle_component[:, 0],
            principle_component[:, 1], c=cancer_dictionary['target'])

"""so we reduced the dimension from 31 to 2 and still we can separate the tumor(Benign,Malignant) as shown in the figure.meaning almost all the information of acutall dataset were keept in ort two principle component"""
