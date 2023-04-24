# Featuring the data like finding outliers,relation among them,null value handle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display


df = pd.read_csv('.\data\Ames_Housing_Data.csv')
# display(df.head())

# display(df.corr()['SalePrice'])
sns.scatterplot(x='Overall Qual', y='SalePrice', data=df)
# plt.show()

# print(df[(df['Overall Qual'] > 8) & (df[('SalePrice')] < 200000)].index)
outliers_removed_data = df[(df['Overall Qual'] > 8)
                           & (df[('SalePrice')] < 200000)].index
df = df.drop(outliers_removed_data, axis=0)

sns.scatterplot(x='Overall Qual', y='SalePrice', data=df)
# plt.show()


def create_age(mu=50, sigma=15, n_samples=100, seed=42):
    np.random.seed(seed)
    sample_age = np.random.normal(loc=mu, scale=sigma, size=n_samples)
    sample_age = np.round(sample_age, decimals=0)
    return sample_age


# print(create_age())
sample = create_age()

sns.boxplot(data=sample)
# plt.show()

ser = pd.Series(sample)
# print(ser)
# print(ser.describe())

# finding the outlier
# print(np.percentile(sample, [75, 25]))
IQR = 56.25 - 41.00
lower_lim = 42 - 1.5 * (IQR)
higher_lim = 56.25 + 1.5 * (IQR)
# print(lower_lim)
# print(ser[ser < lower_lim])
