# Handaling missing data, finding outliers,relation among them,null value handle

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


# ..........................Missing Data handling...........
# with open('./data/Ames_Housing_Feature_Description.txt', 'r') as f:
# print(f.read())
# print(df.head())
# print(df.isnull().sum())


#  taking only column that has a missing data
def missing_data(df):
    parcent_nan = 100 * df.isnull().sum() / len(df)
    parcent_nan = parcent_nan[parcent_nan > 0].sort_values()
    return parcent_nan


parcent_nan = missing_data(df)
# print(parcent_nan)
plt.figure(figsize=(6, 8), dpi=150)
sns.barplot(x=parcent_nan.index, y=parcent_nan)
plt.xticks(rotation=90)
# threshold 1% missing columns ,thats why ylim (0,1)
plt.ylim(0, 1)
# plt.show()

# print(parcent_nan[parcent_nan < 1])
# removing that 1 row which has a null value in electrical and garage area column
df = df.dropna(axis=0, subset=['Electrical', 'Garage Area'])
parcent_nan = missing_data(df)
# print(parcent_nan)

# filling missing value in 'Lot Frontage' with mean value usuing transform method
xx = df.groupby('Neighborhood')['Lot Frontage'].transform(
    lambda val: val.fillna(val.mean()))

# print(xx)
df['MS SubClass'] = df['MS SubClass'].apply(str)
# print(df['MS SubClass'])
object_df = df.select_dtypes(include='object')
numeric_df = df.select_dtypes(exclude='object')
dummies_df = pd.get_dummies(object_df, drop_first=True)

final_df = pd.concat([numeric_df, dummies_df], axis=1)
print(final_df)


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
