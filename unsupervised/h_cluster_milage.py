

from scipy.cluster import hierarchy
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('/data/cluster_mpg.csv')
df

df.describe()

dum_data = pd.get_dummies(df.drop('name', axis=1), drop_first=True)
dum_data

# for hierarchical clustering its better to choose MinMaxScaler as it makes distance of data points between 0 to 1,so it becomes easy to make cluster
scale = MinMaxScaler()
scaled_data = scale.fit_transform(dum_data)
scaled_data

scaled_df = pd.DataFrame(scaled_data, columns=dum_data.columns)
scaled_df

sns.heatmap(scaled_df)

# since each row is a data point and colum are feature we dont want to see the relation between feature but
#  between row or datapoints. thats the actual goal so we set 'col_cluster = False'
sns.clustermap(scaled_df, col_cluster=False)

"""choosing cluster value beforehand"""

model = AgglomerativeClustering(n_clusters=4)
cluster_lebels = model.fit_predict(scaled_df)
cluster_lebels

sns.scatterplot(data=df, x='mpg', y='weight',
                hue=cluster_lebels, palette='viridis')

# setting distance threshold = 0,meaning every point is a cluster at first
model2 = AgglomerativeClustering(n_clusters=None, distance_threshold=0)
cluster = model2.fit_predict(scaled_df)
cluster

linkage_matrix = hierarchy.linkage(model2.children_)
linkage_matrix

plt.figure(figsize=(20, 10), dpi=150)
hierarchy.dendrogram(linkage_matrix, truncate_mode='lastp', p=20)

# theoritical max distance of datapoints
np.sqrt(len(scaled_df.columns))

# as max distance is 3,so its better to choose \distance_threshold = 2
model3 = AgglomerativeClustering(n_clusters=None, distance_threshold=2.5)
cluster2 = model3.fit_predict(scaled_df)
cluster2
