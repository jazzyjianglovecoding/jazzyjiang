
# coding: utf-8

# # Hiscox Project: Clustering Exploration

# In[1]:

# Some methods to show polt in notebook:
get_ipython().magic(u'matplotlib inline')
# Import libraries to be used
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# In[5]:

dataframe= pd.read_csv('Featureselectionbinary.csv')
# Using panda to read the csv file


# In[6]:

ml_dataset = dataframe[[u'StAssets',u'NormalisedPERatio',  u'StIPOPrice', ]]


# In[7]:

# train dataset will be the one on which we will apply ml technics
train = ml_dataset.copy()


# In[8]:

from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans,MiniBatchKMeans,SpectralClustering,AgglomerativeClustering
cluster_range = range( 1, 20 )
cluster_errors1 = []
for num_clusters in cluster_range:
  clusters1 = KMeans( num_clusters )
  clusters1.fit(train)
  cluster_errors1.append( clusters1.inertia_ )
clusters_df1 = pd.DataFrame( { "num_clusters":cluster_range, "cluster_errors1": cluster_errors1 } )
cluster_errors2 = []
for num_clusters in cluster_range:
  clusters2 = MiniBatchKMeans( num_clusters )
  clusters2.fit(train)
  cluster_errors2.append( clusters2.inertia_ )
clusters_df2 = pd.DataFrame( { "num_clusters":cluster_range, "cluster_errors2": cluster_errors2 } )


# In[9]:

plt.figure(figsize=(12,6))
plt.plot( clusters_df1.num_clusters, clusters_df1.cluster_errors1, marker = "o" ,color='green')
plt.plot( clusters_df2.num_clusters, clusters_df2.cluster_errors2, marker = "o" ,color='blue')
plt.xlabel('number of clusters')
plt.ylabel('cluster errors')
plt.title('Elbow Analysis')
plt.legend(loc="upper right")


# In[10]:

clustering_model = KMeans(n_clusters=2)


# In[11]:

time clusters = clustering_model.fit_predict(train)


# In[12]:

print(clustering_model.inertia_)


# In[13]:

from sklearn.metrics import silhouette_score
silhouette = silhouette_score(train.values, clusters, metric='euclidean', sample_size=2000)
print ("Silhouette score :", silhouette)


# In[14]:

final = train.join(pd.Series(clusters, index=train.index, name='cluster'))
final['cluster'] = final['cluster'].map(lambda cluster_id: 'cluster' + str(cluster_id))


# In[15]:

size = pd.DataFrame({'size': final['cluster'].value_counts()})
size.head()


# In[21]:

axis_x = final.columns[2]   # change me
axis_y = final.columns[0]  # change me
from ggplot import ggplot, aes, geom_point
ggplot(aes(axis_x, axis_y, colour='cluster'), final) + geom_point()


# In[27]:

from sklearn.decomposition import PCA as sklearnPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
X = final.iloc[:, 0:3] # Split off features
pca = sklearnPCA(n_components=2) #2-dimensional PCA
transformed = pd.DataFrame(pca.fit_transform(X))
transformed.columns = ['axisx', 'axisy']


# In[28]:

Cluster=final['cluster']
transformed['Cluster'] = pd.Series(Cluster, index=transformed.index)


# In[29]:

axis_x = transformed.columns[0]   # change me
axis_y = transformed.columns[1]  # change me
ggplot(aes(axis_x, axis_y, colour='Cluster'), transformed) + geom_point()


# In[30]:

y=clustering_model.labels_
y=pd.DataFrame(data=y)
y.columns=['Class']
# Select features to include in the plot
plot_feat = [ u'StAssets',u'NormalisedPERatio',  u'StIPOPrice',]
# Concat classes with the normalized data
data_norm = pd.concat([X[plot_feat], y], axis=1)
# Perform parallel coordinate plot
pd.plotting.parallel_coordinates(data_norm,'Class')
plt.show()


# In[31]:

#Initializes plotting library and functions for 3D scatter plots 
from pyspark.ml.feature import VectorAssembler
from sklearn.datasets import make_blobs
from sklearn.datasets import make_gaussian_quantiles
from sklearn.externals import six
import pandas as pd
import numpy as np
import argparse
import json
import re
import os
import sys
import plotly
import plotly.graph_objs as go
plotly.offline.init_notebook_mode()

def rename_columns(train, prefix='x'):
    """
    Rename the columns of a dataframe to have X in front of them

    :param df: data frame we're operating on
    :param prefix: the prefix string
    """
    df = train.copy()
    return df


# In[32]:

# create an artificial dataset with 3 clusters
Y=clustering_model.labels_
X=train.values
df = pd.DataFrame(X)
# and add the Y
df['y'] = Y
# split df into cluster groups
grouped = df.groupby(['y'], sort=True)
# compute sums for every column in every group
mean = grouped.mean()


# In[33]:

data = [go.Heatmap(z=mean.values.tolist(), 
                   y=['Cluster 1', 'Cluster 2'],
                   x=[u'StNumberofYearstoDimissal',u'StRevenue',  u'StMCapNoInsiderHoldings', 
                        u'StGoogleHits',u'StClassPeriodLengthdays'
                     ],
                   colorscale='Viridis')]
plotly.offline.iplot(data, filename='pandas-heatmap')


# In[34]:

y=clustering_model.labels_
y=pd.DataFrame(data=y)
dataframe['Class'] = y
dataframe.to_csv('Featureselectionbinarycluster.csv')

