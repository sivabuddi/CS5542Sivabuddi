# importing required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


# reading the data and looking at the first five rows of the data
data=pd.read_csv("Wholesale customers data.csv")
print(data.head())

# statistics of the data
print(data.describe())
data=data.iloc[:,[3,5]].values

# standardizing the data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# statistics of scaled data
data_s=pd.DataFrame(data_scaled).describe()
print(data_s)

# defining the kmeans function with initialization as k-means++
#kmeans = KMeans(n_clusters=2, init='k-means++')

# fitting the k means algorithm on scaled data
#kmeans.fit(data_scaled)

# inertia on the fitted data
#print(kmeans.inertia_)


# fitting multiple k-means algorithms and storing the values in an empty list
SSE = []
for cluster in range(1,20):
    kmeans = KMeans(n_jobs = -1, n_clusters = cluster, init='k-means++')
    kmeans.fit(data_scaled)
    SSE.append(kmeans.inertia_)

# converting the results into a dataframe and plotting them
frame = pd.DataFrame({'Cluster':range(1,20), 'SSE':SSE})
plt.figure(figsize=(12,6))
plt.plot(frame['Cluster'], frame['SSE'], marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.title("Elbow Curve")
plt.show()

#k means using 5 clusters and k-means++ initialization
kmeans = KMeans(n_clusters = 6,init ='k-means++', max_iter=300, n_init=10,random_state=0 )
kmeans.fit(data_scaled)
pred = kmeans.predict(data_scaled)


pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', -1)

frame = pd.DataFrame(data_scaled)
frame['cluster'] = pred
print(frame)
result = frame['cluster'].value_counts()
print(result)


#6 Visualising the clusters
plt.scatter(data_scaled[frame['cluster']==0, 0], data_scaled[frame['cluster']==0, 1], s=100, c='red', label ='Cluster 1')
plt.scatter(data_scaled[frame['cluster']==1, 0], data_scaled[frame['cluster']==1, 1], s=100, c='blue', label ='Cluster 2')
plt.scatter(data_scaled[frame['cluster']==2, 0], data_scaled[frame['cluster']==2, 1], s=100, c='green', label ='Cluster 3')
plt.scatter(data_scaled[frame['cluster']==3, 0], data_scaled[frame['cluster']==3, 1], s=100, c='cyan', label ='Cluster 4')
plt.scatter(data_scaled[frame['cluster']==4, 0], data_scaled[frame['cluster']==4, 1], s=100, c='magenta', label ='Cluster 5')
plt.scatter(data_scaled[frame['cluster']==5, 0], data_scaled[frame['cluster']==5, 1], s=100, c='black', label ='Cluster 6')
#Plot the centroid. This time we're going to use the cluster centres  #attribute that returns here the coordinates of the centroid.
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='yellow', label = 'Centroids')
plt.title('Clusters of Customers')
plt.xlabel('Milk)')
plt.ylabel('Frozen')
plt.legend(loc='upper left')
plt.show()


# import matplotlib.pyplot as plt
# import seaborn as sns; sns.set()  # for plot styling
# import numpy as np
#
#
# plt.scatter(data_scaled[:, 0], data_scaled[:, 1],c=pred, s=50, cmap='viridis')
#
# centers = kmeans.cluster_centers_
# plt.scatter(centers[:, 0], centers[:, 1],c='black', s=440, alpha=0.5)
# plt.show()

'''
# Initialize plotting library and functions for 3D scatter plots
from sklearn.datasets import make_blobs
from sklearn.datasets import make_gaussian_quantiles
from sklearn.datasets import make_classification, make_regression
#from sklearn.externals import six
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

import six
import sys
sys.modules['sklearn.externals.six'] = six
#import mlrose

 

cluster1=frame.loc[frame['cluster'] == 0]
cluster2=frame.loc[frame['cluster'] == 1]
cluster3=frame.loc[frame['cluster'] == 2]

scatter1 = dict(
    mode = "markers",
    name = "Cluster 1",
    type = "scatter3d",
    x = cluster1.as_matrix()[:,0], y = cluster1.as_matrix()[:,1], z = cluster1.as_matrix()[:,2],
    marker = dict( size=2, color='green')
)
scatter2 = dict(
    mode = "markers",
    name = "Cluster 2",
    type = "scatter3d",
    x = cluster2.as_matrix()[:,0], y = cluster2.as_matrix()[:,1], z = cluster2.as_matrix()[:,2],
    marker = dict( size=2, color='blue')
)
scatter3 = dict(
    mode = "markers",
    name = "Cluster 3",
    type = "scatter3d",
    x = cluster3.as_matrix()[:,0], y = cluster3.as_matrix()[:,1], z = cluster3.as_matrix()[:,2],
    marker = dict( size=2, color='red')
)
cluster1 = dict(
    alphahull = 5,
    name = "Cluster 1",
    opacity = .1,
    type = "mesh3d",
    x = cluster1.as_matrix()[:,0], y = cluster1.as_matrix()[:,1], z = cluster1.as_matrix()[:,2],
    color='green', showscale = True
)

cluster2 = dict(
    alphahull = 5,
    name = "Cluster 2",
    opacity = .1,
    type = "mesh3d",
    x = cluster2.as_matrix()[:,0], y = cluster2.as_matrix()[:,1], z = cluster2.as_matrix()[:,2],
    color='blue', showscale = True
)
cluster3 = dict(
    alphahull = 5,
    name = "Cluster 3",
    opacity = .1,
    type = "mesh3d",
    x = cluster3.as_matrix()[:,0], y = cluster3.as_matrix()[:,1], z = cluster3.as_matrix()[:,2],
    color='red', showscale = True
)
layout = dict(
    title = 'Interactive Cluster Shapes in 3D',
    scene = dict(
        xaxis = dict( zeroline=True ),
        yaxis = dict( zeroline=True ),
        zaxis = dict( zeroline=True ),
    )
)
fig = dict( data=[scatter1, scatter2, scatter3, cluster1, cluster2, cluster3], layout=layout )
# Use py.iplot() for IPython notebook
plotly.offline.iplot(fig, filename='mesh3d_sample')

'''