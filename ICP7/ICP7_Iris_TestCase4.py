import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
sns.set(style="white", color_codes=True)
import warnings
warnings.filterwarnings("ignore")

dataset = pd.read_csv('Iris.csv')
print("Original Data size=",dataset.shape)
# print(dataset.describe())
print(dataset.columns)
x = dataset.iloc[:,3:5]
y = dataset.iloc[:,-1]
#print(x)
# see how many samples we have of each species
print(dataset["Species"].value_counts())


##elbow method to know the number of clusters
wcss = []

for i in range(1,11):
    kmeans = KMeans(n_clusters=i,init='k-means++',max_iter=300,n_init=10,random_state=0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)

plt.plot(range(1,11),wcss)
plt.title('Elbow Curve for Iris Dataset')
plt.xlabel('Number of Clusters')
plt.ylabel('Wcss')
plt.show()


# Set style of scatterplot
sns.set_context("notebook", font_scale=1.1)
sns.set_style("ticks")

# Create scatterplot of dataframe
sns.lmplot('PetalLengthCm', # Horizontal axis
           'PetalWidthCm', # Vertical axis
           data=dataset, # Data source
           fit_reg=False, # Don't fix a regression line
           hue="Species", # Set color
           scatter_kws={"marker": "D", # Set marker style
                        "s": 100}) # S marker size

# Set title
plt.title('Iris Dataset')

# Set x-axis label
plt.xlabel('PetalLengthCm')

# Set y-axis label
plt.ylabel('PetalWidthCm')
plt.show()



# sns.FacetGrid(dataset, hue="Species", size=4).map(plt.scatter, "SepalLengthCm", "SepalWidthCm").add_legend()
# sns.lmplot(x, y, data=x, hue="hue")

# # do same for petals
# sns.FacetGrid(dataset, hue="Species", size=4).map(plt.scatter, "PetalLengthCm", "PetalWidthCm").add_legend()
# #
# sns.FacetGrid(dataset, hue="Species", size=4).map(plt.scatter, "SepalLengthCm", "PetalLengthCm").add_legend()
# plt.show()

from sklearn import preprocessing
scaler = preprocessing.StandardScaler()
scaler.fit(x)
X_scaled_array = scaler.fit_transform(x)
X_scaled = pd.DataFrame(X_scaled_array, columns = x.columns)

from sklearn.cluster import KMeans
nclusters = 3 # this is the k in kmeans
km = KMeans(n_clusters=nclusters)
km.fit(X_scaled_array)
pred = km.predict(X_scaled_array)
X_scaled['clusters']=pred
result = X_scaled['clusters'].value_counts()
print(result)
print(X_scaled['clusters'])

#6 Visualising the clusters
plt.scatter(X_scaled_array[pred==0, 0], X_scaled_array[pred==0, 1], s=100, c='red', label ='Cluster 1')
plt.scatter(X_scaled_array[pred==1, 0], X_scaled_array[pred==1, 1], s=100, c='blue', label ='Cluster 2')
plt.scatter(X_scaled_array[pred==2, 0], X_scaled_array[pred==2, 1], s=100, c='green', label ='Cluster 3')
# plt.scatter(X_scaled_array[pred==3, 0], X_scaled_array[pred==3, 1], s=100, c='black', label ='Cluster 4')
# plt.scatter(X_scaled_array[pred==4, 0], X_scaled_array[pred==4, 1], s=100, c='brown', label ='Cluster 5')
# plt.scatter(X_scaled_array[pred==5, 0], X_scaled_array[pred==5, 1], s=100, c='orange', label ='Cluster 6')

plt.legend()
plt.title("Predicated Clusters ")
plt.xlabel("PetalLengthCm")
plt.ylabel("PetalWidthCm")rm
plt.show()


# predict the cluster for each data point
# y_cluster_kmeans = km.predict(x)
from sklearn import metrics
score = metrics.silhouette_score(X_scaled_array, pred)
print("Silhouette Score={}".format(score))



