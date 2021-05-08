import pandas as pd
import numpy as np
import os
import scipy.stats as stats
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

``` Basic data analysis template with k-means clustering ```
### load raw data into a pandas dataframe
print("\n*** Reading the raw data ***\n")
metadata = pd.read_csv("folder/data.csv")
print("row, col = ", metadata.shape)

### removing outliers with 75% quantile to get better result
z_scores = stats.zscore(metadata)
abs_z_scores = np.abs(z_scores)
filtered_entries = (abs_z_scores < 3).all(axis=1)
new_df = metadata[filtered_entries]
print("Cleaned data: ", new_df.shape)

### randomly picking 'n' samples from the dataset to cluster and analyse
""" We can also input our desired value in the place of 'n' for testing purposes """
sample_df = new_df.sample(n = 1000)
# to drop unnecessary data
test_df = sample_df.drop(['title'], axis=1)
print("\n*** Sampled data:", test_df.shape)

### compute silhouette score for different clusters with k-means
silhouette_coefficients = []
for k in range(3,11,2):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(test_df.values)
    score = silhouette_score(test_df.values, kmeans.labels_)
    silhouette_coefficients.append(score)

# plotting the silhouette index to find the best clustering number
plt.style.use("fivethirtyeight")
plt.plot(range(3,11,2), silhouette_coefficients, marker = "o")
plt.xticks(range(3,11,2))
plt.title("\nSilhouette score for each number of clusters\n")
plt.xlabel("k : Number of clusters")
plt.ylabel("Silhouette coefficient")
plt.show()

### applying PCA before clustering
test_df = test_df.reset_index(drop=True)
data = test_df
pca = PCA(2)
df = pca.fit_transform(data)
# initialize the class object
kmeans = KMeans(n_clusters=5)
# predict the labels of clusters
label = kmeans.fit_predict(df)
# getting the centroids
centroids = kmeans.cluster_centers_
u_labels = np.unique(label)

# plotting the results
for i in u_labels:
    plt.scatter(df[label == i, 0], df[label == i, 1], label = i)
plt.scatter(centroids[:,0], centroids[:,1], s = 80, color = 'black')
plt.legend()
plt.title("\nScattered plot after PCA and Clustering\n")
plt.xlabel("\nPrincipal components_1\n")
plt.ylabel("\nPrincipal components_2\n")
plt.show()

###########################################
### adding label text for clustered samples
###########################################
test_df['Cluster'] = label
mapping = {0:'text1', 1:'text2', 2:'text3', 3:'text4', 4:'text5'}
test_df = test_df.replace({'Cluster': mapping})
# dumping the labelled data to disk
if not os.path.exists('folder/output'):
    os.makedirs('folder/output')
test_df.to_csv("folder/output/clustered_data.csv", index=False)
print("\n*** Successfully saved clustered and labelled data ***\n")

########################################
#### plotting the results
########################################
# 2D scattered plotting
for i in u_labels:
    temp = test_df[test_df['Cluster']==mapping[i]]
    plt.scatter(temp['title'], temp['title'], label = mapping[i])
plt.legend()
plt.title("\title\n")
plt.xlabel("\title\n")
plt.ylabel("\title\n")
plt.show()

# 3D plot for the clustered data
X = test_df.values
fig = plt.figure(1, figsize=(7,7))
ax = Axes3D(fig, rect=[0, 0, 0.95, 1], elev=48, azim=134)
ax.scatter(X[:, 5], X[:, 4], X[:, 3], c=label.astype(np.float), edgecolor="k", s=50)
plt.title("\n3D plot after k-means, n_cluster = 5\n", fontsize=16)
ax.set_xlabel("\n\title")
ax.set_ylabel("\n\title")
ax.set_zlabel("\n\title")
plt.show()
