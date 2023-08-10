import numpy as np
import pandas as pd

df = pd.read_excel('/Users/shoroukeladawy/Desktop/Review_ratings.xlsx')
X = df.iloc[:, 1:].values
#print(X)


def kmeans(X, k, max_iters=100):

    centroids = X[np.random.choice(X.shape[0], k, replace=False), :]
    clusters = np.zeros(X.shape[0])

    for i in range(max_iters):
        distances = np.sqrt(np.sum((X - centroids[:, np.newaxis])**2, axis=2))
        clusters = np.argmin(distances, axis=0)

        for j in range(k):
            centroids[j] = np.mean(X[clusters == j], axis=0)

    return centroids, clusters


k = int(input("Enter the number of clusters: "))
centroids, clusters = kmeans(X, k)

user_names = df.iloc[:, 0].values

cluster_groups = {}
for i in range(k):
    users = user_names[clusters == i]
    cluster_groups[f'Cluster {i+1} ({len(users)} users)'] = users

for cluster, users in cluster_groups.items():
    print(f"{cluster}: {', '.join(users)}")


def detect_outliers(X, centroids, clusters, threshold):

    outliers = []

    for i, centroid in enumerate(centroids):
        indices = np.where(clusters == i)[0]
        distances = np.sqrt(np.sum((X[indices] - centroid)**2, axis=1))
        cluster_outliers = indices[distances > threshold]
        outliers.extend(cluster_outliers)

    user_names = df.iloc[:, 0].values
    outlier_names = user_names[outliers]

    return outlier_names

threshold = float(input("Enter the maximum distance from the centroid for outliers: "))
outliers = detect_outliers(X, centroids, clusters, threshold)
if len(outliers) > 0:
    print(f"Outliers: {', '.join(outliers)}", "count:", len(outliers))
else:
    print("No outliers detected.")
