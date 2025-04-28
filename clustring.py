import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.decomposition import PCA

data = load_iris()
X = data.data
y_true = data.target

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

def evaluate_and_plot(labels, title):
    print(f"\n--- {title} ---")
    print("Silhouette Score:", silhouette_score(X_scaled, labels))
    print("Calinski-Harabasz Score:", calinski_harabasz_score(X_scaled, labels))
    print("Davies-Bouldin Score:", davies_bouldin_score(X_scaled, labels))
    print("Adjusted Rand Index:", adjusted_rand_score(y_true, labels))
    print("Normalized Mutual Info:", normalized_mutual_info_score(y_true, labels))

    plt.figure(figsize=(6, 4))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', s=50)
    plt.title(f"{title} (PCA Projection)")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.colorbar()
    plt.show()

kmeans = KMeans(n_clusters=3, random_state=42)
kmeans_labels = kmeans.fit_predict(X_scaled)
evaluate_and_plot(kmeans_labels, "KMeans Clustering")

agglo = AgglomerativeClustering(n_clusters=3)
agglo_labels = agglo.fit_predict(X_scaled)
evaluate_and_plot(agglo_labels, "Agglomerative Clustering")

dbscan = DBSCAN(eps=0.6, min_samples=5)
dbscan_labels = dbscan.fit_predict(X_scaled)
if len(set(dbscan_labels)) > 1:
    evaluate_and_plot(dbscan_labels, "DBSCAN Clustering")
else:
    print("\n--- DBSCAN Clustering ---")
    print("DBSCAN could not form multiple clusters with given parameters.")
