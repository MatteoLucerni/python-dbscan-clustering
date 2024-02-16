import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets._samples_generator import make_moons
from sklearn.cluster import AgglomerativeClustering, KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.model_selection import ParameterGrid

plt.rcParams["figure.figsize"] = (12, 8)
sns.set_theme()

X, _ = make_moons(n_samples=500, noise=0.1)
plt.scatter(X[:, 0], X[:, 1])
plt.show()

# KMeans
km = KMeans(
    n_clusters=2,
)

plt.figure()
pred = km.fit_predict(X)
plt.scatter(X[:, 0], X[:, 1], c=pred, cmap="viridis")
plt.show()

silhouette = silhouette_score(X, pred)
print("Silhouette score:", silhouette)


# Agglomerative
agg = AgglomerativeClustering(n_clusters=2, linkage="ward")

plt.figure()
pred = agg.fit_predict(X)
plt.scatter(X[:, 0], X[:, 1], c=pred, cmap="viridis")
plt.show()

silhouette = silhouette_score(X, pred)
print("Silhouette score:", silhouette)

# DBSCAN
db = DBSCAN(eps=0.2, min_samples=10)

plt.figure()
pred = db.fit_predict(X)
plt.scatter(X[:, 0], X[:, 1], c=pred, cmap="viridis")
plt.show()

silhouette = silhouette_score(X, pred)
print("Silhouette score:", silhouette)

# Parameter Search fro DBScan
param_grid = {
    "eps": [0.05, 0.1, 0.15, 0.2, 0.25, 0.3],
    "min_samples": [3, 5, 7, 10],
}

grid = ParameterGrid(param_grid)

best_score = -1
best_params = None

for params in grid:
    model = DBSCAN(eps=params["eps"], min_samples=params["min_samples"])
    pred = model.fit_predict(X)

    if len(set(pred)) - (1 if -1 in pred else 0) > 1:
        score = silhouette_score(X, pred)
        if score > best_score:
            best_score = score
            best_params = params

print(f"Best Silhouette: {best_score}")
print(f"best params: {best_params}")

db = DBSCAN(eps=best_params["eps"], min_samples=best_params["min_samples"])

plt.figure()
pred = db.fit_predict(X)
plt.scatter(X[:, 0], X[:, 1], c=pred, cmap="viridis")
plt.show()
