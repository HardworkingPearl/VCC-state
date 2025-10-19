import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.datasets import load_iris

# === 1) Basic PCA + inverse_transform ===
X = load_iris().data  # shape (150, 4)
breakpoint()
pca = PCA(n_components=2, random_state=0)
Z = pca.fit_transform(X)               # project to 2D     (150, 2)
X_rec = pca.inverse_transform(Z)       # reconstruct to 4D (150, 4)