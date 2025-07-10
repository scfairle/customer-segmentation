
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Load data
df = pd.read_csv("customer_segmentation_dataset.csv")

# Select features
features = df[["Age", "Annual Income (k$)", "Spending Score (1-100)"]]
scaler = StandardScaler()
scaled = scaler.fit_transform(features)

# PCA for visualization
pca = PCA(n_components=2)
pca_features = pca.fit_transform(scaled)

# KMeans clustering
kmeans = KMeans(n_clusters=3, random_state=42)
df["Cluster"] = kmeans.fit_predict(scaled)
df["PCA1"] = pca_features[:, 0]
df["PCA2"] = pca_features[:, 1]

# Visualize clusters
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x="PCA1", y="PCA2", hue="Cluster", palette="Set2", s=100, alpha=0.8)
plt.title("Customer Segments (KMeans + PCA)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend(title="Cluster")
plt.tight_layout()
plt.savefig("customer_clusters.png")

# Export cluster profiles
profiles = df.groupby("Cluster")[["Age", "Annual Income (k$)", "Spending Score (1-100)"]].mean().round(1)
profiles["Num Customers"] = df["Cluster"].value_counts().sort_index().values
profiles.to_csv("cluster_profiles.csv")
