# K-means Clustering Exercise: Wine Dataset Analysis
# ===================================================

"""
This exercise explores K-means clustering on the Wine dataset.
Students will apply K-means clustering, evaluate results, and explore
the impact of different parameters and preprocessing steps.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Part 1: Load and explore the dataset
# ------------------------------------

# Load the Wine dataset
wine = load_wine()
X = wine.data
y = wine.target

# Create a DataFrame for easier exploration
df = pd.DataFrame(X, columns=wine.feature_names)
df['target'] = y

# TASK 1: Explore the dataset
# - Print basic information about the dataset
print("Dataset shape:", X.shape)
print("Number of classes:", len(np.unique(y)))
print("Features:", wine.feature_names)

# - Examine basic statistics
print("\nFeature statistics:")
print(df.describe())

# - Check class distribution
print("\nClass distribution:")
print(df['target'].value_counts())

# - Visualize feature distributions
plt.figure(figsize=(15, 10))
for i, feature in enumerate(wine.feature_names):
    plt.subplot(4, 4, i+1)
    for target in range(3):
        plt.hist(df[df['target'] == target][feature], alpha=0.5, label=f'Class {target}')
    plt.title(feature)
    plt.legend()
    plt.tight_layout()
plt.savefig('wine_feature_distributions.png')
plt.close()

# Part 2: Prepare the data
# -----------------------

# TASK 2: Scale the features
# - Implement standardization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# - Perform PCA for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# - Visualize the data in 2D
plt.figure(figsize=(10, 8))
for target in range(3):
    plt.scatter(X_pca[y == target, 0], X_pca[y == target, 1],
                label=f'Class {target}', alpha=0.7, edgecolors='w', s=60)
plt.title('PCA of Wine Dataset (True Labels)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('wine_pca_true_labels.png')
plt.close()

# Part 3: Apply K-means clustering
# -------------------------------

# TASK 3: Implement K-means with different numbers of clusters
# - Try a range of k values
ks = range(2, 10)
inertias = []
silhouette_scores = []

for k in ks:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)

    # Only calculate silhouette score for k > 1
    if k > 1:
        silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))

# - Plot the elbow curve
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(ks, inertias, 'o-', markersize=8)
plt.title('Elbow Method')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(ks[1:], silhouette_scores, 'o-', markersize=8)
plt.title('Silhouette Score Method')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Silhouette Score')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('wine_kmeans_metrics.png')
plt.close()

# - Based on the plots, choose the optimal k
optimal_k = 3  # This should be determined by the students

# TASK 4: Apply K-means with the optimal k
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(X_scaled)

# - Visualize clustering results with PCA
plt.figure(figsize=(10, 8))
for cluster in range(optimal_k):
    plt.scatter(X_pca[cluster_labels == cluster, 0], X_pca[cluster_labels == cluster, 1],
                label=f'Cluster {cluster}', alpha=0.7, edgecolors='w', s=60)
plt.title(f'K-means Clustering of Wine Dataset (k={optimal_k})')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('wine_kmeans_clusters.png')
plt.close()

# Part 4: Evaluate the clustering results
# --------------------------------------

# TASK 5: Compare clustering results with true labels
# - Calculate the adjusted Rand index
ari = adjusted_rand_score(y, cluster_labels)
print(f"\nAdjusted Rand Index: {ari:.4f}")

# - Create a confusion matrix
conf_matrix = confusion_matrix(y, cluster_labels)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=[f'Cluster {i}' for i in range(optimal_k)],
            yticklabels=[f'Class {i}' for i in range(3)])
plt.title('Confusion Matrix: True Classes vs. Clusters')
plt.ylabel('True Class')
plt.xlabel('Assigned Cluster')
plt.savefig('wine_confusion_matrix.png')
plt.close()

# TASK 6: Examine cluster centers
# - Convert cluster centers back to original feature scale
centers = scaler.inverse_transform(kmeans.cluster_centers_)
centers_df = pd.DataFrame(centers, columns=wine.feature_names)
centers_df.index = [f'Cluster {i}' for i in range(optimal_k)]

# - Visualize feature importance for each cluster
plt.figure(figsize=(15, 10))
sns.heatmap(centers_df, cmap='coolwarm', annot=True, fmt='.2f')
plt.title('Feature Values at Cluster Centers')
plt.savefig('wine_cluster_centers.png')
plt.close()

# Part 5: Challenge tasks for students
# ----------------------------------

"""
TASK 7: Experiment with different preprocessing techniques
- Try different scaling methods (MinMaxScaler, RobustScaler)
- Try different dimensionality reduction techniques (t-SNE, UMAP)
- How do these affect your clustering results?

TASK 8: Experiment with different K-means parameters
- Try different initialization methods
- Try different numbers of initializations
- How do these affect your clustering results?

TASK 9: Try other clustering algorithms
- Implement hierarchical clustering or DBSCAN
- Compare the results with K-means
- Which algorithm better captures the structure of the wine dataset?

TASK 10: Feature selection impact
- Choose subsets of features based on domain knowledge
- Run clustering on reduced feature sets
- Analyze which features are most important for clustering
"""

# Extended visualization: Feature importance by cluster
# ----------------------------------------------------

# One approach: Radar chart of cluster centers
def radar_chart(centers_df):
    # Number of variables
    categories = centers_df.columns
    N = len(categories)

    # Create a figure
    fig = plt.figure(figsize=(10, 10))

    # Create angles for each feature
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the loop

    # Create the plot
    ax = plt.subplot(111, polar=True)

    # Draw one axis per variable and add labels
    plt.xticks(angles[:-1], categories, size=12)

    # Draw the chart for each cluster
    for i, cluster in enumerate(centers_df.index):
        values = centers_df.loc[cluster].values.tolist()
        values += values[:1]  # Close the loop
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=cluster)
        ax.fill(angles, values, alpha=0.1)

    # Add legend
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.title('Cluster Centers Profile', size=15)
    plt.savefig('wine_radar_chart.png')
    plt.close()

radar_chart(centers_df)

print("\nExercise completed. Check the generated visualizations for analysis.")