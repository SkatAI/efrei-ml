# Topic Modeling with BERTopic

## Setup and Imports


# Install required packages (if needed)
# !pip install bertopic sentence-transformers scikit-learn matplotlib plotly umap-learn hdbscan

# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
import plotly.express as px
from umap import UMAP
from hdbscan import HDBSCAN


## 1. Loading the Dataset


# Load the 20 Newsgroups dataset
newsgroups = fetch_20newsgroups(subset='all',
                                remove=('headers', 'footers', 'quotes'),
                                random_state=42)

# Extract documents and their categories
documents = newsgroups.data
categories = newsgroups.target
category_names = newsgroups.target_names

print(f"Number of documents: {len(documents)}")
print(f"Number of categories: {len(category_names)}")
print(f"Categories: {category_names}")

# Display a sample document
print("\nSample document:")
print(documents[0][:300] + "...")


## 2. Preprocessing


# Check if we need to do additional preprocessing
import re
from nltk.corpus import stopwords
import nltk

# Download stopwords if needed
# nltk.download('stopwords')

# We'll examine a document to see if additional preprocessing is needed
print("Original document sample:")
print(documents[0][:300] + "...")

# BERTopic doesn't necessarily require extensive preprocessing like traditional methods
# but let's filter out very short documents that might not provide meaningful content

# Filter out extremely short documents (less than 20 characters)
min_length = 20
documents_filtered = [doc for doc in documents if len(doc) > min_length]
print(f"Number of documents after filtering: {len(documents_filtered)}")

# Note: We're not doing extensive preprocessing because:
# 1. Transformer models like BERT understand context and language better than traditional models
# 2. They've been pre-trained on vast corpora with similar text structures
# 3. They handle word variations, synonyms, and context naturally
# 4. The embeddings capture semantic meaning even with noisy text


## 3. Loading the Transformer Embedding Model


# Load a pre-trained Sentence Transformer model
# This model converts text into fixed-size embeddings that capture semantic meaning
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')  # Smaller, faster model

# Alternative models with different size/performance tradeoffs:
# - 'paraphrase-MiniLM-L3-v2': Smallest/fastest, less accurate
# - 'all-MiniLM-L6-v2': Good balance of speed and quality
# - 'all-mpnet-base-v2': Higher quality, but slower and more resource-intensive

print(f"Embedding model loaded: {embedding_model}")
print(f"Embedding dimensions: {embedding_model.get_sentence_embedding_dimension()}")

# We could manually create embeddings, but BERTopic will handle this automatically
# sample_embedding = embedding_model.encode(documents[0])
# print(f"Sample embedding shape: {sample_embedding.shape}")


## 4. Creating and Training the BERTopic Model


# Initialize BERTopic with our preferred parameters
# First with default settings
topic_model = BERTopic(embedding_model=embedding_model)

# Fit the model and transform the documents
topics, probs = topic_model.fit_transform(documents_filtered)

# Get an overview of the topics
print(f"Number of topics found: {len(topic_model.get_topic_info())}")
print("\nTop topics by size:")
print(topic_model.get_topic_info().head(10))

# Check out some of the topics
print("\nExample topics and their key terms:")
for topic_id in sorted(list(set(topics)))[:5]:  # Show first 5 topics
    if topic_id != -1:  # Skip the outlier topic (-1)
        print(f"Topic {topic_id}: {topic_model.get_topic(topic_id)}")


## 5. Finding the Optimal Number of Topics


# BERTopic with HDBSCAN handles this automatically through clustering
# But we can experiment with parameters that affect the number of topics

# Let's create a function to test different parameters
def test_topic_parameters(documents, embedding_model, n_neighbors_range, min_cluster_size_range):
    results = []

    for n_neighbors in n_neighbors_range:
        for min_cluster_size in min_cluster_size_range:
            # Configure UMAP and HDBSCAN with specific parameters
            umap_model = UMAP(n_neighbors=n_neighbors, n_components=5,
                             min_dist=0.0, metric='cosine', random_state=42)
            hdbscan_model = HDBSCAN(min_cluster_size=min_cluster_size,
                                  min_samples=min_cluster_size-5 if min_cluster_size > 5 else 1,
                                  metric='euclidean', gen_min_span_tree=True,
                                  prediction_data=True)

            # Create and train model
            model = BERTopic(embedding_model=embedding_model,
                            umap_model=umap_model,
                            hdbscan_model=hdbscan_model)
            topics, _ = model.fit_transform(documents)

            # Calculate coherence score (higher is better)
            coherence = model.calculate_probabilities(documents).max(axis=1).mean()

            # Get number of topics excluding outlier topic (-1)
            num_topics = len(set(topics)) - (1 if -1 in topics else 0)

            # Calculate percent of documents in -1 (outlier) topic
            outlier_pct = sum(1 for t in topics if t == -1) / len(topics)

            results.append({
                'n_neighbors': n_neighbors,
                'min_cluster_size': min_cluster_size,
                'num_topics': num_topics,
                'coherence': coherence,
                'outlier_pct': outlier_pct
            })

            print(f"n_neighbors={n_neighbors}, min_cluster_size={min_cluster_size}: {num_topics} topics, {outlier_pct:.2f} outliers")

    return pd.DataFrame(results)

# Test different parameters (this can take time, so using a small range for demonstration)
n_neighbors_range = [15, 30, 50]
min_cluster_size_range = [10, 20, 30]

results_df = test_topic_parameters(documents_filtered[:2000], embedding_model,
                                 n_neighbors_range, min_cluster_size_range)

# Visualize the results
plt.figure(figsize=(12, 8))
for n_neighbors in n_neighbors_range:
    subset = results_df[results_df['n_neighbors'] == n_neighbors]
    plt.plot(subset['min_cluster_size'], subset['num_topics'],
            marker='o', label=f'n_neighbors={n_neighbors}')

plt.xlabel('Min Cluster Size')
plt.ylabel('Number of Topics')
plt.title('Parameter Impact on Number of Topics')
plt.legend()
plt.grid(True)
plt.show()

# Choose the best parameters based on balance of topics and coherence
best_params = results_df.sort_values(by=['coherence', 'outlier_pct'],
                                   ascending=[False, True]).iloc[0]
print(f"\nBest parameters: n_neighbors={best_params['n_neighbors']}, " +
      f"min_cluster_size={best_params['min_cluster_size']}")
print(f"This gives {best_params['num_topics']} topics with {best_params['outlier_pct']:.2f} outlier percentage")

# Train the optimal model with these parameters
optimal_umap = UMAP(n_neighbors=int(best_params['n_neighbors']),
                  n_components=5, min_dist=0.0, metric='cosine', random_state=42)
optimal_hdbscan = HDBSCAN(min_cluster_size=int(best_params['min_cluster_size']),
                        metric='euclidean', gen_min_span_tree=True, prediction_data=True)

optimal_model = BERTopic(embedding_model=embedding_model,
                        umap_model=optimal_umap,
                        hdbscan_model=optimal_hdbscan)

optimal_topics, optimal_probs = optimal_model.fit_transform(documents_filtered)

print(f"\nOptimal model found {len(set(optimal_topics)) - 1} topics")  # -1 for outlier topic


## 6. Interpreting the Results


# Get topic information
topic_info = optimal_model.get_topic_info()
print(topic_info.head(10))

# Examine top topics
print("\nTop 5 topics by size:")
for idx, row in topic_info.head(6).iterrows():
    if row['Topic'] != -1:  # Skip outlier topic
        topic_id = row['Topic']
        topic_words = optimal_model.get_topic(topic_id)
        topic_size = row['Count']
        print(f"Topic {topic_id} (Size: {topic_size}): {topic_words}")

# Compare with the original categories
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

# Ensure we're using the same filtered documents for evaluation
filtered_indices = [i for i, doc in enumerate(documents) if len(doc) > min_length]
filtered_categories = [categories[i] for i in filtered_indices]

# Calculate clustering metrics
ari = adjusted_rand_score(filtered_categories, optimal_topics)
nmi = normalized_mutual_info_score(filtered_categories, optimal_topics)

print(f"\nAdjusted Rand Index: {ari:.4f}")
print(f"Normalized Mutual Information: {nmi:.4f}")

# These metrics show how well our topics align with the original categories
# Higher values (closer to 1.0) indicate better alignment

# Let's see how topics relate to original categories
topic_category_df = pd.DataFrame({
    'Topic': optimal_topics,
    'Category': [category_names[cat] for cat in filtered_categories]
})

# Get distribution of categories within each topic
topic_category_crosstab = pd.crosstab(
    topic_category_df['Topic'],
    topic_category_df['Category'],
    normalize='index'
) * 100  # Convert to percentages

print("\nCategory distribution within top topics (%):")
print(topic_category_crosstab.head(10))

# Get the most representative documents for selected topics
print("\nExample documents from top topics:")
for topic_id in sorted(list(set(optimal_topics)))[:5]:
    if topic_id != -1:  # Skip outlier topic
        topic_docs = optimal_model.get_representative_docs(topic_id)
        print(f"\nTopic {topic_id}:")
        print(f"Representative document: {topic_docs[0][:200]}...")


## 7. Visualizing the Results


# 1. Topic Word Scores Visualization
optimal_model.visualize_barchart(top_n_topics=10)
plt.title("Top Terms for Major Topics")
plt.tight_layout()
plt.show()

# 2. Topic Similarity Map
topic_sim_map = optimal_model.visualize_topics()
plt.title("Topic Similarity Map")
plt.tight_layout()
plt.show()

# 3. Topic Hierarchy
hierarchical_topics = optimal_model.hierarchical_topics(documents_filtered)
hierarchical_viz = optimal_model.visualize_hierarchy(hierarchical_topics=hierarchical_topics)
plt.title("Topic Hierarchy")
plt.tight_layout()
plt.show()

# 4. Topic-Document Distribution
optimal_model.visualize_distribution(optimal_topics)
plt.title("Topic Distribution")
plt.tight_layout()
plt.show()

# 5. Topic Heatmap - comparing topics
fig = optimal_model.visualize_heatmap(n_clusters=10)
plt.title("Topic Similarity Heatmap")
plt.tight_layout()
plt.show()

# 6. Interactive Topic Space Visualization using UMAP and Plotly
# First, get reduced embeddings (if not already available)
embeddings = optimal_model.umap_model.transform(optimal_model._extract_embeddings(documents_filtered[:5000]))  # limit to 5000 for visualization

# Create dataframe for plotting
viz_df = pd.DataFrame({
    'UMAP1': embeddings[:, 0],
    'UMAP2': embeddings[:, 1],
    'Topic': [f'Topic {t}' if t != -1 else 'Outliers' for t in optimal_topics[:5000]],
    'Category': [category_names[categories[i]] if i < len(categories) else 'Unknown'
                for i in filtered_indices[:5000]]
})

# Plot with Plotly
fig = px.scatter(
    viz_df, x='UMAP1', y='UMAP2', color='Topic',
    hover_data=['Category'], title='Document Embedding Space by Topic',
    opacity=0.7
)
fig.show()

# 7. Topic-over-Time Evolution (using publication dates if available)
# For demonstration, we'll create a synthetic time attribute
import random
from datetime import datetime, timedelta

# Create synthetic dates (if the actual dataset doesn't have dates)
start_date = datetime(2022, 1, 1)
random.seed(42)
dates = [start_date + timedelta(days=random.randint(0, 365)) for _ in range(len(documents_filtered))]

# Visualize topics over time
try:
    topics_over_time = optimal_model.topics_over_time(documents_filtered,
                                                   timestamps=dates,
                                                   global_tuning=True,
                                                   evolution_tuning=True)

    optimal_model.visualize_topics_over_time(topics_over_time, top_n_topics=10)
    plt.title("Topic Evolution Over Time")
    plt.tight_layout()
    plt.show()
except Exception as e:
    print(f"Could not generate topics over time: {e}")


## 8. Advanced Analysis and Insights


# Find similar topics
topic_id = 0  # Choose a topic of interest
similar_topics = optimal_model.find_topics("computer", top_n=5)
print(f"Topics similar to 'computer':")
for topic, score in similar_topics:
    print(f"Topic {topic} (Similarity: {score:.4f}): {optimal_model.get_topic(topic)}")

# Dynamic topic modeling (if needed)
from bertopic.representation import MaximalMarginalRelevance

# Use MMR for more diverse topic representations
optimal_model.update_topics(documents_filtered, representation_model=MaximalMarginalRelevance())
print("\nUpdated topics with more diversity:")
for topic_id in range(5):
    print(f"Topic {topic_id}: {optimal_model.get_topic(topic_id)}")

# Topic reduction to simplify the model if needed
reduced_model = optimal_model.reduce_topics(documents_filtered, nr_topics=15)
print(f"\nReduced to {len(set(reduced_model[0])) - 1} topics")

# Save the model for later use
optimal_model.save("bertopic_newsgroups_model")
print("\nModel saved successfully.")


## 9. Conclusion and Key Takeaways


# Compare BERTopic with traditional methods
# Advantages:
# 1. No need for extensive preprocessing
# 2. Better semantic understanding through contextualized embeddings
# 3. Automatic determination of topic numbers
# 4. Better handling of polysemy and synonymy
# 5. Superior topic coherence

# Limitations:
# 1. Computationally more intensive
# 2. Requires more memory for large datasets
# 3. Less interpretable than traditional methods like LDA
# 4. Embedding quality affects results significantly

print("""
Key Takeaways:
1. BERTopic leverages transformer embeddings for better semantic understanding
2. Minimal preprocessing is needed compared to traditional methods
3. UMAP + HDBSCAN clustering automatically determines optimal topic numbers
4. c-TF-IDF provides interpretable topic representations
5. Topic coherence and separation is generally better than with LDA
6. The model can be fine-tuned via UMAP and HDBSCAN parameters
""")
