!pip install pymongo
!pip install umap-learn
!pip install seaborn
!pip install pandas
!pip install matplotlib
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pymongo import MongoClient
from umap import UMAP
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, normalized_mutual_info_score
from scipy.spatial.distance import cdist

def connect_mongodb():
    uri = "mongodb+srv://Cluster13662:PawanGupta666@cluster13662.s1t3w.mongodb.net/?retryWrites=true&w=majority&appName=Cluster13662"
    return MongoClient(uri)

def get_data():
    client = connect_mongodb()
    db = client['Organization5']
    collection = db['Non Profit1']
    data = list(collection.find({}, {
        "tag_embedding": 1,
        "description_embedding": 1,
        "Industries": 1
    }))
    tag_embeddings, desc_embeddings, industries = [], [], []
    for doc in data:
        try:
            if all(field in doc for field in ['tag_embedding', 'description_embedding', 'Industries']):
                if not doc['Industries'] or doc['Industries'] == "":
                    continue
                tag_emb = np.frombuffer(doc['tag_embedding'], dtype=np.float32)
                desc_emb = np.frombuffer(doc['description_embedding'], dtype=np.float32)
                tag_embeddings.append(tag_emb)
                desc_embeddings.append(desc_emb)
                industries.append(doc['Industries'])
        except:
            continue
    client.close()
    return np.array(tag_embeddings), np.array(desc_embeddings), industries

def plot_embedding(embedding_2d, labels, title, filename, method="UMAP"):
    plt.figure(figsize=(15, 10))
    unique_industries = sorted(set(labels))
    color_palette = sns.color_palette("husl", n_colors=len(unique_industries))
    color_dict = dict(zip(unique_industries, color_palette))
    df_plot = pd.DataFrame({"x": embedding_2d[:, 0], "y": embedding_2d[:, 1], "Industry": labels})
    for industry in unique_industries:
        mask = df_plot['Industry'] == industry
        plt.scatter(df_plot.loc[mask, 'x'], df_plot.loc[mask, 'y'], label=industry, c=[color_dict[industry]], s=100, alpha=0.7)
    plt.title(f"{method} Visualization of Organization {title}", pad=20)
    plt.xlabel(f"{method} Dimension 1")
    plt.ylabel(f"{method} Dimension 2")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.tight_layout()
    plt.savefig(f'exp1/{filename}.png', dpi=300, bbox_inches='tight')
    plt.close()

def calculate_db_index(X, labels):
    n_clusters = len(np.unique(labels))
    if n_clusters <= 1:
        return 0
    centroids = np.array([X[labels == i].mean(axis=0) for i in range(n_clusters)])
    cluster_distances = [np.mean([np.linalg.norm(x - centroids[i]) for x in X[labels == i]]) for i in range(n_clusters)]
    db_index = 0
    for i in range(n_clusters):
        max_ratio = 0
        for j in range(n_clusters):
            if i != j:
                ratio = (cluster_distances[i] + cluster_distances[j]) / np.linalg.norm(centroids[i] - centroids[j])
                max_ratio = max(max_ratio, ratio)
        db_index += max_ratio
    return db_index / n_clusters

def calculate_cosine_ratio(X, labels):
    intra_class_sim, inter_class_sim = [], []
    unique_labels = np.unique(labels)
    for label in unique_labels:
        class_samples = X[labels == label]
        if len(class_samples) > 1:
            similarities = cdist(class_samples, class_samples, metric='cosine')
            intra_class_sim.extend(similarities[np.triu_indices(len(similarities), k=1)])
        other_samples = X[labels != label]
        if len(class_samples) > 0 and len(other_samples) > 0:
            similarities = cdist(class_samples, other_samples, metric='cosine')
            inter_class_sim.extend(similarities.flatten())
    intra_mean = np.mean(intra_class_sim) if intra_class_sim else 0
    inter_mean = np.mean(inter_class_sim) if inter_class_sim else 1
    return inter_mean / intra_mean if intra_mean != 0 else 0

def evaluate_unsupervised(X, true_labels, name, n_clusters=10):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    pred_labels = kmeans.fit_predict(X)
    silhouette = silhouette_score(X, pred_labels)
    db = calculate_db_index(X, pred_labels)
    cosine_ratio = calculate_cosine_ratio(X, pred_labels)
    nmi = normalized_mutual_info_score(true_labels, pred_labels)
    return {
        'Embedding Type': name,
        'Silhouette Score': silhouette,
        'DB Index': db,
        'Cosine Ratio': cosine_ratio,
        'NMI': nmi
    }

def main():
    if not os.path.exists('exp1'):
        os.makedirs('exp1')

    print("Getting data from MongoDB...")
    tag_embeddings, desc_embeddings, industries = get_data()
    industry_counts = pd.Series(industries).value_counts()
    industry_counts = industry_counts[industry_counts.index != "Non-profit Organizations"]
    top_10_industries = industry_counts.head(10).index.tolist()
    print(f"Top 10 industries:\n{top_10_industries}")

    mask = pd.Series(industries).isin(top_10_industries)
    tag_embeddings_filtered = tag_embeddings[mask]
    desc_embeddings_filtered = desc_embeddings[mask]
    industries_filtered = np.array(industries)[mask]

    print(f"Filtered data size: {len(industries_filtered)}")

    print("Saving original embeddings...")
    np.save('exp1/nonprofit_tag_embeddings.npy', tag_embeddings_filtered)
    np.save('exp1/nonprofit_desc_embeddings.npy', desc_embeddings_filtered)
    pd.Series(industries_filtered).to_csv('exp1/nonprofit_industry_labels.csv', index=False)

    print("Performing UMAP for visualization only...")
    umap = UMAP(n_components=2, random_state=42)
    tag_umap = umap.fit_transform(tag_embeddings_filtered)
    desc_umap = umap.fit_transform(desc_embeddings_filtered)
    plot_embedding(tag_umap, industries_filtered, "Tag Embeddings", "nonprofit_tag_umap", "UMAP")
    plot_embedding(desc_umap, industries_filtered, "Description Embeddings", "nonprofit_desc_umap", "UMAP")

    print("Evaluating overall high-dimensional clustering...")
    metrics = [
        evaluate_unsupervised(tag_embeddings_filtered, industries_filtered, "Tag Embedding"),
        evaluate_unsupervised(desc_embeddings_filtered, industries_filtered, "Description Embedding")
    ]

    print("Evaluating per-industry clustering...")
    per_industry_metrics = []
    for industry in top_10_industries:
        inds = np.array(industries_filtered) == industry
        tag_sub = tag_embeddings_filtered[inds]
        desc_sub = desc_embeddings_filtered[inds]
        if len(tag_sub) >= 5:
            per_industry_metrics.append(evaluate_unsupervised(tag_sub, [industry]*len(tag_sub), f"{industry} - Tag"))
        if len(desc_sub) >= 5:
            per_industry_metrics.append(evaluate_unsupervised(desc_sub, [industry]*len(desc_sub), f"{industry} - Desc"))

    pd.DataFrame(metrics).to_csv('exp1/clustering_metrics_highdim.csv', index=False)
    pd.DataFrame(per_industry_metrics).to_csv('exp1/per_industry_clustering_metrics.csv', index=False)

    print("All clustering evaluations complete. See exp1 folder.")

if __name__ == "__main__":
    main()
