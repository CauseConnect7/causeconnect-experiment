import os
import numpy as np
import pandas as pd
from pymongo import MongoClient
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples, normalized_mutual_info_score
from scipy.spatial.distance import cdist
from scipy.stats import wilcoxon

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
    return silhouette, {
        'Embedding Type': name,
        'Silhouette Score': silhouette,
        'DB Index': db,
        'Cosine Ratio': cosine_ratio,
        'NMI': nmi
    }

def main():
    if not os.path.exists('exp1'):
        os.makedirs('exp1')

    tag_embeddings, desc_embeddings, industries = get_data()

    # 不使用 top10，仅去除空行业
    mask = pd.Series(industries) != "For-profit Organizations"
    tag_embeddings_filtered = tag_embeddings[mask]
    desc_embeddings_filtered = desc_embeddings[mask]
    industries_filtered = np.array(industries)[mask]

    print(f"Total usable samples (non-empty industries): {len(industries_filtered)}")

    # 全体样本的 clustering evaluation
    sil_tag, metrics_tag = evaluate_unsupervised(tag_embeddings_filtered, industries_filtered, "Tag Embedding")
    sil_desc, metrics_desc = evaluate_unsupervised(desc_embeddings_filtered, industries_filtered, "Description Embedding")
    pd.DataFrame([metrics_tag, metrics_desc]).to_csv('exp1/clustering_metrics_full.csv', index=False)

    # Wilcoxon 检验（全体样本的 silhouette score）
    tag_labels = KMeans(n_clusters=10, random_state=42).fit_predict(tag_embeddings_filtered)
    desc_labels = KMeans(n_clusters=10, random_state=42).fit_predict(desc_embeddings_filtered)
    sil_tag_scores = silhouette_samples(tag_embeddings_filtered, tag_labels)
    sil_desc_scores = silhouette_samples(desc_embeddings_filtered, desc_labels)

    print(f"Silhouette scores matched samples: {len(sil_tag_scores)} vs {len(sil_desc_scores)}")
    print(f"Silhouette mean (tag): {np.mean(sil_tag_scores):.4f}, std: {np.std(sil_tag_scores):.4f}")
    print(f"Silhouette mean (desc): {np.mean(sil_desc_scores):.4f}, std: {np.std(sil_desc_scores):.4f}")

    stat, p_val = wilcoxon(sil_tag_scores, sil_desc_scores)

    with open("exp1/significance_test_full.txt", "w") as f:
        f.write(f"Wilcoxon signed-rank test for silhouette scores (full dataset)\n")
        f.write(f"Sample size: {len(sil_tag_scores)} paired samples\n")
        f.write(f"Tag mean ± std: {np.mean(sil_tag_scores):.4f} ± {np.std(sil_tag_scores):.4f}\n")
        f.write(f"Desc mean ± std: {np.mean(sil_desc_scores):.4f} ± {np.std(sil_desc_scores):.4f}\n")
        f.write(f"Statistic: {stat:.3f}\n")
        f.write(f"P-value: {p_val:.4f}\n")

    print(f"Wilcoxon test completed (full set): W={stat:.3f}, p={p_val:.4f}")


if __name__ == "__main__":
    main()
