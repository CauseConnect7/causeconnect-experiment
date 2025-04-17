# 🧠 CauseConnect Experiments

This repository contains the experimental scripts and evaluations for the **CauseConnect** research system — an AI-powered platform for nonprofit–for-profit partnership discovery. These experiments were conducted as part of a thesis project at the University of Washington Information School.

## 🔍 Overview

The experiments analyze how structured tag-based embeddings (using GPT + OpenAI ADA) improve semantic retrieval quality compared to traditional description-based embeddings, across three dimensions:

1. **Embedding Similarity Distributions**
2. **Cluster Coherence**
3. **Cold-Start Matching Relevance**

Each experiment is implemented in a modular Python script, using MongoDB for data, OpenAI embeddings, and sklearn metrics for clustering and ranking.

---

## 📁 Directory Structure
```bash
. 
├── LICENSE 
├── exp1_similarity_analysis/ 
│ └── main.py 
├── exp2-1_top10_industry_analysis.py 
├── exp2-2_global_significance_test.py 
├── exp3-1__score_and_match_rate_analysis.py 
├── exp3-2_ranking_alignment.py 
└── README.md
```

---

## 🧪 Experiments Breakdown

### 📊 `exp1_similarity_analysis/`
Analyzes pairwise cosine similarity distributions under tag-based vs. description-based embeddings. Includes:
- KDE plots
- Boxplots
- CDF analysis
- Kolmogorov–Smirnov (KS) test

📂 Output saved in `/exp1/`.

---

### 🧩 `exp2-1_top10_industry_analysis.py`
Performs cluster analysis within the **top 10 industries**:
- UMAP projection for visualization
- Clustering metrics: Silhouette, Davies–Bouldin, Cosine Ratio, NMI

---

### 🧠 `exp2-2_global_significance_test.py`
Runs **global clustering evaluation** (across all organizations with valid industries), including:
- Wilcoxon signed-rank test
- Full silhouette score comparison

---

### 💬 `exp3-1__score_and_match_rate_analysis.py`
User-based cold-start partner relevance evaluation:
- Average rating analysis (tag vs. description)
- Normality tests, t-test, Wilcoxon test
- Match rate: % of recommendations rated ≥ 6
- Visualization: boxplot, KDE, Q-Q plots

---

### 📈 `exp3-2_ranking_alignment.py`
Evaluates ranking alignment quality:
- NDCG@20
- Kendall’s τ
- Wilcoxon test on rank consistency

---

## ⚙️ Environment

### Python Dependencies
```bash
pip install pymongo seaborn pandas matplotlib scikit-learn umap-learn
```
