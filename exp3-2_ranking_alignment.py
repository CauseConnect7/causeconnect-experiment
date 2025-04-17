from pymongo import MongoClient
import pandas as pd
import numpy as np
from scipy.stats import kendalltau
from sklearn.metrics import ndcg_score

# 连接 MongoDB
uri = "yourmongo db"
client = MongoClient(uri)
db = client['Organization5']
collection = db['User']

# 获取用户数据
cursor = collection.find({
    "ratings.set_A": {"$exists": True},
    "ratings.set_B": {"$exists": True},
    "algorithm_assignment.set_A": {"$exists": True},
    "algorithm_assignment.set_B": {"$exists": True}
})
users = list(cursor)

# 计算函数
def compute_ndcg_and_tau(user_ratings):
    org_ids = list(user_ratings.keys())
    scores = np.array([user_ratings[org] for org in org_ids])
    y_score = scores[np.arange(len(scores))].reshape(1, -1)
    y_true = np.sort(scores)[::-1].reshape(1, -1)
    ndcg = ndcg_score(y_true, y_score)
    recommended_rank = np.arange(len(scores))
    ideal_rank = np.argsort(-scores)
    tau, _ = kendalltau(recommended_rank, ideal_rank)
    return ndcg, tau

# 统计每个用户结果
results = []
for user in users:
    try:
        set_a_algo = user['algorithm_assignment']['set_A']
        set_b_algo = user['algorithm_assignment']['set_B']

        set_a_scores = {f"A_{i}": user['ratings']['set_A'][f"A_{i}"]['score'] for i in range(20) if f"A_{i}" in user['ratings']['set_A']}
        set_b_scores = {f"B_{i}": user['ratings']['set_B'][f"B_{i}"]['score'] for i in range(20) if f"B_{i}" in user['ratings']['set_B']}

        if len(set_a_scores) == 20 and len(set_b_scores) == 20:
            mean_a = np.mean(list(set_a_scores.values()))
            mean_b = np.mean(list(set_b_scores.values()))

            ndcg_a, tau_a = compute_ndcg_and_tau(set_a_scores)
            ndcg_b, tau_b = compute_ndcg_and_tau(set_b_scores)

            if set_a_algo == "complex":
                results.append({
                    "user_id": str(user["_id"]),
                    "complex_mean": mean_a,
                    "simple_mean": mean_b,
                    "ndcg_complex": ndcg_a,
                    "ndcg_simple": ndcg_b,
                    "tau_complex": tau_a,
                    "tau_simple": tau_b
                })
            else:
                results.append({
                    "user_id": str(user["_id"]),
                    "complex_mean": mean_b,
                    "simple_mean": mean_a,
                    "ndcg_complex": ndcg_b,
                    "ndcg_simple": ndcg_a,
                    "tau_complex": tau_b,
                    "tau_simple": tau_a
                })
    except Exception as e:
        print(f"Error processing user {user.get('_id')}: {e}")
        continue

# 保存或输出
df = pd.DataFrame(results)
df.to_csv("cold_start_ndcg_tau_results.csv", index=False)
print(df.head())

import pandas as pd
import numpy as np
from scipy.stats import wilcoxon

# 读取CSV文件
file_path = "cold_start_ndcg_tau_results.csv"
df = pd.read_csv(file_path)

# 计算：NDCG 和 Kendall's tau 哪个算法更优
ndcg_better_ratio = (df["ndcg_complex"] > df["ndcg_simple"]).mean()
tau_better_ratio = (df["tau_complex"] > df["tau_simple"]).mean()

# 计算平均值
mean_ndcg_complex = df["ndcg_complex"].mean()
mean_ndcg_simple = df["ndcg_simple"].mean()
mean_tau_complex = df["tau_complex"].mean()
mean_tau_simple = df["tau_simple"].mean()

# Wilcoxon 检验（是否 complex 比 simple 显著更好）
ndcg_stat, ndcg_p = wilcoxon(df["ndcg_complex"], df["ndcg_simple"], alternative="greater")
tau_stat, tau_p = wilcoxon(df["tau_complex"], df["tau_simple"], alternative="greater")

# 输出汇总结果
print("===== NDCG 和 Kendall's τ 评估结果 =====")
print(f"🔹 NDCG 优于 Simple 的用户比例: {ndcg_better_ratio:.2%}")
print(f"🔹 τ 优于 Simple 的用户比例: {tau_better_ratio:.2%}")
print(f"🔹 平均 NDCG - Complex: {mean_ndcg_complex:.4f}, Simple: {mean_ndcg_simple:.4f}")
print(f"🔹 平均 τ     - Complex: {mean_tau_complex:.4f}, Simple: {mean_tau_simple:.4f}")
print(f"🔹 Wilcoxon 测试（NDCG）: W = {ndcg_stat:.3f}, p = {ndcg_p:.5f}")
print(f"🔹 Wilcoxon 测试（τ）   : W = {tau_stat:.3f}, p = {tau_p:.5f}")

