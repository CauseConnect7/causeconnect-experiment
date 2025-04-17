from pymongo import MongoClient
import pandas as pd
import numpy as np
from scipy.stats import wilcoxon, ttest_rel, shapiro, probplot
import matplotlib.pyplot as plt
import seaborn as sns

# 设置 seaborn 风格
sns.set(style="whitegrid")

# 连接 MongoDB
uri = "your mongodb"
client = MongoClient(uri)
db = client['Organization5']
collection = db['User']

# 获取数据
cursor = collection.find({
    "ratings.set_A": {"$exists": True},
    "ratings.set_B": {"$exists": True},
    "algorithm_assignment.set_A": {"$exists": True},
    "algorithm_assignment.set_B": {"$exists": True}
})
users = list(cursor)

# 处理评分
paired_scores = []
for user in users:
    try:
        set_a_algo = user['algorithm_assignment']['set_A']
        set_b_algo = user['algorithm_assignment']['set_B']
        set_a_scores = [user['ratings']['set_A'][f"A_{i}"]['score'] for i in range(20) if f"A_{i}" in user['ratings']['set_A']]
        set_b_scores = [user['ratings']['set_B'][f"B_{i}"]['score'] for i in range(20) if f"B_{i}" in user['ratings']['set_B']]
        if len(set_a_scores) == 20 and len(set_b_scores) == 20:
            mean_a = np.mean(set_a_scores)
            mean_b = np.mean(set_b_scores)
            if set_a_algo == "complex":
                complex_score = mean_a
                simple_score = mean_b
            else:
                complex_score = mean_b
                simple_score = mean_a
            paired_scores.append({
                "user_id": str(user["_id"]),
                "complex_mean": complex_score,
                "simple_mean": simple_score
            })
    except:
        continue

# 创建 DataFrame
df = pd.DataFrame(paired_scores)

# 正态性检验
shapiro_complex = shapiro(df["complex_mean"])
shapiro_simple = shapiro(df["simple_mean"])

# 配对 t-test 与 Wilcoxon 检验
t_stat, t_p = ttest_rel(df["complex_mean"], df["simple_mean"], alternative="greater")
w_stat, w_p = wilcoxon(df["complex_mean"], df["simple_mean"], alternative="greater")

# 打印结果（原始统计分析部分保持不变）
print("===== Cold-Start Matching Evaluation =====")
print(f"Sample Size: {len(df)}")
print(f"Mean (Complex): {df['complex_mean'].mean():.3f}")
print(f"Mean (Simple):  {df['simple_mean'].mean():.3f}")
print(f"Preference for Complex: {(df['complex_mean'] > df['simple_mean']).mean()*100:.1f}%")
print("\n--- Normality Test (Shapiro-Wilk) ---")
print(f"Complex p={shapiro_complex.pvalue:.4f}, {'Normal' if shapiro_complex.pvalue > 0.05 else 'Non-normal'}")
print(f"Simple  p={shapiro_simple.pvalue:.4f}, {'Normal' if shapiro_simple.pvalue > 0.05 else 'Non-normal'}")
print("\n--- Paired t-test ---")
print(f"t = {t_stat:.3f}, p = {t_p:.5f}")
print("\n--- Wilcoxon Signed-Rank Test ---")
print(f"w = {w_stat:.3f}, p = {w_p:.5f}")

# 重新绘图：description based 在左、tag-based 在右 + 默认蓝橙配色
plt.figure(figsize=(8, 6))

# 构造长格式数据，将列名称修改为输出图片所需的新名称
df_long = df[["complex_mean", "simple_mean"]].rename(
    columns={"complex_mean": "tag-based", "simple_mean": "description based"}
).melt(var_name="Algorithm", value_name="Score")

# 指定颜色：description based（蓝），tag-based（橙）
palette = {"description based": "#1f77b4", "tag-based": "#ff7f0e"}  # seaborn deep 默认前两色

sns.boxplot(data=df_long, x="Algorithm", y="Score", order=["description based", "tag-based"], width=0.4, palette=palette)
sns.swarmplot(data=df_long, x="Algorithm", y="Score", order=["description based", "tag-based"], color=".25")

plt.title("Boxplot of Mean Scores (description based vs. tag-based)")
plt.ylabel("Mean Score")
plt.grid(True)
plt.tight_layout()
plt.savefig("boxplot_swarm.png")
plt.close()

# 2. KDE Plot：更改图例标签
plt.figure(figsize=(8, 6))
sns.kdeplot(df["complex_mean"], label="tag-based", fill=True)
sns.kdeplot(df["simple_mean"], label="description based", fill=True)
plt.title("KDE of Mean Scores")
plt.xlabel("Mean Score")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("kde_plot.png")
plt.close()

# 3. Q-Q Plot: tag-based
plt.figure(figsize=(6, 6))
probplot(df["complex_mean"], dist="norm", plot=plt)
plt.title("Q-Q Plot: tag-based Mean")
plt.grid(True)
plt.tight_layout()
plt.savefig("qqplot_complex.png")
plt.close()

# 4. Q-Q Plot: description based
plt.figure(figsize=(6, 6))
probplot(df["simple_mean"], dist="norm", plot=plt)
plt.title("Q-Q Plot: description based Mean")
plt.grid(True)
plt.tight_layout()
plt.savefig("qqplot_simple.png")
plt.close()

print("📁 所有图表已保存为 boxplot_swarm.png, kde_plot.png, qqplot_complex.png, qqplot_simple.png")


# 添加 Match Rate 分析模块
match_rate_data = []
for user in users:
    try:
        set_a_algo = user['algorithm_assignment']['set_A']
        set_b_algo = user['algorithm_assignment']['set_B']
        set_a_scores = [user['ratings']['set_A'][f"A_{i}"]['score'] for i in range(20) if f"A_{i}" in user['ratings']['set_A']]
        set_b_scores = [user['ratings']['set_B'][f"B_{i}"]['score'] for i in range(20) if f"B_{i}" in user['ratings']['set_B']]
        if len(set_a_scores) == 20 and len(set_b_scores) == 20:
            match_a = sum([1 for score in set_a_scores if score >= 6]) / 20
            match_b = sum([1 for score in set_b_scores if score >= 6]) / 20
            if set_a_algo == "complex":
                match_rate_data.append({
                    "user_id": str(user["_id"]),
                    "tag_match_rate": match_a,
                    "desc_match_rate": match_b
                })
            else:
                match_rate_data.append({
                    "user_id": str(user["_id"]),
                    "tag_match_rate": match_b,
                    "desc_match_rate": match_a
                })
    except:
        continue

# 生成 match rate DataFrame 并打印统计摘要
df_match = pd.DataFrame(match_rate_data)
mean_tag_match = df_match["tag_match_rate"].mean()
mean_desc_match = df_match["desc_match_rate"].mean()
tag_better_match_share = (df_match["tag_match_rate"] > df_match["desc_match_rate"]).mean()

print("\n--- Match Rate Analysis ---")
print(f"Average Match Rate (Tag-based): {mean_tag_match:.3f}")
print(f"Average Match Rate (Description-based): {mean_desc_match:.3f}")
print(f"Tag-based match rate higher in: {tag_better_match_share*100:.1f}% of users")
