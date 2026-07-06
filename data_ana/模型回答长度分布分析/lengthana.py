import os
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# =========================
# 1. 参数配置
# =========================
ROOT_DIR = "/path/to/ROOT_DIR"   # ← 修改为你的总目录
BIN_SIZE = 300                  # 每 300 长度一个 bin
MAX_LEN = 6000                  # 可按实际情况调大
BINS = np.arange(0, MAX_LEN + BIN_SIZE, BIN_SIZE)

# =========================
# 2. 读取 A 字段长度
# =========================
def extract_A_lengths(json_path):
    lengths = []
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for ans in data.get("Ans", []):
        if "len" in ans:
            lengths.append(ans["len"])
        elif "A" in ans:
            lengths.append(len(ans["A"]))
    return lengths


# =========================
# 3. 遍历所有 Q_i
# =========================
folder_lengths = defaultdict(list)
all_lengths = []

for folder in sorted(os.listdir(ROOT_DIR)):
    if not folder.startswith("Q_"):
        continue

    folder_path = os.path.join(ROOT_DIR, folder)
    if not os.path.isdir(folder_path):
        continue

    for file in os.listdir(folder_path):
        if not file.endswith(".json"):
            continue

        json_path = os.path.join(folder_path, file)
        lengths = extract_A_lengths(json_path)

        folder_lengths[folder].extend(lengths)
        all_lengths.extend(lengths)


# =========================
# 4. 绘图函数
# =========================
def plot_length_distribution(lengths, title, save_path):
    hist, edges = np.histogram(lengths, bins=BINS)

    plt.figure(figsize=(10, 5))
    plt.bar(edges[:-1], hist, width=BIN_SIZE, align="edge")
    plt.xlabel("Length of A (bin size = 300)")
    plt.ylabel("Count")
    plt.title(title)
    plt.xticks(edges, rotation=45)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


# =========================
# 5. 每个 Q_i 的小分布图
# =========================
os.makedirs("plots_per_Q", exist_ok=True)

for folder, lengths in folder_lengths.items():
    if len(lengths) == 0:
        continue

    save_path = os.path.join("plots_per_Q", f"{folder}_length_dist.png")
    plot_length_distribution(
        lengths,
        title=f"Length Distribution of A in {folder}",
        save_path=save_path
    )


# =========================
# 6. 总体分布图
# =========================
plot_length_distribution(
    all_lengths,
    title="Overall Length Distribution of A (All Q_i)",
    save_path="overall_length_distribution.png"
)

print("统计完成：")
print(f"  小分布图输出至：plots_per_Q/")
print(f"  总体分布图：overall_length_distribution.png")
