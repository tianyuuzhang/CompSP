import os
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# =====================================
# 1. 路径与参数
# =====================================
OUTPUT_DIR = "/remote-home/zty/save/lengana"
os.makedirs(OUTPUT_DIR, exist_ok=True)

BIN_SIZE = 100
MAX_LEN = 5000
BINS = np.arange(0, MAX_LEN + BIN_SIZE, BIN_SIZE)

# =====================================
# 2. 读取 A 的长度（仅 toxicA 含 unsafe）
# =====================================
def extract_unsafe_A_lengths(json_path):
    lengths = []

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 顶层是 list
    for item in data:
        for ans in item.get("Ans", []):

            toxicA = ans.get("toxicA", [])
            if not isinstance(toxicA, list):
                continue

            # 只统计 toxicA 中出现 unsafe 的指令
            if "unsafe" not in toxicA:
                continue

            if "len" in ans:
                lengths.append(ans["len"])
            elif "A" in ans:
                lengths.append(len(ans["A"]))

    return lengths


# =====================================
# 3. 遍历 Q_0 ~ Q_99
# =====================================
folder_lengths = defaultdict(list)
all_lengths = []

for i in range(100):
    q_dir = f"/remote-home/xzh/for_zty/gpt3dot5_data_test/qwen/save/Q_{i}"
    json_path = os.path.join(q_dir, f"Q_r_{i}_0_100_txfilled.json")

    if not os.path.exists(json_path):
        print(f"[跳过] 文件不存在: {json_path}")
        continue

    lengths = extract_unsafe_A_lengths(json_path)
    folder_lengths[f"Q_{i}"].extend(lengths)
    all_lengths.extend(lengths)

    print(f"[完成] Q_{i}: unsafe 指令 {len(lengths)} 条")


# =====================================
# 4. 绘图函数
# =====================================
def plot_length_distribution(lengths, title, save_path):
    hist, edges = np.histogram(lengths, bins=BINS)

    plt.figure(figsize=(10, 5))
    plt.bar(edges[:-1], hist, width=BIN_SIZE, align="edge")
    plt.xlabel("Length of A (bin = 100)")
    plt.ylabel("Count")
    plt.title(title)
    plt.xticks(edges, rotation=45)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


# =====================================
# 5. 每个 Q_i 的分布图（仅 unsafe）
# =====================================


# =====================================
# 6. 总体分布图（仅 unsafe）
# =====================================
plot_length_distribution(
    all_lengths,
    title="Overall Unsafe A Length Distribution (All Q_i)",
    save_path=os.path.join(OUTPUT_DIR, "overall_unsafe_len_dist_qwen.png")
)

print("\n统计完成（unsafe only）")
print(f"总体分布图: {OUTPUT_DIR}/overall_unsafe_len_dist_qwen.png")
