import os
import json
import numpy as np
import matplotlib.pyplot as plt

# =====================================
# 1. 参数
# =====================================
BASE_PATH = "/remote-home/xzh/for_zty/gpt3dot5_data_test"
OUTPUT_DIR = "/remote-home/zty/save/gra/txcorr"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODELS = ["qwen", "4omini", "llama"]

MODEL_DISPLAY_NAME = {
    "qwen": "Qwen-Plus",
    "4omini": "GPT-4o-mini",
    "llama": "Llama-3.1-8B"
}

PAIR_NAMES = [
    ("qwen", "4omini"),
    ("qwen", "llama"),
    ("4omini", "llama")
]

MAX_Q = 79
EXPECTED_TOTAL = 80 * 100  # 8000

GRID_MIN = 1
GRID_MAX = 19
GRID_SIZE = GRID_MAX - GRID_MIN + 1  # 19

# =====================================
# 2. 抽取 tx / txl（key = Q_i + round）
# =====================================
def extract_records(model_name):
    """
    return:
    {
        (Q_i, round): {
            "ASR": int,
            "ALR": int
        }
    }
    """
    records = {}

    for i in range(0, MAX_Q + 1):
        json_path = f"{BASE_PATH}/{model_name}/save/Q_{i}/Q_r_{i}_0_100_txfilled.json"
        if not os.path.exists(json_path):
            continue

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for item in data:
            if "tx" not in item or "txl" not in item or "round" not in item:
                continue

            key = (i, item["round"])
            records[key] = {
                "ASR": int(item["tx"]),
                "ALR": int(item["txl"])
            }

    return records

# =====================================
# 3. 加载数据
# =====================================
all_records = {
    model: extract_records(model)
    for model in MODELS
}

# =====================================
# 4. 构建 19×19 热力图
# =====================================
def build_heatmap(rec_a, rec_b, metric):
    heatmap = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)

    common_keys = sorted(set(rec_a.keys()) & set(rec_b.keys()))
    total_points = len(common_keys)

    valid_points = 0

    for k in common_keys:
        x = rec_a[k][metric]
        y = rec_b[k][metric]

        # 只保留 1~19
        if GRID_MIN <= x <= GRID_MAX and GRID_MIN <= y <= GRID_MAX:
            heatmap[y - GRID_MIN, x - GRID_MIN] += 1
            valid_points += 1

    print(
        f"总对齐点数: {total_points} | "
        f"进入热力图: {valid_points} | "
        f"被过滤: {total_points - valid_points}"
    )

    if total_points != EXPECTED_TOTAL:
        print(f"[WARNING] 原始点数异常，应为 {EXPECTED_TOTAL}")

    return heatmap

# =====================================
# 5. 作图：横3 × 纵2
# =====================================
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
metrics = ["ASR", "ALR"]

for col_idx, (model_a, model_b) in enumerate(PAIR_NAMES):
    rec_a = all_records[model_a]
    rec_b = all_records[model_b]

    print("\n" + "=" * 80)
    print(f"PAIR: {MODEL_DISPLAY_NAME[model_a]} vs {MODEL_DISPLAY_NAME[model_b]}")

    for row_idx, metric in enumerate(metrics):
        heatmap = build_heatmap(rec_a, rec_b, metric)

        ax = axes[row_idx, col_idx]
        im = ax.imshow(
            heatmap,
            origin="lower",
            cmap="Reds"
        )

        ax.set_xlabel(f"{MODEL_DISPLAY_NAME[model_a]} {metric}")
        ax.set_ylabel(f"{MODEL_DISPLAY_NAME[model_b]} {metric}")

        ax.set_title(
            f"{metric}: {MODEL_DISPLAY_NAME[model_a]} vs {MODEL_DISPLAY_NAME[model_b]}"
        )

        ax.set_xticks(range(GRID_SIZE))
        ax.set_yticks(range(GRID_SIZE))
        ax.set_xticklabels(range(GRID_MIN, GRID_MAX + 1))
        ax.set_yticklabels(range(GRID_MIN, GRID_MAX + 1))

        # 写格子数量
        for y in range(GRID_SIZE):
            for x in range(GRID_SIZE):
                if heatmap[y, x] > 0:
                    ax.text(
                        x, y,
                        str(heatmap[y, x]),
                        ha="center",
                        va="center",
                        fontsize=6
                    )

plt.tight_layout()
save_path = os.path.join(
    OUTPUT_DIR,
    "ASR_ALR_pairwise_heatmap_19x19_nonzero_non20.png"
)
plt.savefig(save_path, dpi=300)
plt.close()

print(f"\n完成：19×19（排除 0 / 20）热力图已保存至：\n{save_path}")
