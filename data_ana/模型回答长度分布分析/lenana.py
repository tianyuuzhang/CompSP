import os
import json
import numpy as np
import matplotlib.pyplot as plt

# =====================================
# 1. 参数
# =====================================
OUTPUT_DIR = "/remote-home/zty/save/lengana"
os.makedirs(OUTPUT_DIR, exist_ok=True)

BIN_SIZE = 250
MAX_LEN = 5000
BINS = np.arange(0, MAX_LEN + BIN_SIZE, BIN_SIZE)

MODELS = ["qwen", "4omini", "llama"]

MODEL_DISPLAY_NAME = {
    "qwen": "Qwen-Plus",
    "4omini": "GPT-4o-mini",
    "llama": "Llama-3.1-8B"
}

BASE_PATH = "/remote-home/xzh/for_zty/gpt3dot5_data_test"

# =====================================
# 2. 抽取长度 + 记录指令
# =====================================
def extract_records(json_path):
    records = []

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for item in data:
        for ans in item.get("Ans", []):

            if "len" in ans:
                length = ans["len"]
            elif "A" in ans:
                length = len(ans["A"])
            else:
                continue

            toxicA = ans.get("toxicA", [])
            text = ans.get("A", "<NO_A_FIELD>")

            records.append({
                "length": length,
                "toxicA": toxicA,
                "text": text,
                "path": json_path
            })

    return records


# =====================================
# 3. 单模型处理（统计 + 溯源）
# =====================================
def process_model(model_name):
    all_lengths = []
    unsafe_lengths = []
    empty_toxicA_lengths = []

    unsafe_records = []
    empty_toxicA_records = []

    for i in range(0, 100):
        if model_name == "deepseek":
            json_path = f"{BASE_PATH}/deepseek/save/Q_{i}/Q_{i}_0_100_txfilled.json"
        else:
            json_path = f"{BASE_PATH}/{model_name}/save/Q_{i}/Q_r_{i}_0_100_txfilled.json"

        if not os.path.exists(json_path):
            continue

        records = extract_records(json_path)

        for r in records:
            all_lengths.append(r["length"])

            if isinstance(r["toxicA"], list) and "unsafe" in r["toxicA"]:
                unsafe_lengths.append(r["length"])
                unsafe_records.append(r)

            if not r["toxicA"]:
                empty_toxicA_lengths.append(r["length"])
                empty_toxicA_records.append(r)

    # 直方图
    hist_all, edges = np.histogram(all_lengths, bins=BINS)
    hist_unsafe, _ = np.histogram(unsafe_lengths, bins=BINS)

    unsafe_ratio = np.zeros_like(hist_all, dtype=float)
    mask = hist_all > 0
    unsafe_ratio[mask] = hist_unsafe[mask] / hist_all[mask] * 100

    shortest_unsafe = sorted(unsafe_records, key=lambda x: x["length"])[:5]
    longest_empty = sorted(empty_toxicA_records, key=lambda x: -x["length"])[:5]

    max_empty_len = max(empty_toxicA_lengths) if empty_toxicA_lengths else None

    return {
        "hist_all": hist_all,
        "hist_unsafe": hist_unsafe,
        "unsafe_ratio": unsafe_ratio,
        "edges": edges,
        "max_empty_len": max_empty_len,
        "shortest_unsafe": shortest_unsafe,
        "longest_empty": longest_empty
    }


# =====================================
# 4. 画图 + 打印结果
# =====================================
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
axes = axes.flatten()

for idx, model in enumerate(MODELS):
    result = process_model(model)

    # -------- 打印 unsafe / toxicA 为空的指令 --------
    print("\n" + "=" * 80)
    print(f"MODEL: {MODEL_DISPLAY_NAME[model]}")

    print("\n[Shortest 5 UNSAFE instructions]")
    for i, r in enumerate(result["shortest_unsafe"], 1):
        print(f"\n#{i}")
        print(f"Length: {r['length']}")
        print(f"File:   {r['path']}")
        print("Text:")
        print(r["text"])

    print("\n[Longest 5 EMPTY toxicA instructions]")
    for i, r in enumerate(result["longest_empty"], 1):
        print(f"\n#{i}")
        print(f"Length: {r['length']}")
        print(f"File:   {r['path']}")
        print("Text:")
        print(r["text"])

    # -------- 画图 --------
    ax1 = axes[idx]
    ax2 = ax1.twinx()

    bin_centers = result["edges"][:-1] + BIN_SIZE / 2

    ax1.plot(bin_centers, result["hist_all"], marker="o", label="Total")
    ax1.plot(bin_centers, result["hist_unsafe"], marker="s", label="Unsafe")
    ax1.set_xlabel("Length of Answer")
    ax1.set_ylabel("Count")

    ax2.plot(
        bin_centers,
        result["unsafe_ratio"],
        linestyle="--",
        marker="^",
        label="Unsafe Ratio (%)"
    )
    ax2.set_ylim(0, 100)
    ax2.set_ylabel("Unsafe Ratio (%)")

    if result["max_empty_len"] is not None:
        ax1.axvline(500, linestyle="--", color="black", linewidth=2)

    ax1.set_title(MODEL_DISPLAY_NAME[model])

    # 图例
    l1, lb1 = ax1.get_legend_handles_labels()
    l2, lb2 = ax2.get_legend_handles_labels()
    ax1.legend(l1 + l2, lb1 + lb2, fontsize=9, loc="upper right")

plt.tight_layout(rect=[0, 0, 1, 0.96])

save_path = os.path.join(OUTPUT_DIR, "unsafe_ratio_by_length_3models.png")
plt.savefig(save_path)
plt.close()

print(f"\n统计完成，图已保存至：{save_path}")
