import os
import json
import pandas as pd
import numpy as np
from collections import Counter

# ===========================
# ====== 你的路径 ===========
# ===========================
BASE_PATH = "/remote-home/xzh/for_zty/"
DATASETS = ["gpt3dot5_data_test", "gpt3dot5_400data_test"]
MODELS = ["4omini", "deepseek", "qwen", "llama"]

ASR_BINS = {
    "B0": (0.0, 0.0),      # ASR = 0
    "B1": (0.0, 0.3),      # (0, 0.3]
    "B2": (0.3, 0.7),      # (0.3, 0.7]
    "B3": (0.7, 1.0),      # (0.7, 1)
    "B4": (1.0, 1.0),      # ASR = 1
}
BIN_ORDER = ["B0","B1","B2","B3","B4"]

RESULT_DIR = os.path.join(BASE_PATH, "asr_text_results")
OUTPUT = os.path.join(BASE_PATH, "asr_text_results", "unified_term_asr_stats.csv")
os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)

# ===========================
# === 1) ASR 桶分配 ==========
# ===========================

def assign_bin(asr: float) -> str:
    for k, (low, high) in ASR_BINS.items():
        if (low == high == asr) or (low < asr <= high):
            return k
    return "OTHER"

# ===========================
# === 2) 统计“问题总数/桶” ===
# ===========================

def count_questions_per_bin():
    """
    统计：每个 ASR 桶的“问题总数”
    """
    bin_counter = Counter()

    for ds in DATASETS:
        for model in MODELS:
            save_path = os.path.join(BASE_PATH, ds, model, "save")
            if not os.path.exists(save_path):
                continue

            for root, dirs, files in os.walk(save_path):
                for f in files:
                    if f.startswith("Q_r_") and f.endswith("txfilled.json"):
                        path = os.path.join(root, f)
                        with open(path, "r", encoding="utf-8") as fh:
                            try:
                                data = json.load(fh)
                                attacks = data if isinstance(data, list) else data.get("attacks", [])
                            except Exception:
                                continue

                            for attack in attacks:
                                tx = attack.get("tx")
                                if tx is None:
                                    continue
                                asr = tx / 20.0
                                b = assign_bin(asr)
                                if b != "OTHER":
                                    bin_counter[b] += 1

    return pd.Series(bin_counter).reindex(BIN_ORDER).fillna(0).astype(int)

bin_question_counts = count_questions_per_bin()
print("=== 问题总数 / ASR 桶 ===")
print(bin_question_counts)

# ===========================
# === 3) 读入已有 CSV ========
# ===========================

def load_all_cross_bin_stats(result_dir: str) -> pd.DataFrame:
    dfs = []
    for f in os.listdir(result_dir):
        if f.endswith("_cross_bin_stats.csv"):
            dfs.append(pd.read_csv(os.path.join(result_dir, f)))

    df = pd.concat(dfs, ignore_index=True)
    return df

cross = load_all_cross_bin_stats(RESULT_DIR)

# ===========================
# === 4) 统一：用“问题总数”做分母 ===
# ===========================

for b in BIN_ORDER:
    cross[f"p_safe_{b}"] = cross[f"{b}_safe"] / np.maximum(1, bin_question_counts[b])
    cross[f"p_unsafe_{b}"] = cross[f"{b}_unsafe"] / np.maximum(1, bin_question_counts[b])

# ===========================
# === 5) 重新计算排序指标 ===
# ===========================

def row_var_safe(row):
    vals = [row[f"p_safe_{b}"] for b in BIN_ORDER]
    return np.var(vals)

def row_var_unsafe(row):
    vals = [row[f"p_unsafe_{b}"] for b in BIN_ORDER]
    return np.var(vals)

def row_range(row, kind="safe"):
    vals = [row[f"p_{kind}_{b}"] for b in BIN_ORDER]
    return max(vals) - min(vals)

def row_sep(row):
    diffs = [abs(row[f"p_unsafe_{b}"] - row[f"p_safe_{b}"]) for b in BIN_ORDER]
    return np.mean(diffs)

cross["var_safe"] = cross.apply(row_var_safe, axis=1)
cross["var_unsafe"] = cross.apply(row_var_unsafe, axis=1)
cross["score_main"] = cross["var_safe"] + cross["var_unsafe"]
cross["range"] = cross.apply(lambda r: row_range(r,"safe") + row_range(r,"unsafe"), axis=1)
cross["sep"] = cross.apply(row_sep, axis=1)

# ===========================
# === 6) 输出“统一版”表 ======
# ===========================

keep_cols = (
    ["term", "score_main", "var_safe", "var_unsafe", "range", "sep", 
     "chi2_p", "safe_trend_corr", "unsafe_trend_corr", 
     "safe_shape", "unsafe_shape"]
    + [f"p_safe_{b}" for b in BIN_ORDER]
    + [f"p_unsafe_{b}" for b in BIN_ORDER]
)

unified = cross[keep_cols].sort_values("score_main", ascending=False)
unified.to_csv(OUTPUT, index=False)

print(f"\n=== 已输出【统一口径】结果至：\n{OUTPUT}")
