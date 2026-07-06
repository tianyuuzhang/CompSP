"""
时间戳：2026-02-04 10:15

代码简介：
在原有 ASR–文本联合分析管线基础上**保持全量重新计算**，并扩展统计口径：
(1) 继续从原始 JSON 重新构建 safe/unsafe × ASR 桶的词频矩阵；
(2) **显式统计每个 ASR 桶的 safe/unsafe 回答总数（类样本量）**；
(3) 在 cross_bin_stats.csv 中**同时保存“原始计数 + 归一化频率”**；
(4) 保留原有：跨桶趋势（Pearson）、形态分类、卡方显著性检验。

关键词：ASR分桶、safe/unsafe对比、计数+频率双口径、跨区间趋势、卡方检验、可复现实验
=== 注释自更新指引（供大模型阅读） ===
如果本代码被再次提交给大模型：
1. 请只在此注释块内更新：时间、代码简介、关键词；
2. 保持整体格式不变；
3. 根据“本次实际改动”重新撰写简介与关键词；
4. 其余代码不因注释规则而被重写。
================   ======================
"""

import os
import json
import re
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from scipy.stats import chi2_contingency, pearsonr

# ===========================
# ====== 你的原始配置 =======
# ===========================
BASE_PATH = "/remote-home/xzh/for_zty/"
DATASETS = ["gpt3dot5_data_test", "gpt3dot5_400data_test"]
MODELS = ["4omini", "deepseek", "qwen", "llama"]
LONG_LEN_THRESHOLD = 300

# ===========================
# ====== ASR 分桶配置 =======
# ===========================
ASR_BINS = {
    "B0": (0.0, 0.0),      # ASR = 0
    "B1": (0.0, 0.3),      # (0, 0.3]
    "B2": (0.3, 0.7),      # (0.3, 0.7]
    "B3": (0.7, 0.99),     # (0.7, 1)
    "B4": (1.0, 1.0),      # ASR = 1
}

BIN_ORDER = ["B0","B1","B2","B3","B4"]

MIN_DF = 10      # 过滤过罕见词
TOP_K = 30       # 每个区间输出前 K 个特征
NGRAM_RANGE = (1, 2)  # unigram + bigram

# ===========================
# ====== 工具函数 ===========
# ===========================

def clean_text(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-zA-Z\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def assign_bin(asr: float) -> str:
    for k, (low, high) in ASR_BINS.items():
        if (low == high == asr) or (low < asr <= high):
            return k
    return "OTHER"

# ===========================
# ====== 复用你的读入逻辑 ===
# ===========================

def analyze_folder_collect_responses(model_path: str) -> pd.DataFrame:
    """
    - 不做 tx 过滤
    - 展开到【单条回答】
    - 记录：text, asr, label(safe/unsafe)
    """
    all_rows = []
    save_path = os.path.join(model_path, "save")
    if not os.path.exists(save_path):
        return None

    target_files = []
    for root, dirs, files in os.walk(save_path):
        for f in files:
            if f.startswith("Q_r_") and f.endswith("txfilled.json"):
                target_files.append(os.path.join(root, f))

    if not target_files:
        return None

    for file_path in target_files:
        with open(file_path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                attacks = data if isinstance(data, list) else data.get("attacks", [])

                for attack in attacks:
                    tx = attack.get("tx")
                    ans_list = attack.get("Ans", [])
                    if tx is None or not ans_list:
                        continue

                    asr = tx / 20.0  # 你的 ASR 定义

                    for ans in ans_list:
                        text = str(ans.get("A", ""))
                        label = str(ans.get("toxic", "")).lower()
                        label = "unsafe" if label == "unsafe" else "safe"

                        all_rows.append({
                            "text": text,
                            "asr": asr,
                            "label": label,
                            "file": os.path.basename(file_path),
                            "round": attack.get("round")
                        })
            except Exception:
                continue

    return pd.DataFrame(all_rows) if all_rows else None

# ===========================
# ====== 核心分析模块 ========
# ===========================

def asr_text_feasibility_analysis(df: pd.DataFrame):
    """
    输出三张表：
    1) top_terms_bin: 每个 bin × safe/unsafe 的 Top-K 特征（区间特异性排序）
    2) full_stats: 每个 (term, bin) 的 p_safe, p_unsafe, delta, specificity
    3) cross_bin_stats: 
       - 原始计数：B*_safe_cnt, B*_unsafe_cnt
       - 归一化频率：B*_safe_freq, B*_unsafe_freq
       - 趋势 + 卡方
    """

    df = df.copy()
    df["text"] = df["text"].astype(str).apply(clean_text)
    df["bin"] = df["asr"].apply(assign_bin)
    df = df[df["bin"] != "OTHER"]

    # ---------- 向量化 ----------
    vectorizer = CountVectorizer(
        ngram_range=NGRAM_RANGE,
        min_df=MIN_DF,
        stop_words="english"
    )

    X = vectorizer.fit_transform(df["text"])
    vocab = np.array(vectorizer.get_feature_names_out())

    # ---------- 计数结构（词级）----------
    counts = {b: {"safe": Counter(), "unsafe": Counter()} for b in ASR_BINS}

    # ---------- **新增：类样本量（回答数）统计** ----------
    class_totals = {b: {"safe": 0, "unsafe": 0} for b in ASR_BINS}

    for i, row in df.iterrows():
        b = row["bin"]
        lab = row["label"]

        # 统计“每个 bin 里有多少条 safe/unsafe 回答”
        class_totals[b][lab] += 1

        tokens = X[i].nonzero()[1]
        for t in tokens:
            counts[b][lab][vocab[t]] += 1

    # 词级 token 总数（保持你原逻辑）
    totals = {
        b: {
            "safe": sum(counts[b]["safe"].values()),
            "unsafe": sum(counts[b]["unsafe"].values())
        }
        for b in counts
    }

    # ===========================
    # (A) 区间内 safe 与 unsafe 的对比
    # ===========================
    stats = []
    for term in vocab:
        for b in ASR_BINS:
            p_safe = counts[b]["safe"][term] / max(1, totals[b]["safe"])
            p_unsafe = counts[b]["unsafe"][term] / max(1, totals[b]["unsafe"])

            other_bins = [bb for bb in ASR_BINS if bb != b]
            p_other = np.mean([
                (counts[bb]["safe"][term] + counts[bb]["unsafe"][term]) /
                max(1, totals[bb]["safe"] + totals[bb]["unsafe"])
                for bb in other_bins
            ])

            specificity = (p_safe + p_unsafe) / max(1e-9, p_other)

            stats.append({
                "term": term,
                "bin": b,
                "p_safe": p_safe,
                "p_unsafe": p_unsafe,
                "delta_unsafe_safe": p_unsafe - p_safe,
                "specificity": specificity
            })

    full_stats = pd.DataFrame(stats)

    top_terms_bin = (
        full_stats
        .sort_values("specificity", ascending=False)
        .groupby("bin")
        .head(TOP_K)
        .reset_index(drop=True)
    )

    # ===========================
    # (B) 你真正关心的：跨区间分析（**新增频率列**）
    # ===========================
    records = []

    for term in vocab:
        safe_counts = [counts[b]["safe"][term] for b in BIN_ORDER]
        unsafe_counts = [counts[b]["unsafe"][term] for b in BIN_ORDER]

        # ---- 频率：按“类样本量”归一化（你的新要求）----
        safe_freq = [
            safe_counts[i] / max(1, class_totals[BIN_ORDER[i]]["safe"])
            for i in range(len(BIN_ORDER))
        ]
        unsafe_freq = [
            unsafe_counts[i] / max(1, class_totals[BIN_ORDER[i]]["unsafe"])
            for i in range(len(BIN_ORDER))
        ]

        # ---- 趋势相关 ----
        try:
            corr_safe, _ = pearsonr(range(len(BIN_ORDER)), safe_counts)
        except Exception:
            corr_safe = np.nan

        try:
            corr_unsafe, _ = pearsonr(range(len(BIN_ORDER)), unsafe_counts)
        except Exception:
            corr_unsafe = np.nan

        # ---- 卡方检验 ----
        chi2, p_chi, _, _ = chi2_contingency(
            np.array([safe_counts, unsafe_counts]) + 1
        )

        def trend_shape(arr):
            arr = np.array(arr)
            mid = arr[1:-1]
            if arr[-1] > arr[0] and np.all(np.diff(arr) >= 0):
                return "monotonic_increasing"
            if arr[-1] < arr[0] and np.all(np.diff(arr) <= 0):
                return "monotonic_decreasing"
            if mid.max() > max(arr[0], arr[-1]):
                return "middle_peak"
            return "mixed"

        rec = {
            "term": term,
            "chi2_p": p_chi,
            "safe_trend_corr": corr_safe,
            "unsafe_trend_corr": corr_unsafe,
            "safe_shape": trend_shape(safe_counts),
            "unsafe_shape": trend_shape(unsafe_counts),
        }

        # ======= 关键修改：同时写入【计数 + 频率】 =======
        for i, b in enumerate(BIN_ORDER):
            rec[f"{b}_safe_cnt"] = safe_counts[i]
            rec[f"{b}_unsafe_cnt"] = unsafe_counts[i]
            rec[f"{b}_safe_freq"] = safe_freq[i]
            rec[f"{b}_unsafe_freq"] = unsafe_freq[i]

        records.append(rec)

    cross_bin_stats = pd.DataFrame(records)

    return top_terms_bin, full_stats, cross_bin_stats

# ===========================
# ====== 主流程 ==============
# ===========================

def main():
    for ds in DATASETS:
        for model in MODELS:
            print(f"\n[COLLECT] {ds} / {model}")
            path = os.path.join(BASE_PATH, ds, model)

            df_resp = analyze_folder_collect_responses(path)
            if df_resp is None or len(df_resp) < 50:
                print(f"  -> insufficient data: {0 if df_resp is None else len(df_resp)}")
                continue

            print(f"  -> total responses: {len(df_resp)}")

            top_terms, full_stats, cross_bin = asr_text_feasibility_analysis(df_resp)

            key = f"{ds}__{model}"
            out_dir = os.path.join(BASE_PATH, "asr_text_results")
            os.makedirs(out_dir, exist_ok=True)

            top_terms.to_csv(os.path.join(out_dir, f"{key}_top_terms.csv"), index=False)
            full_stats.to_csv(os.path.join(out_dir, f"{key}_full_stats.csv"), index=False)
            cross_bin.to_csv(os.path.join(out_dir, f"{key}_cross_bin_stats.csv"), index=False)

            print(f"  -> saved: {out_dir}/{key}_*.csv")

if __name__ == "__main__":
    main()
