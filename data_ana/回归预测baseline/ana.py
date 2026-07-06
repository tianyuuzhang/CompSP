import os
import json
import numpy as np
import pandas as pd
import random
from collections import defaultdict

# ============================================================
# 1. 全局配置与硬编码数据 (保持不变)
# ============================================================

QUESTION_IDS_JBB = [9, 30, 73, 10, 37, 15, 42, 1, 50, 41, 45, 65, 70, 72, 21, 55, 56, 78, 48, 76]
QUESTION_IDS_HB = [190, 211, 120, 192, 94, 280, 356, 244, 230, 154, 336, 165, 318, 167, 248, 385, 123, 198, 354, 284]

MODELS = {
    "4omini": "4omini",
    "llama": "llama",
    "qwen": "qwen"
}

G_METHOD_DATA = {
    "4omini": {"JBB": (79.7, 1.8), "HB": (49.7, 14.7)},
    "llama":  {"JBB": (79.2, 1.4), "HB": (48.5, 10.1)},
    "qwen":   {"JBB": (57.8, 11.9), "HB": (41.6, 16.2)}
}

REGRESSION_ROOT_JB = "/remote-home/xzh/for_zty/gpt3dot5_data_test/regression"
REGRESSION_ROOT_HB = "/remote-home/xzh/for_zty/gpt3dot5_data_test/regression/400"

MAX_ATTEMPTS = 20.0
RANDOM_SIMULATIONS = 10000  # 建议根据性能调整，10000次在大数据量下可能较慢
TOP_K_BUCKET = 5
TX_BUCKETS = [(0, 24), (25, 49), (50, 74), (75, 99)]

# ============================================================
# 2. 基础计算工具
# ============================================================

def load_json(path):
    if not os.path.exists(path): return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except: return None

def calculate_first_success_cost(tx_sequence):
    total = 0.0
    for tx in tx_sequence:
        if tx <= 1e-6: 
            total += MAX_ATTEMPTS
        else:
            total += MAX_ATTEMPTS / tx
            return total
    return total

def baseline_asr_fasc(txs):
    if not txs: return np.nan, np.nan
    asr = np.mean(txs) / MAX_ATTEMPTS * 100
    fasc_vals = [calculate_first_success_cost(random.sample(txs, len(txs))) for _ in range(1000)] # 基准线采样1000次足够
    return asr, float(np.mean(fasc_vals))

def r_method_metrics_stochastic(items):
    """
    R_方法随机模拟核心：
    同时计算 ASR (Top-5) 和 FASC
    """
    if len(items) < 4: return np.nan, np.nan
    
    sim_asr_results = []
    sim_fasc_results = []
    
    for _ in range(RANDOM_SIMULATIONS):
        shuffled = items[:]
        random.shuffle(shuffled)
        
        n = len(shuffled)
        group_size = n // 4
        
        current_sim_group_asr = []
        current_sim_group_fasc = []
        
        for i in range(4):
            start = i * group_size
            end = (i + 1) * group_size if i < 3 else n
            group = shuffled[start:end]
            if not group: continue
            
            # 按预测值 (x[1]) 降序排列
            sorted_group = sorted(group, key=lambda x: x[1], reverse=True)
            tx_seq = [x[0] for x in sorted_group]
            
            # 1. 计算该组 ASR (取前 K 个)
            top_k_txs = tx_seq[:TOP_K_BUCKET]
            group_asr = (np.mean(top_k_txs) / MAX_ATTEMPTS) * 100
            current_sim_group_asr.append(group_asr)
            
            # 2. 计算该组 FASC
            current_sim_group_fasc.append(calculate_first_success_cost(tx_seq))
            
        if current_sim_group_asr:
            sim_asr_results.append(np.mean(current_sim_group_asr))
            sim_fasc_results.append(np.mean(current_sim_group_fasc))
            
    return (
        float(np.mean(sim_asr_results)) if sim_asr_results else np.nan,
        float(np.mean(sim_fasc_results)) if sim_fasc_results else np.nan
    )

# ============================================================
# 3. 核心统计逻辑
# ============================================================

def compute_question_metrics(data):
    txs = np.array([float(x.get("tx", 0)) for x in data])
    preds = np.array([float(x.get("predicted_tx", 0)) for x in data])
    
    b_asr_list, b_fasc_list, r_asr_list, r_fasc_list = [], [], [], []

    for lo, hi in TX_BUCKETS:
        mask = (txs >= lo) & (txs <= hi)
        if mask.sum() < 4: continue 
        
        tx_bucket, pred_bucket = txs[mask], preds[mask]
        
        # 基线计算
        basr, bfasc = baseline_asr_fasc(list(tx_bucket))
        b_asr_list.append(basr)
        b_fasc_list.append(bfasc)
        
        # R_方法 计算 (ASR 和 FASC 现在都通过随机模拟获得)
        items = list(zip(tx_bucket, pred_bucket))
        r_asr, r_fasc = r_method_metrics_stochastic(items)
        
        r_asr_list.append(r_asr)
        r_fasc_list.append(r_fasc)

    return (
        np.nanmean(b_asr_list) if b_asr_list else np.nan, 
        np.nanmean(b_fasc_list) if b_fasc_list else np.nan, 
        np.nanmean(r_asr_list) if r_asr_list else np.nan, 
        np.nanmean(r_fasc_list) if r_fasc_list else np.nan
    )

# ============================================================
# 4. 后续执行与 LaTeX（保持原样）
# ============================================================

def run_dataset(question_ids, root_dir, model_folder):
    results = defaultdict(list)
    if not os.path.exists(os.path.join(root_dir, model_folder)):
        return {k: np.nan for k in ["b_asr", "b_fasc", "r_asr", "r_fasc"]}, 0

    for q in question_ids:
        path = os.path.join(root_dir, model_folder, "save", f"Q_r_{q}_0_100_txfilled.json")
        data = load_json(path)
        if not data or len(data) < 20: continue

        m = compute_question_metrics(data)
        if not np.isnan(m[0]):
            results["b_asr"].append(m[0]); results["b_fasc"].append(m[1])
            results["r_asr"].append(m[2]); results["r_fasc"].append(m[3])

    stats = {k: float(np.nanmean(v)) if v else np.nan for k, v in results.items()}
    return stats, len(results["b_asr"])

def generate_latex(df):
    latex = r"""
\begin{table*}[htbp]
    \centering
    \caption{Performance Comparison: Baseline (B), Stochastic Bucket Top-5 (R), and Global (G)}
    \label{tab:asr_full_comparison}
    \resizebox{\textwidth}{!}{
    \begin{tabular}{lcccccccccccc}
        \toprule
        \multirow{2}{*}{Model} & \multicolumn{6}{c}{JBB Dataset} & \multicolumn{6}{c}{HB Dataset} \\
        \cmidrule(lr){2-7} \cmidrule(lr){8-13}
        & B\_ASR & R\_ASR & G\_ASR & B\_FASC & R\_FASC & G\_FASC & B\_ASR & R\_ASR & G\_ASR & B\_FASC & R\_FASC & G\_FASC \\
        \midrule
"""
    for _, row in df.iterrows():
        line = f"        {row['Model']} & {row['JB_B_ASR']:.1f} & {row['JB_R_ASR']:.1f} & {row['JB_G_ASR']:.1f} & {row['JB_B_FASC']:.1f} & {row['JB_R_FASC']:.1f} & {row['JB_G_FASC']:.1f} & " \
               f"{row['HB_B_ASR']:.1f} & {row['HB_R_ASR']:.1f} & {row['HB_G_ASR']:.1f} & {row['HB_B_FASC']:.1f} & {row['HB_R_FASC']:.1f} & {row['HB_G_FASC']:.1f} \\\\"
        latex += line + "\n"
        
        def get_imp(new, base, inverse=False):
            if np.isnan(new) or np.isnan(base) or base == 0: return 0.0
            if inverse: return (base - new) / base * 100
            return (new - base) / base * 100

        imp_line = f"        (Imp\\%) & - & {get_imp(row['JB_R_ASR'], row['JB_B_ASR']):.1f}\\% & {get_imp(row['JB_G_ASR'], row['JB_B_ASR']):.1f}\\% & - & {get_imp(row['JB_R_FASC'], row['JB_B_FASC'], True):.1f}\\% & {get_imp(row['JB_G_FASC'], row['JB_B_FASC'], True):.1f}\\% & " \
                   f"- & {get_imp(row['HB_R_ASR'], row['HB_B_ASR']):.1f}\\% & {get_imp(row['HB_G_ASR'], row['HB_B_ASR']):.1f}\\% & - & {get_imp(row['HB_R_FASC'], row['HB_B_FASC'], True):.1f}\\% & {get_imp(row['HB_G_FASC'], row['HB_B_FASC'], True):.1f}\\% \\\\"
        latex += imp_line + "\n"

    latex += r"""        \bottomrule
    \end{tabular}
    }
\end{table*}"""
    return latex

def run_full_analysis():
    summary_rows = []
    for model_label, folder in MODELS.items():
        jb_stats, jb_q = run_dataset(QUESTION_IDS_JBB, REGRESSION_ROOT_JB, folder)
        hb_stats, hb_q = run_dataset(QUESTION_IDS_HB, REGRESSION_ROOT_HB, folder)
        jb_g_asr, jb_g_fasc = G_METHOD_DATA[model_label]["JBB"]
        hb_g_asr, hb_g_fasc = G_METHOD_DATA[model_label]["HB"]

        summary_rows.append({
            "Model": model_label,
            "JB_B_ASR": jb_stats["b_asr"], "JB_R_ASR": jb_stats["r_asr"], "JB_G_ASR": jb_g_asr,
            "JB_B_FASC": jb_stats["b_fasc"], "JB_R_FASC": jb_stats["r_fasc"], "JB_G_FASC": jb_g_fasc,
            "HB_B_ASR": hb_stats["b_asr"], "HB_R_ASR": hb_stats["r_asr"], "HB_G_ASR": hb_g_asr,
            "HB_B_FASC": hb_stats["b_fasc"], "HB_R_FASC": hb_stats["r_fasc"], "HB_G_FASC": hb_g_fasc,
            "Qs": f"{jb_q}/{hb_q}"
        })

    df = pd.DataFrame(summary_rows)
    cols = ["Model", "JB_B_ASR", "JB_R_ASR", "JB_G_ASR", "JB_B_FASC", "JB_R_FASC", "JB_G_FASC",
            "HB_B_ASR", "HB_R_ASR", "HB_G_ASR", "HB_B_FASC", "HB_R_FASC", "HB_G_FASC"]
    
    print("\n" + "="*140)
    print("FINAL RESULTS (R_Method uses 10000-sim Stochastic ASR & FASC)")
    print("="*140)
    print(df[cols].to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    print("\n" + "="*30 + " LATEX CODE " + "="*30)
    print(generate_latex(df))

if __name__ == "__main__":
    run_full_analysis()
