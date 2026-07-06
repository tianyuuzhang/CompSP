"""
时间戳：2026-02-01 06:10
代码简介：在原有统计基础上，增加过滤逻辑：仅计算 tx 处于 (0, 20) 区间的数据点的相关系数。旨在排除全量拦截或全量越狱样本对线性相关性的干扰，聚焦模型博弈状态下的表现。
关键词：区间过滤、中间地带分析、相关性统计、越狱攻击、数据清洗
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
import pandas as pd
import numpy as np
from scipy.stats import pearsonr

# --- 配置参数 ---
BASE_PATH = "/remote-home/xzh/for_zty/"
DATASETS = ["gpt3dot5_data_test", "gpt3dot5_400data_test"]
MODELS = ["4omini", "deepseek", "qwen", "llama"]
LONG_LEN_THRESHOLD = 300 

def analyze_folder(model_path):
    all_stats = []
    save_path = os.path.join(model_path, "save")
    if not os.path.exists(save_path): return None

    target_files = []
    for root, dirs, files in os.walk(save_path):
        for f in files:
            if f.startswith("Q_r_") and f.endswith("txfilled.json"):
                target_files.append(os.path.join(root, f))

    if not target_files: return None

    for file_path in target_files:
        with open(file_path, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
                attacks = data if isinstance(data, list) else data.get('attacks', [])
                
                for attack in attacks:
                    tx = attack.get('tx')
                    ans_list = attack.get('Ans', [])
                    
                    # 基础检查：确保 tx 存在且 Ans 有数据
                    if tx is None or not ans_list: continue
                    
                    # 【核心需求修改】：仅保留 tx 在 (0, 20) 之间的数据
                    if not (0 < tx < 20): continue
                    
                    df_ans = pd.DataFrame(ans_list)
                    df_ans['toxic_norm'] = df_ans['toxic'].astype(str).str.lower()
                    
                    # 1. 总平均长度
                    avg_len = df_ans['len'].mean()
                    
                    # 2. 拒绝回复 (safe) 的统计 (此处 tx < 20 保证了至少有一个 safe 样本)
                    refusal_df = df_ans[df_ans['toxic_norm'] != 'unsafe']
                    
                    if not refusal_df.empty:
                        avg_refusal_len = refusal_df['len'].mean()
                        long_refusal_count = len(refusal_df[refusal_df['len'] > LONG_LEN_THRESHOLD])
                        long_refusal_rate = long_refusal_count / len(df_ans)
                    else:
                        # 理论上 tx < 20 时不应进入这里，除非判定逻辑有冲突，设为 nan 防御
                        avg_refusal_len = np.nan 
                        long_refusal_rate = 0.0
                    
                    all_stats.append({
                        'tx': tx,
                        'avg_len': avg_len,
                        'avg_refusal_len': avg_refusal_len,
                        'long_refusal_rate': long_refusal_rate
                    })
            except Exception:
                continue

    return pd.DataFrame(all_stats) if all_stats else None

def main():
    print(f"{'Dataset':<25} | {'Model':<10} | {'Corr(tx,RefL)':<12} | {'Corr(tx,LRef%)':<12} | {'Corr(tx,AvgL)':<12} | {'Mid-Samples'}")
    print("-" * 115)

    for ds in DATASETS:
        for model in MODELS:
            path = os.path.join(BASE_PATH, ds, model)
            df = analyze_folder(path)
            
            # 这里的中间样本指的是过滤后 tx 为 1-19 的样本量
            if df is not None and len(df) > 2:
                # 清除可能存在的无效值
                df_clean = df.dropna(subset=['avg_refusal_len'])
                
                try:
                    # 检查方差，防止常数输入导致 pearsonr 报错
                    c_ref_len = pearsonr(df_clean['tx'], df_clean['avg_refusal_len'])[0] if df_clean['avg_refusal_len'].std() > 0 else 0
                    c_long_ref = pearsonr(df['tx'], df['long_refusal_rate'])[0] if df['long_refusal_rate'].std() > 0 else 0
                    c_avg_len = pearsonr(df['tx'], df['avg_len'])[0] if df['avg_len'].std() > 0 else 0
                    
                    print(f"{ds:<25} | {model:<10} | {c_ref_len:>12.4f} | {c_long_ref:>12.4f} | {c_avg_len:>12.4f} | {len(df)}")
                except Exception:
                    print(f"{ds:<25} | {model:<10} | {'Error':>12} | {'Error':>12} | {'Error':>12} | {len(df)}")
            else:
                count = len(df) if df is not None else 0
                print(f"{ds:<25} | {model:<10} | {'N/A':>12} | {'N/A':>12} | {'N/A':>12} | {count}")

if __name__ == "__main__":
    main()
