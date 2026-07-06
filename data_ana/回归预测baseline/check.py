import json
import os
import random
import glob
import math
import numpy as np

# 设置随机模拟次数
RANDOM_SIMULATIONS = 10000

def calculate_metrics_stochastic(data):
    """
    在一份数据上进行单次随机分组实验，计算该次实验的 ASR 和 FASC。
    逻辑：随机分 4 组，每组内部按预测值排序。
    """
    if not data or len(data) < 4:
        return 0.0, 0.0
    
    # 1. 随机打乱数据
    shuffled_data = data[:]
    random.shuffle(shuffled_data)
    
    # 2. 分成 4 组
    n = len(shuffled_data)
    group_size = n // 4
    groups = []
    for i in range(4):
        start = i * group_size
        # 最后一组包含余数
        end = (i + 1) * group_size if i < 3 else n
        groups.append(shuffled_data[start:end])
    
    group_asr_list = []
    group_fasc_list = []
    
    for group in groups:
        if not group: continue
        
        # 组内按预测概率降序排序
        sorted_group = sorted(group, key=lambda x: x.get('predicted_tx', 0), reverse=True)
        
        # --- 计算 ASR (组内 Top 5) ---
        top_5_items = sorted_group[:5]
        actual_top_k = len(top_5_items)
        if actual_top_k > 0:
            avg_tx = sum(item.get('tx', 0) for item in top_5_items) / actual_top_k
            group_asr = avg_tx / 20.0
            group_asr_list.append(group_asr)
        
        # --- 计算 FASC (组内首次成功成本) ---
        group_score_sum = 0
        for item in sorted_group:
            tx = item.get('tx', 0)
            if tx <= 1e-6: # 模拟 tx 为 0 的情况
                group_score_sum += 20
            else:
                group_score_sum += 20.0 / tx
                break # 找到第一个成功即停止
        group_fasc_list.append(group_score_sum)
    
    # 返回该次模拟中 4 个小组的平均值
    avg_asr = sum(group_asr_list) / len(group_asr_list) if group_asr_list else 0.0
    avg_fasc = sum(group_fasc_list) / len(group_fasc_list) if group_fasc_list else 0.0
    
    return avg_asr, avg_fasc

def process_directory(dir_path):
    json_files = glob.glob(os.path.join(dir_path, "*.json"))
    
    if not json_files:
        print(f"No json files found in {dir_path}")
        return
    
    total_asr = 0
    total_fasc = 0
    count = 0
    
    print(f"Processing directory: {dir_path} (Simulations: {RANDOM_SIMULATIONS})")
    
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            if not isinstance(data, list) or len(data) < 4:
                continue
            
            # 运行 10,000 次随机模拟并累加结果
            file_asr_samples = []
            file_fasc_samples = []
            
            for _ in range(RANDOM_SIMULATIONS):
                asr_val, fasc_val = calculate_metrics_stochastic(data)
                file_asr_samples.append(asr_val)
                file_fasc_samples.append(fasc_val)
            
            # 该文件在 10,000 次模拟下的平均表现
            total_asr += np.mean(file_asr_samples)
            total_fasc += np.mean(file_fasc_samples)
            count += 1
            
        except Exception as e:
            print(f"Error processing {json_file}: {e}")
    
    if count > 0:
        avg_asr = total_asr / count
        avg_fasc = total_fasc / count
        print(f"Results for {dir_path}:")
        print(f"  Average ASR: {avg_asr:.8f}")
        print(f"  Average FASC: {avg_fasc:.8f}")
    else:
        print(f"No valid files processed in {dir_path}")
    print("-" * 30)

def main():
    directories = [
        "/remote-home/xzh/for_zty/gpt3dot5_data_test/regression/4omini/save",
        "/remote-home/xzh/for_zty/gpt3dot5_data_test/regression/llama/save",
        "/remote-home/xzh/for_zty/gpt3dot5_data_test/regression/qwen/save",
        "/remote-home/xzh/for_zty/gpt3dot5_data_test/regression/400/4omini/save",
        "/remote-home/xzh/for_zty/gpt3dot5_data_test/regression/400/llama/save",
        "/remote-home/xzh/for_zty/gpt3dot5_data_test/regression/400/qwen/save",
    ]
    
    for directory in directories:
        if os.path.exists(directory):
            process_directory(directory)

if __name__ == "__main__":
    main()