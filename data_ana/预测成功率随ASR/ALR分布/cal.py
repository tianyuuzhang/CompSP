import os
import json
import glob
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

# ---------------------------------------------------
# 1. 参数配置 (分析模式)
# ---------------------------------------------------

# 最终结果输出目录
OUTPUT_FOLDER_ROOT = "/remote-home/zty/save/ASR_ALR_ACCU" 
os.makedirs(OUTPUT_FOLDER_ROOT, exist_ok=True)

# 输入文件的基础路径 (假设所有任务结果都存储在这里)
# 【注意】这里应指向您上一个脚本生成的六个JSON文件所在的目录
INPUT_RESULTS_FOLDER = "/remote-home/zty/save/ASR_ALR_ACCU" 

# 任务配置：键为任务名，值为结果文件名后缀
TASK_CONFIGS = {
    "Qwen-ASR": "100_qwen_tx",
    "4omini-ASR": "100_4omini_tx",
    "Llama-ASR": "100_llama_tx",
    "Qwen-ALR": "100_qwen_txl",
    "4omini-ALR": "100_4omini_txl",
    "Llama-ALR": "100_llama_txl",
}

# 可视化参数
HEATMAP_BIN_SIZE = 0.1
DIFF_PLOT_BIN_SIZE = 0.1 # 差值图x轴的区间大小

# 自定义颜色映射 (绿 -> 白(0.5) -> 红)
# 0.0（绿色）-> 0.5（白色）-> 1.0（红色）
CUSTOM_CMAP_COLORS = [
    (0.0, 'green'),
    (0.5, 'white'),
    (1.0, 'red')
]
CUSTOM_CMAP = LinearSegmentedColormap.from_list("custom_gwr", CUSTOM_CMAP_COLORS)

# ---------------------------------------------------
# 新增：统计指标计算函数
# ---------------------------------------------------
def calculate_statistics(df: pd.DataFrame, task_name: str) -> dict:
    """
    计算单个任务的2项核心统计指标：
    1. 总体预测成功率（准确率）
    2. 非0/1区间的预测成功率（q1_ratio和q2_ratio均不在{0,1}范围内的样本）
    """
    # 1. 总体预测成功率
    df['correct'] = (df['label'] == df['predicted_label']).astype(int)
    overall_accuracy = df['correct'].mean()
    overall_count = len(df)
    
    # 2. 筛选：q1_ratio和q2_ratio既非0也非1的样本
    mask_non_01 = (df['q1_ratio'] != 0) & (df['q1_ratio'] != 1) & \
                  (df['q2_ratio'] != 0) & (df['q2_ratio'] != 1)
    df_non_01 = df[mask_non_01]
    
    non_01_accuracy = df_non_01['correct'].mean() if len(df_non_01) > 0 else np.nan
    non_01_count = len(df_non_01)
    
    return {
        "task_name": task_name,
        "overall_accuracy": round(overall_accuracy, 4),
        "overall_sample_count": overall_count,
        "non_01_accuracy": round(non_01_accuracy, 4) if not np.isnan(non_01_accuracy) else np.nan,
        "non_01_sample_count": non_01_count
    }

def save_statistics(all_stats: list, output_folder: str):
    """
    将所有任务的统计结果保存为CSV文件，并打印到控制台
    """
    # 转换为 DataFrame
    stats_df = pd.DataFrame(all_stats)
    
    # 保存 CSV
    stats_path = os.path.join(output_folder, "prediction_accuracy_statistics.csv")
    stats_df.to_csv(stats_path, index=False, encoding='utf-8')
    print(f"\n--- 统计结果已保存至: {stats_path} ---")
    
    # 打印格式化结果
    print("\n=== 所有任务的预测成功率统计 ===")
    print(f"{'任务名称':<12} | {'总体准确率':<10} | {'总体样本数':<10} | {'非0/1区间准确率':<15} | {'非0/1样本数':<10}")
    print("-" * 70)
    for _, row in stats_df.iterrows():
        overall_acc = f"{row['overall_accuracy']:.4f}"
        non_01_acc = f"{row['non_01_accuracy']:.4f}" if not np.isnan(row['non_01_accuracy']) else "N/A"
        print(f"{row['task_name']:<12} | {overall_acc:<10} | {row['overall_sample_count']:<10} | {non_01_acc:<15} | {row['non_01_sample_count']:<10}")

# ---------------------------------------------------
# 2. 数据处理与可视化函数（原逻辑保留，无修改）
# ---------------------------------------------------

def prepare_heatmap_data(df: pd.DataFrame):
    """
    计算并格式化热力图所需的数据 (准确率和样本数)。
    对于没有样本的单元格，准确率设置为 0.5，计数设置为 0。
    """
    bins = np.arange(0, 1 + HEATMAP_BIN_SIZE, HEATMAP_BIN_SIZE)
    # 标签用区间的下限表示
    labels = [f"{i:.1f}" for i in bins[:-1]] 
    
    # 使用 pd.cut 进行分箱
    df['q1_bin'] = pd.cut(df['q1_ratio'], bins=bins, labels=labels, right=False)
    df['q2_bin'] = pd.cut(df['q2_ratio'], bins=bins, labels=labels, right=False)

    df['correct'] = (df['label'] == df['predicted_label']).astype(int)

    # 计算准确率和样本数量
    grouped = df.groupby(['q1_bin', 'q2_bin'], observed=False)['correct']
    
    # 计算准确率。使用 fill_value=0.5 填充没有数据的分箱
    accuracy_data = grouped.mean().unstack(fill_value=0.5) 
    
    # 计算计数。使用 fill_value=0 填充没有数据的分箱
    count_data = grouped.size().unstack(fill_value=0)

    # 创建用于热力图注解的文本标签
    annot_labels = pd.DataFrame('', index=accuracy_data.index, columns=accuracy_data.columns)

    for r_idx in accuracy_data.index:
        for c_idx in accuracy_data.columns:
            acc = accuracy_data.loc[r_idx, c_idx]
            count = count_data.loc[r_idx, c_idx]
            
            # 【核心修改】处理空单元格的显示
            if count > 0:
                # 有数据：显示实际准确率和计数
                annot_labels.loc[r_idx, c_idx] = f"{acc:.2f}\n({count})"
            else:
                # 无数据：显示默认准确率 0.50 和计数 (0)
                annot_labels.loc[r_idx, c_idx] = "0.50\n(0)"

    return accuracy_data, annot_labels, labels


def create_single_heatmap(ax, accuracy_data, annot_labels, task_title, tick_labels):
    """
    在指定的 Axes 上绘制单个热力图。
    """
    sns.heatmap(
        accuracy_data,
        ax=ax,
        annot=annot_labels,
        fmt="s",
        cmap=CUSTOM_CMAP,
        linewidths=.5,
        cbar=False, 
        vmin=0.0,
        vmax=1.0,
        cbar_kws={'label': 'Accuracy'},
        xticklabels=tick_labels,
        yticklabels=tick_labels,
    )
    
    # 设置刻度标签方向
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    
    ax.set_title(task_title, fontsize=16)
    
    # 设置轴标签
    ax.set_xlabel('$R_{ALR}$ (txl ratio)', fontsize=14)
    ax.set_ylabel('$R_{ASR}$ (tx ratio)', fontsize=14)


def visualize_all_results(all_data_map, output_folder):
    """
    加载所有任务数据，生成 2x3 拼接图。
    """
    print("--- 开始结果可视化：生成 2x3 热力图 ---")
    
    # 【核心修改】改为 2 行 3 列布局，调整 figsize
    fig, axes = plt.subplots(2, 3, figsize=(22, 15)) 
    axes = axes.flatten() 
    
    tick_labels = None 

    for i, (task_title, task_key) in enumerate(TASK_CONFIGS.items()):
        
        # 1. 获取数据
        try:
            df = all_data_map[task_key]
        except KeyError:
            print(f"警告: 任务 {task_key} 缺少数据，跳过。")
            continue

        # 2. 数据处理
        accuracy_data, annot_labels, current_tick_labels = prepare_heatmap_data(df)

        if tick_labels is None:
            tick_labels = current_tick_labels
        
        # 3. 绘制到对应的子图上
        ax = axes[i]
        create_single_heatmap(ax, accuracy_data, annot_labels, task_title, tick_labels)
    
    # --- 统一色条 ---
    sm = plt.cm.ScalarMappable(cmap=CUSTOM_CMAP, norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    
    # 【核心修改】调整色条位置，使其在 2x3 布局中更居中和合适
    cbar_ax = fig.add_axes([0.92, 0.1, 0.015, 0.8]) # [left, bottom, width, height]
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('Prediction Accuracy', fontsize=16)

    # 调整布局，防止重叠
    plt.suptitle('Prediction Accuracy Heatmap by Training Data Type ($R_{ASR}$ vs $R_{ALR}$)', fontsize=24, y=0.98)
    # 为色条和标题留出空间
    plt.tight_layout(rect=[0, 0.0, 0.90, 0.95]) 
    
    heatmap_path = os.path.join(output_folder, 'combined_accuracy_heatmap_2x3_gwr.png')
    plt.savefig(heatmap_path, dpi=300)
    plt.close()
    print(f"组合热力图已保存至: {heatmap_path}")

# ---------------------------------------------------
# 3. 主流程（新增统计计算逻辑）
# ---------------------------------------------------
def main():
    
    # 1. 确保输出目录存在
    os.makedirs(OUTPUT_FOLDER_ROOT, exist_ok=True)
    
    # 2. 读取所有任务的结果文件
    all_data_map = {}
    # 新增：存储所有任务的统计结果
    all_statistics = []
    
    print(f"--- 正在读取结果文件 (目录: {INPUT_RESULTS_FOLDER}) ---")
    
    for task_title, task_key in TASK_CONFIGS.items():
        results_path = os.path.join(INPUT_RESULTS_FOLDER, f"eval_results_{task_key}.json")
        
        if os.path.exists(results_path):
            try:
                with open(results_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                df = pd.json_normalize(data)
                
                required_cols = ['q1_ratio', 'q2_ratio', 'label', 'predicted_label']
                if all(col in df.columns for col in required_cols):
                    all_data_map[task_key] = df
                    print(f"-> 成功加载 {task_key}：{len(df)} 条数据。")
                    
                    # 新增：计算当前任务的统计指标
                    task_stats = calculate_statistics(df, task_title)
                    all_statistics.append(task_stats)
                else:
                    print(f"警告: {task_key} 文件缺少必要字段，跳过。")
                    print(f"需要: {required_cols}, 实际存在: {list(df.columns)}")

            except Exception as e:
                print(f"错误: 无法解析或处理文件 {results_path}，已跳过。错误: {e}")
        else:
            print(f"警告: 未找到文件 {results_path}，已跳过。")

    # 新增：计算并保存统计结果
    if all_statistics:
        save_statistics(all_statistics, OUTPUT_FOLDER_ROOT)
    
    # 3. 可视化
    if all_data_map:
        visualize_all_results(all_data_map, OUTPUT_FOLDER_ROOT)
    else:
        print("没有可用于可视化的数据。请检查 INPUT_RESULTS_FOLDER 和文件名。")

if __name__ == "__main__":
    main()
