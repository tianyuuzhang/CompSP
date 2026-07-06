import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
from itertools import combinations
import sys

# 配置路径和模型名称
base_dir = "/remote-home/xzh/for_zty/gpt3dot5_data_test"
output_dir = "/remote-home/zty/save/ana_txcorr"
os.makedirs(output_dir, exist_ok=True)  # 创建输出目录（若不存在）
output_file = os.path.join(output_dir, "tx_correlation_analysis.txt")
cache_file = os.path.join(output_dir, "tx_data_cache.npy")  # tx数据缓存文件

model_names = ["qwen", "llama", "4omini"]
model_paths = {name: os.path.join(base_dir, name, "save") for name in model_names}
q_range = range(80)  # Q_0 到 Q_79
total_qa = 20  # 从数据格式推断totQA=20
tx_values = list(range(21))  # tx值范围：0-20（转为列表，确保轴序正确）


# 自定义输出类：同时打印到控制台和文件
class Tee:
    def __init__(self, filename):
        self.file = open(filename, 'w', encoding='utf-8')
        self.stdout = sys.stdout

    def write(self, message):
        self.stdout.write(message)
        self.file.write(message)

    def flush(self):
        self.stdout.flush()
        self.file.flush()

    def close(self):
        self.file.close()
        sys.stdout = self.stdout


# 重定向输出到文件和控制台
tee = Tee(output_file)
sys.stdout = tee


def load_tx_data_from_json(model_path):
    """从JSON文件加载单个模型的tx数据（原始加载逻辑）"""
    tx_data = {}
    for q in q_range:
        q_dir = os.path.join(model_path, f"Q_{q}")
        json_file = os.path.join(q_dir, f"Q_r_{q}_0_100_txfilled.json")
        
        try:
            if not os.path.exists(json_file):
                raise FileNotFoundError(f"文件不存在（序号不一致？）")
            
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            tx_list = [item['tx'] for item in data[:total_qa]]
            tx_data[f"Q_{q}"] = tx_list
            print(f"成功加载 {json_file}（序号校验通过）")
            
        except Exception as e:
            print(f"加载文件失败 {json_file}: {e}")
            tx_data[f"Q_{q}"] = [None] * total_qa  # 标记错误数据
    return tx_data


def load_all_tx_data():
    """加载所有模型的tx数据：优先从缓存文件读取，无缓存则从JSON加载并保存缓存"""
    if os.path.exists(cache_file):
        print(f"\n发现tx数据缓存文件 {cache_file}，正在从缓存加载...")
        all_data = np.load(cache_file, allow_pickle=True).item()
        print(f"缓存加载完成！共加载 {len(all_data)} 个模型的tx数据")
        return all_data
    else:
        print(f"\n未发现tx数据缓存文件，正在从JSON文件加载...")
        all_data = {name: load_tx_data_from_json(path) for name, path in model_paths.items()}
        np.save(cache_file, all_data)
        print(f"JSON加载完成！已将tx数据保存到缓存文件 {cache_file}（下次运行将直接读取缓存）")
        return all_data


# 加载所有模型数据（优先缓存）
all_data = load_all_tx_data()


def categorize_tx(tx_value):
    """将tx值分类为0、1-19、20（用于3×3矩阵）"""
    if tx_value == 0:
        return 0
    elif 1 <= tx_value <= 19:
        return 1
    elif tx_value == 20:
        return 2
    else:
        return -1  # 异常值


def compute_21x21_counts(a_data, b_data):
    """计算两个模型的21×21交叉计数矩阵（tx值0-20，用于热力图）"""
    counts = np.zeros((21, 21), dtype=int)
    for a, b in zip(a_data, b_data):
        if a is None or b is None:
            continue
        if 0 <= a <= 20 and 0 <= b <= 20:
            counts[a, b] += 1
    return counts


def compute_3x3_counts(a_data, b_data):
    """计算两个模型的3×3交叉计数矩阵（用于概览）"""
    counts = np.zeros((3, 3), dtype=int)
    for a, b in zip(a_data, b_data):
        if a is None or b is None:
            continue
        a_cat = categorize_tx(a)
        b_cat = categorize_tx(b)
        if 0 <= a_cat < 3 and 0 <= b_cat < 3:
            counts[a_cat, b_cat] += 1
    return counts


# 1. 计算每个Q_i的相关性和3×3交叉矩阵（保留细节分析，不生成图片）
q_corrs = {name: {} for name in combinations(model_names, 2)}
q_cross_3x3 = {name: {} for name in combinations(model_names, 2)}

for q in q_range:
    q_name = f"Q_{q}"
    q_datas = {name: all_data[name][q_name] for name in model_names}
    
    for (m1, m2) in combinations(model_names, 2):
        data1 = q_datas[m1]
        data2 = q_datas[m2]
        
        # 过滤无效值
        valid_pairs = [(d1, d2) for d1, d2 in zip(data1, data2) 
                      if d1 is not None and d2 is not None and 
                      0 <= d1 <= 20 and 0 <= d2 <= 20]
        
        if len(valid_pairs) < 2:
            q_corrs[(m1, m2)][q_name] = np.nan
            q_cross_3x3[(m1, m2)][q_name] = np.zeros((3, 3))
            continue
        
        # 计算Spearman相关性
        corr, _ = spearmanr([p[0] for p in valid_pairs], [p[1] for p in valid_pairs])
        q_corrs[(m1, m2)][q_name] = corr
        
        # 计算3×3交叉矩阵
        cross_3x3 = compute_3x3_counts(data1, data2)
        q_cross_3x3[(m1, m2)][q_name] = cross_3x3


# 2. 计算总体相关性、3×3矩阵和21×21矩阵（用于总结和热力图）
total_corrs = {}
total_cross_3x3 = {}
total_cross_21x21 = {}

for (m1, m2) in combinations(model_names, 2):
    # 收集所有Q_i的有效数据对
    all_pairs = []
    for q in q_range:
        q_name = f"Q_{q}"
        data1 = all_data[m1][q_name]
        data2 = all_data[m2][q_name]
        valid_pairs = [(d1, d2) for d1, d2 in zip(data1, data2)
                      if d1 is not None and d2 is not None and
                      0 <= d1 <= 20 and 0 <= d2 <= 20]
        all_pairs.extend(valid_pairs)
    
    # 总体相关性
    corr, _ = spearmanr([p[0] for p in all_pairs], [p[1] for p in all_pairs])
    total_corrs[(m1, m2)] = corr
    
    # 总体3×3矩阵
    total_data1 = [p[0] for p in all_pairs]
    total_data2 = [p[1] for p in all_pairs]
    total_cross_3x3[(m1, m2)] = compute_3x3_counts(total_data1, total_data2)
    
    # 总体21×21矩阵（用于热力图）
    total_cross_21x21[(m1, m2)] = compute_21x21_counts(total_data1, total_data2)


# 3. 绘制总体21×21热力图（核心修改：颜色从白色→红色渐变，纵轴0在下20在上）
def plot_total_heatmap(matrix, m1, m2):
    """绘制并保存总体21×21热力图（0→最大值：白色→红色渐变，纵轴0在下20在上）"""
    # 设置中文字体兜底（实际图表用英文）
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False
    
    plt.figure(figsize=(12, 10))
    
    # 1. 核心修改：自定义白色→红色渐变的色映射
    # 创建从白色（#FFFFFF）到红色（#FF0000）的渐变色板，包含256个色阶
    colors = ["#FFFFFF"] + [plt.cm.Reds(i) for i in np.linspace(0.1, 1, 255)]
    white_red_cmap = plt.cm.colors.ListedColormap(colors)
    # 设置vmin=0（最小值对应白色），vmax=矩阵最大值（对应红色）
    vmin = 0
    vmax = matrix.max() if matrix.max() > 0 else 1  # 避免最大值为0时色映射异常
    
    # 2. 纵轴上下翻转（矩阵行反转 + 标签反转）
    flipped_matrix = np.flipud(matrix)  # 矩阵行反转：tx=0→热力图最下方
    flipped_yticklabels = tx_values[::-1]  # 标签反转：0在下，20在上
    
    # 3. 绘制热力图（白色→红色渐变）
    sns.heatmap(
        flipped_matrix,  # 反转后的矩阵
        annot=False, 
        fmt="d", 
        cmap=white_red_cmap,  # 白色→红色渐变
        vmin=vmin,  # 0值对应白色
        vmax=vmax,  # 最大值对应红色
        xticklabels=tx_values,  # 横轴：左→右 0→20
        yticklabels=flipped_yticklabels,  # 纵轴：下→上 0→20
        cbar_kws={"label": "Sample Count", "shrink": 0.8}  # 颜色条适配图表大小
    )
    
    # 4. 图表标签（标注颜色和轴序）
    plt.title(f"Total tx Value Cross-Count Heatmap - {m1} vs {m2}", fontsize=16, pad=20)
    plt.xlabel(f"{m2} tx Value (Left→Right: 0→20)", fontsize=14, labelpad=10)
    plt.ylabel(f"{m1} tx Value (Bottom→Top: 0→20)", fontsize=14, labelpad=10)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    
    # 保存热力图
    heatmap_path = os.path.join(output_dir, f"total_{m1}_vs_{m2}_tx_heatmap.png")
    plt.tight_layout()
    plt.savefig(heatmap_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"总体热力图已保存：{heatmap_path}")


# 4. 输出辅助函数（格式化打印矩阵）
def print_3x3_matrix(matrix, row_labels, col_labels, title):
    print(f"\n{title}")
    print(f"{'':<10}" + "".join([f"{col:<10}" for col in col_labels]))
    for i, row in enumerate(matrix):
        print(f"{row_labels[i]:<10}" + "".join([f"{val:<10}" for val in row]))


def print_21x21_matrix(matrix, title):
    print(f"\n{title}（21×21 详细交叉计数，行：前模型 tx 值，列：后模型 tx 值）")
    print("       " + " ".join([f"{i:4d}" if i % 5 == 0 else "    " for i in tx_values]))
    print("       " + "-" * (21 * 5 - 1))
    for i in tx_values:
        if i % 5 == 0:
            row_label = f"{i:4d} | "
        else:
            row_label = "     | "
        print(row_label + " ".join([f"{val:4d}" if j % 5 == 0 else "    " for j, val in enumerate(matrix[i])]))
    print(f"注：行列号每5个显示一次（如列0、5、10...），未显示的序号按递增规律推断")


# 5. 输出每个Q_i的细节结果（放在前面，不生成图片）
labels_3x3 = ["tx=0", "1≤tx≤19", "tx=20"]

for (m1, m2) in combinations(model_names, 2):
    print(f"\n\n===== {m1} vs {m2} 各 Q_i 细节分析 =====")
    # 输出前10个Q_i作为示例
    for q in list(q_range)[:10]:
        q_name = f"Q_{q}"
        print(f"\n----- {q_name} -----")
        print(f"Spearman相关性: {q_corrs[(m1, m2)][q_name]:.4f}")
        print_3x3_matrix(
            q_cross_3x3[(m1, m2)][q_name], 
            labels_3x3, 
            labels_3x3, 
            f"{q_name} 3×3 交叉计数矩阵"
        )
    print(f"\n（注：仅显示前10个Q_i的结果，剩余70个Q_i的细节已写入 {output_file}）")


# 6. 输出总结性结果（放在最后，含总体热力图）
print(f"\n\n\n===== 模型间总体分析总结 =====")
for (m1, m2) in combinations(model_names, 2):
    print(f"\n\n===== {m1} vs {m2} 总体结果 =====")
    # 总体相关性
    print(f"1. 总体 Spearman 相关性系数: {total_corrs[(m1, m2)]:.4f}")
    # 总体3×3矩阵
    print(f"\n2. 总体 3×3 交叉计数矩阵（概览）")
    print_3x3_matrix(
        total_cross_3x3[(m1, m2)], 
        labels_3x3, 
        labels_3x3, 
        f"{m1} vs {m2} 总体3×3矩阵"
    )
    # 总体21×21矩阵（简化显示）
    print(f"\n3. 总体 21×21 详细交叉计数矩阵（tx值0-20）")
    print_21x21_matrix(total_cross_21x21[(m1, m2)], f"{m1} vs {m2} 总体21×21矩阵")
    # 生成并保存总体热力图
    print(f"\n4. 生成总体热力图...")
    plot_total_heatmap(total_cross_21x21[(m1, m2)], m1, m2)


# 最终提示
print(f"\n\n===== 分析全部完成 =====")
print(f"1. 详细分析结果已保存至: {output_file}")
print(f"2. 总体热力图已保存至: {output_dir}（文件名格式：total_模型1_vs_模型2_tx_heatmap.png）")
print(f"3. tx数据缓存文件已保存至: {cache_file}（下次运行将跳过JSON读取，直接加载缓存）")

# 关闭输出重定向
tee.close()