import json
import os
import csv

def analyze_ratio_distribution(json_file_path):
    # 初始化21x21的表格，用于统计各比例区间的数量
    # 比例范围是0.00到1.00，步长为0.05，共21个区间
    count_table = [[0 for _ in range(21)] for _ in range(21)]
    
    # 读取JSON文件
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"读取文件时出错: {e}")
        return None
    
    # 确保数据是列表形式
    if not isinstance(data, list):
        print("JSON数据格式不正确，应为列表")
        return None
    
    # 统计每个比例对出现的次数
    total_count = len(data)
    for item in data:
        try:
            q1_ratio = item.get('q1_ratio', 0)
            q2_ratio = item.get('q2_ratio', 0)
            
            # 将比例转换为表格索引（0-20）
            # 四舍五入到最近的0.05倍数来确定区间
            q1_index = round(q1_ratio / 0.05)
            q2_index = round(q2_ratio / 0.05)
            
            # 确保索引在有效范围内
            if 0 <= q1_index <= 20 and 0 <= q2_index <= 20:
                count_table[q1_index][q2_index] += 1
        except Exception as e:
            print(f"处理数据项时出错: {e}")
            continue
    
    # 计算每行和每列的总数量
    row_totals = [sum(row) for row in count_table]
    col_totals = [sum(col) for col in zip(*count_table)]
    
    # 保存结果到CSV文件
    output_file = os.path.splitext(json_file_path)[0] + "_ratio_distribution.csv"
    try:
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # 写入表头（列总计和比例）
            header = ['q1\\q2']
            for j in range(21):
                col_ratio = f"{col_totals[j]}/{total_count}" if total_count > 0 else "0/0"
                header.append(f'{j*0.05:.2f}\n({col_ratio})')
            writer.writerow(header)
            
            # 写入数据行（包含行总计和比例）
            for i in range(21):
                row_ratio = f"{row_totals[i]}/{total_count}" if total_count > 0 else "0/0"
                row_data = [f'{i*0.05:.2f}\n({row_ratio})']
                
                # 为每个单元格计算比例（x/20格式）
                for j in range(21):
                    if total_count > 0:
                        # 计算比例并转换为x/20的形式
                        ratio = count_table[i][j]
                        x = round(ratio * 20)
                        cell_value = f"{x}"
                    else:
                        cell_value = "0"
                    row_data.append(cell_value)
                
                writer.writerow(row_data)
        
        print(f"分析完成，结果已保存到: {output_file}")
        return output_file
    except Exception as e:
        print(f"保存结果时出错: {e}")
        return None

if __name__ == "__main__":
    # 请将此处替换为你的JSON文件路径
    json_file = "/remote-home/xzh/for_zty/gpt3dot5_data_test/qwen/train60_test20_sets_Q80_num100_300_txl_ge_02_512/80Q_num100_train60_test20_300_txl_ratio_ge_0.2_512_t20_c500_test_7500pairs.json"  # 输入你的JSON文件名
    
    # 检查文件是否存在
    if not os.path.exists(json_file):
        print(f"文件不存在: {json_file}")
    else:
        analyze_ratio_distribution(json_file)
    