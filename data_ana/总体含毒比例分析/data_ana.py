import os
import json
from collections import Counter
from tqdm import tqdm

def analyze_toxicity_reports():
    """
    遍历指定目录结构，统计JSON文件中的毒性分析结果。
    """
    # --- 配置 ---
    # 请将此路径修改为包含 Q_0, Q_1, ... 文件夹的根目录
    BASE_PATH = "/remote-home/xzh/for_zty/gpt3dot5_data_test/qwen/save" 
    
    # --- 初始化所有统计变量 ---
    
    # 任务1: 至少包含一个 "unsafe" 的问题总数
    questions_with_any_unsafe = set()

    # 任务2: 每个问题中，至少包含一个 "unsafe" 的 round 数量
    unsafe_rounds_per_question = Counter()

    # 任务3: 所有问题中，至少包含一个 "unsafe" 的 round 总数
    total_unsafe_rounds = 0

    # 任务5: 至少包含一个 "unsafe" 的 Ans 元素总数 (用户跳过了4)
    total_unsafe_ans_items = 0

    # 任务6: 'toxicA' 列表内容的分布情况
    toxicA_distribution = Counter()
    
    # 任务7: 为每个问题收集最多3个包含 "unsafe" 的 Round 示例
    unsafe_round_examples = {}

    # --- 新增: 为每个问题收集一个 "unsafe" Ans 项的示例 ---
    single_unsafe_ans_examples = {}

    print("开始分析数据...")

    # 遍历 80 个问题文件夹 (Q_0 to Q_79)
    for question_id in tqdm(range(80), desc="处理问题文件夹"):
        folder_path = os.path.join(BASE_PATH, f"Q_{question_id}")
        
        filename = f"Q_r_{question_id}_0_100_txfilled.json"
        json_file_path = os.path.join(folder_path, filename)

        if not os.path.exists(json_file_path):
            continue

        try:
            with open(json_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except json.JSONDecodeError:
            print(f"错误: JSON 文件格式错误，跳过 -> {json_file_path}")
            continue
        
        # 遍历文件中的每一个 round (0-99)
        for round_data in data:
            round_has_unsafe_flag = False
            
            ans_list = round_data.get("Ans", [])
            if not isinstance(ans_list, list):
                continue

            # 遍历一个 round 中的每一个 Ans 项
            for ans_item in ans_list:
                toxicA = ans_item.get("toxicA", [])
                
                # --- 任务6: 统计 toxicA 分布 ---
                if not toxicA:
                    toxicA_distribution["empty"] += 1
                else:
                    unsafe_count = toxicA.count("unsafe")
                    safe_count = toxicA.count("safe")
                    
                    if unsafe_count == 0:
                        toxicA_distribution[f"{safe_count}_safe_0_unsafe"] += 1
                    elif unsafe_count == 1:
                        toxicA_distribution[f"{safe_count}_safe_1_unsafe"] += 1
                    else:
                        toxicA_distribution[f"anomalous_{safe_count}_safe_{unsafe_count}_unsafe"] += 1
                
                # 如果 toxicA 列表中含有 "unsafe"
                if "unsafe" in toxicA:
                    questions_with_any_unsafe.add(question_id)
                    total_unsafe_ans_items += 1
                    round_has_unsafe_flag = True

                    # --- 新增: 收集单个 Ans 示例 ---
                    # 如果还没有为这个问题收集过示例，就收集这第一个
                    if question_id not in single_unsafe_ans_examples:
                        single_unsafe_ans_examples[question_id] = {
                            "question": round_data.get("question", "N/A"),
                            "answer_text": ans_item.get("A", "N/A")
                        }

            # 在检查完一个 round 的所有 Ans 项后，更新 round 级别的统计和示例
            if round_has_unsafe_flag:
                unsafe_rounds_per_question[question_id] += 1
                total_unsafe_rounds += 1
                
                if question_id not in unsafe_round_examples:
                    unsafe_round_examples[question_id] = []
                
                if len(unsafe_round_examples[question_id]) < 3:
                    unsafe_round_examples[question_id].append(round_data)
    
    # --- 构建统计报告字符串 ---
    report_lines = []
    report_lines.append("="*50)
    report_lines.append("                毒性分析统计报告")
    report_lines.append("="*50 + "\n")

    report_lines.append(f"1. 至少包含一个'unsafe'的问题总数: {len(questions_with_any_unsafe)} / 80 个问题")
    report_lines.append("-" * 50)
    
    report_lines.append("2. 每个问题中包含'unsafe'的Round数量:")
    if not unsafe_rounds_per_question:
        report_lines.append("   所有80个问题中均未发现包含'unsafe'的Round。")
    else:
        for q_id in range(80):
            count = unsafe_rounds_per_question.get(q_id, 0)
            report_lines.append(f"   - 问题 Q_{q_id}: {count} 个 rounds")
    report_lines.append("-" * 50)

    report_lines.append(f"3. 所有问题中包含'unsafe'的Round总数: {total_unsafe_rounds} / {80*100} 个 rounds")
    report_lines.append("-" * 50)
    
    report_lines.append(f"5. 所有Ans项中包含'unsafe'的总数: {total_unsafe_ans_items} / {80*100*20} 个 Ans 项")
    report_lines.append("-" * 50)

    report_lines.append("6. 'toxicA' 标签分布情况:")
    if not toxicA_distribution:
        report_lines.append("   未处理任何 Ans 项。")
    for category, count in sorted(toxicA_distribution.items()):
        report_lines.append(f"   - {category}: {count} 次")
    report_lines.append("-" * 50)

    final_report_string = "\n".join(report_lines)
    
    # --- 打印统计报告到控制台 ---
    print("\n" + final_report_string)

    # --- 新增: 将统计报告写入文件 ---
    summary_filename = "toxicity_summary_report.txt"
    try:
        with open(summary_filename, 'w', encoding='utf-8') as f:
            f.write(final_report_string)
        print(f"\n统计报告已成功写入文件: {summary_filename}")
    except Exception as e:
        print(f"错误: 写入统计报告文件失败: {e}")

    # --- 将 Round 示例写入文件 ---
    example_filename = "unsafe_round_examples.txt"
    try:
        with open(example_filename, 'w', encoding='utf-8') as f:
            f.write("每个问题最多3个包含'unsafe'的Round示例\n")
            f.write("="*80 + "\n\n")
            
            if not unsafe_round_examples:
                f.write("没有找到任何包含'unsafe'评级的Round。\n")
            
            for q_id in range(80):
                f.write(f"--- 问题 Q_{q_id} ---\n\n")
                if q_id in unsafe_round_examples:
                    examples = unsafe_round_examples[q_id]
                    if examples:
                        f.write(f"问题内容: {examples[0].get('question', 'N/A')}\n\n")
                    for i, round_example in enumerate(examples):
                        f.write(f"示例 #{i+1} (完整 Round 内容):\n")
                        f.write(json.dumps(round_example, indent=4, ensure_ascii=False))
                        f.write("\n\n")
                else:
                    f.write("未找到该问题的'unsafe' Round示例。\n\n")
        
        print(f"Round示例已成功写入文件: {example_filename}")
    except Exception as e:
        print(f"错误: 写入Round示例文件失败: {e}")

    # --- 新增: 将单个 Ans 示例写入文件 ---
    single_ans_example_filename = "single_unsafe_ans_examples.txt"
    try:
        with open(single_ans_example_filename, 'w', encoding='utf-8') as f:
            f.write("每个问题首个'unsafe'回答示例\n")
            f.write("="*80 + "\n\n")

            if not single_unsafe_ans_examples:
                f.write("没有找到任何包含'unsafe'评级的回答。\n")

            for q_id in range(80):
                f.write(f"--- 问题 Q_{q_id} ---\n\n")
                if q_id in single_unsafe_ans_examples:
                    example = single_unsafe_ans_examples[q_id]
                    f.write(f"问题内容: {example['question']}\n\n")
                    f.write("回答内容:\n")
                    f.write(example['answer_text'])
                    f.write("\n\n")
                else:
                    f.write("未找到该问题的'unsafe'回答示例。\n\n")
        
        print(f"单个Ans示例已成功写入文件: {single_ans_example_filename}")
    except Exception as e:
        print(f"错误: 写入单个Ans示例文件失败: {e}")

    print("\n分析完成。")


# 运行主函数
if __name__ == "__main__":
    analyze_toxicity_reports()
