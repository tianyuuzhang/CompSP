import os
import json
import torch
import numpy as np
from transformers import AutoTokenizer, LlamaForSequenceClassification
from peft import PeftModel, PeftConfig
from tqdm import tqdm
import sys
import gc  # 引入垃圾回收模块

# ----------------------
# 1. 全局配置
# ----------------------
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # 设置使用的GPU编号

# 请根据您的环境修改这些路径
base_model_path = "/remote-home/model/llama-3.1-8B/Llama-3.1-8B"
output_base_dir = "/remote-home/zty/save/ASR_ALR_ACCU" # 最终结果输出路径

# 任务配置
TX_TXL_MODEL_PATHS = {
   # "100_qwen_tx": "/remote-home/model/llama-3.1-8B/llama-3.1-8B-ZTY-classification_qwen_80Q_num75_train60_test20_500_tx_ratio_ge_0_2025-10-15-21",
    #"100_4omini_tx": "/remote-home/model/llama-3.1-8B/llama-3.1-8B-ZTY-classification_4omini_80Q_num75_train60_test20_500_tx_ratio_ge_0_2025-10-28-10",
   # "100_llama_tx": "/remote-home/model/llama-3.1-8B/llama-3.1-8B-ZTY-classification_llama_80Q_num75_train60_test20_500_tx_ratio_ge_0_2025-10-27-09",
   # "100_qwen_txl": "/remote-home/model/llama-3.1-8B/llama-3.1-8B-ZTY-classification_qwen_80Q_num75_train60_test20_500_txl_ratio_ge_0_2025-11-15-04",
    #"100_4omini_txl": "/remote-home/model/llama-3.1-8B/llama-3.1-8B-ZTY-classification_4omini_80Q_num75_train60_test20_500_txl_ratio_ge_0_2025-11-16-20", 
    "100_llama_txl": "/remote-home/model/llama-3.1-8B/llama-3.1-8B-ZTY-classification_llama_80Q_num75_train60_test20_500_txl_ratio_ge_0_2025-11-16-04",  
}

# 任务对应的测试文件路径
TX_TXL_TEST_PATHS = {
   # "100_qwen_tx": "/remote-home/xzh/for_zty/gpt3dot5_data_test/qwen/trainData/train60_test20_sets_Q80_num100_500_tx_ge_0.2_random/80Q_num100_train60_test20_500_tx_ratio_ge_0.2_random_t20_c500_test_10000pairs.json",
    #"100_4omini_tx": "/remote-home/xzh/for_zty/gpt3dot5_data_test/4omini/trainData/train60_test20_sets_Q80_num75_500_tx_ge_0.2_random/80Q_num75_train60_test20_500_tx_ratio_ge_0.2_random_t20_c500_test_8500pairs.json",
    #"100_llama_tx": "/remote-home/xzh/for_zty/gpt3dot5_data_test/llama/trainData/train60_test20_sets_Q80_num75_500_tx_ge_0.2_random/80Q_num75_train60_test20_500_tx_ratio_ge_0.2_random_t20_c500_test_9500pairs.json",
    #"100_qwen_txl": "/remote-home/xzh/for_zty/gpt3dot5_data_test/qwen/trainData/qwen_train60_test20_sets_Q80_num75_500_txl_ge_0.2_random/qwen_80Q_num75_train60_test20_500_txl_ratio_ge_0.2_random_t20_c500_test_6000pairs.json",
   # "100_4omini_txl": "/remote-home/xzh/for_zty/gpt3dot5_data_test/4omini/trainData/4omini_train60_test20_sets_Q80_num75_500_txl_ge_0.2_random/4omini_80Q_num75_train60_test20_500_txl_ratio_ge_0.2_random_t20_c500_test_4000pairs.json",
    "100_llama_txl": "/remote-home/xzh/for_zty/gpt3dot5_data_test/llama/trainData/llama_train60_test20_sets_Q80_num75_500_txl_ge_0.2_random/llama_80Q_num75_train60_test20_500_txl_ratio_ge_0.2_random_t20_c500_test_8500pairs.json",
}

# ----------------------
# 2. 模型加载和配置函数
# ----------------------
def load_model_and_tokenizer(lora_base_path: str, checkpoint_subdir: str = None):
    """
    加载基础模型和 LoRA 适配器。
    """
    print(f"-> 正在加载基础模型: {base_model_path}")
    
    if checkpoint_subdir:
        lora_adapter_path = os.path.join(lora_base_path, checkpoint_subdir)
    else:
        lora_adapter_path = os.path.join(lora_base_path, "lora")
    
    # 加载基础模型
    try:
        # device_map="auto" 会自动分配设备，不要再手动调用 .to(device)
        model = LlamaForSequenceClassification.from_pretrained(
            base_model_path,
            num_labels=2,
            device_map="auto", 
            use_cache=False
        )
    except Exception as e:
        print(f"错误: 基础模型加载失败: {e}")
        sys.exit(1)

    # 加载分词器
    try:
        tokenizer = AutoTokenizer.from_pretrained(lora_base_path)
    except Exception as e:
        print(f"警告: Tokenizer从LoRA路径加载失败，尝试从基础模型加载。错误: {e}")
        tokenizer = AutoTokenizer.from_pretrained(base_model_path)

    model.resize_token_embeddings(len(tokenizer)+1)
    
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.config.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.convert_tokens_to_ids('[PAD]')
    
    print(f"-> 正在加载 LoRA 适配器: {lora_adapter_path}")
    try:
        if not os.path.exists(lora_adapter_path) and os.path.exists(os.path.join(lora_adapter_path, 'lora')):
             lora_adapter_path = os.path.join(lora_adapter_path, 'lora')
        
        model = PeftModel.from_pretrained(model, lora_adapter_path)
    except Exception as e:
        print(f"错误: LoRA 适配器加载失败（{lora_adapter_path}）: {e}")
        sys.exit(1)

    # 【重要修改】移除 model.to(device) 和 model.eval()，PeftModel 加载后默认状态通常没问题，
    # device_map="auto" 会自动处理设备。为了确保是评估模式：
    model.eval()
    
    print("模型加载完成。")
    return model, tokenizer

# ----------------------
# 3. 预测函数
# ----------------------
def predict_probability(model, tokenizer, formatted_text: str) -> dict:
    """
    对拼接好的输入文本进行比较。
    """
    if not formatted_text:
        return {"prob_A_is_better": 0.5, "prob_B_is_better": 0.5}

    # 指定 device="cuda"，因为我们设置了 CUDA_VISIBLE_DEVICES
    device = "cuda"
    
    # 将输入数据移动到 GPU
    inputs = tokenizer(formatted_text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
    
    with torch.no_grad():
        logits = model(**inputs).logits
        probabilities = torch.softmax(logits, dim=1).squeeze()
        
        # 确保是一个标量而不是单元素tensor
        if probabilities.dim() == 0:
             # 处理 batch_size=1 且 squeeze 把维度全挤掉的情况（虽然二分类通常有两个输出）
             # 实际上 softmax 后应该是 [prob0, prob1]
             pass 

        prob_a_better = probabilities[1].item()
        prob_b_better = probabilities[0].item()
        
    return {
        "prob_A_is_better": prob_a_better,
        "prob_B_is_better": prob_b_better
    }


# ----------------------
# 4. 显存清理函数
# ----------------------
def clear_gpu_memory(model=None, tokenizer=None):
    """清理显存和内存"""
    if model is not None:
        del model
    if tokenizer is not None:
        del tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    print("-> 显存已清理")


# ----------------------
# 5. 主流程
# ----------------------
def main():
    os.makedirs(output_base_dir, exist_ok=True)
    task_keys = list(TX_TXL_MODEL_PATHS.keys())

    for task_key in tqdm(task_keys, desc="Overall Task Progress"):
        print(f"\n{'='*50}")
        print(f"--- 正在处理任务: {task_key} ---")
        print(f"{'='*50}")

        model_base_path = TX_TXL_MODEL_PATHS[task_key]
        test_file_path = TX_TXL_TEST_PATHS[task_key]
        output_path = os.path.join(output_base_dir, f"eval_results_{task_key}.json")
        
        # 如果结果文件已存在，可以选择跳过（可选）
        # if os.path.exists(output_path):
        #     print(f"文件 {output_path} 已存在，跳过。")
        #     continue

        checkpoint_subdir = None
        if "4omini_txl" in task_key:
            checkpoint_subdir = "checkpoint/checkpoint-4500"
        elif "llama_txl" in task_key:
            checkpoint_subdir = "checkpoint/checkpoint-5157"
        
        # 2. 加载模型和分词器
        try:
            model, tokenizer = load_model_and_tokenizer(model_base_path, checkpoint_subdir)
        except Exception as e:
            tqdm.write(f"任务 {task_key} 模型加载失败，跳过。错误: {e}")
            continue

        # 3. 加载测试数据
        try:
            with open(test_file_path, 'r', encoding='utf-8') as f:
                test_data = json.load(f)
            tqdm.write(f"已加载 {len(test_data)} 条测试数据")
        except Exception as e:
            tqdm.write(f"读取测试文件失败: {e}")
            clear_gpu_memory(model, tokenizer)
            continue
        
        # 4. 进行预测
        results = []
        for item in tqdm(test_data, desc=f"Predicting {task_key}", leave=False):
            formatted_text = item.get('text')
            if not formatted_text:
                continue

            try:
                probabilities = predict_probability(model, tokenizer, formatted_text)
                
                item_result = item.copy()
                item_result['evaluation_results'] = probabilities
                item_result['predicted_label'] = 1 if probabilities['prob_A_is_better'] > 0.5 else 0
                results.append(item_result)
            
            except Exception as e:
                tqdm.write(f"预测错误: {e}")
                continue

        # 5. 保存结果
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            tqdm.write(f"结果已保存至: {output_path}")
        except Exception as e:
            tqdm.write(f"保存失败: {e}")

        # 6. 【关键步骤】清理显存，为下一个循环做准备
        clear_gpu_memory(model, tokenizer)

    print(f"\n{'='*50}")
    print("所有任务处理完成！")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
