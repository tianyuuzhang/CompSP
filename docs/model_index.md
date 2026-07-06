# 模型索引

## 基座模型

- `/remote-home/model/llama-3.1-8B/Llama-3.1-8B`：CompSP 序分类的基座模型。
- `/remote-home/model/llama-3.1-8B/Llama-3.1-8B-Instruct`：早期脚本里用到的本地目标生成模型。

## 当前配置里观察到的 CompSP LoRA 适配器

- `/remote-home/model/llama-3.1-8B/llama-3.1-8B-ZTY-classification_qwen_80Q_num75_train60_test20_500_tx_ratio_ge_0_2026-04-04-15/lora`
- `/remote-home/model/llama-3.1-8B/llama-3.1-8B-ZTY-classification_Drattack_qwen_80Q_num75_train60_test20_500_tx_ratio_ge_0_2026-04-26-16/lora`
- `/remote-home/model/llama-3.1-8B/llama-3.1-8B-ZTY-classification_Drattack_llama_80Q_num75_train60_test20_500_tx_ratio_ge_0_2026-05-01-14/checkpoint/checkpoint-3626`
- `/remote-home/model/llama-3.1-8B/llama-3.1-8B-ZTY-classification_Drattack_4omini_80Q_num75_train60_test20_500_tx_ratio_ge_0_2026-05-22-11/lora`
- `/remote-home/model/llama-3.1-8B/llama-3.1-8B-ZTY-classification_pair_llama_10Q_num75_train60_test20_500_tx_ratio_ge_0_2026-05-08-11/lora`
- `/remote-home/model/llama-3.1-8B/llama-3.1-8B-ZTY-classification_4omini_80Q_num75_train60_test20_500_tx_ratio_ge_0_2026-05-04-11/lora`

这些模型目录都是外部依赖，不会被复制进 `tidy_CompSP`。
