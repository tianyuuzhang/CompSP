# src

论文工作流对应的可导入清洁代码。

- `compsp/ofa_pipeline.py`：从原始危险问题到 `txfilled` 记录的 OFA 数据流程。
- `compsp/prompts.py`：用于大纲生成、填充指令和安全判定的论文提示词模板。
- `compsp/clients.py`：使用环境变量的 OpenAI 兼容异步 API 客户端。
- `compsp/ranking/evaluator.py`：LoRA 序列分类 pairwise 推理。
- `compsp/ranking/ranker.py`：Bradley-Terry 分数拟合和排序导出。
- `compsp/asr_registry.py`：ASR/ALR `txfilled` 数据集的标准路径注册表。
- `compsp/asr_dataset.py`：标准化加载器、校验器和 pairwise 样本构造器。
- `compsp/whitebox/`：激活提取、方向打分和排序对齐辅助工具。

除非通过 CLI 参数传入，代码尽量避免写死绝对实验路径。
