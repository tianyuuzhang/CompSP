# ranking

pairwise CompSP 推理和排序工具。

- `evaluator.py`：加载 LoRA 序列分类器并生成 pairwise prompt 优势矩阵。
- `ranker.py`：根据 pairwise 矩阵拟合 Bradley-Terry 分数并导出排序后的 prompt ID。

实现口径：

- 输入记录使用 `q1` 字段，pairwise prompt 格式保持为旧 `q02rank` 的 `Text A / Text B` 模板。
- `start` 和 `end` 是闭区间；例如 `0_24` 表示 25 条候选指令。
- tokenizer 优先从 LoRA 路径读取；若 LoRA 目录没有 tokenizer 文件，则回退到 base model 路径。这与旧 `/remote-home/zty/src/q02rank/evaluator.py` 的行为一致。
- 若 adapter 中保存了 `embed_tokens.weight`，模型会按 adapter 的 embedding 行数 resize。旧 pair_llama LoRA 的 embedding 行数为 `128257`，比 base tokenizer 长度 `128256` 多 1，因此不能简单使用 `len(tokenizer)`。
- BTL 输出中的 `item_id` 等于原始 JSON 列表下标，供 `data/saverk` 和白盒对齐脚本使用。
