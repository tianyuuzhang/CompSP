# compsp

清洁版 CompSP 论文工作流对应的 Python 包。

- `ofa_pipeline.py`：负责 OFA 数据生成和安全判定。
- `prompts.py`：保存论文实验使用的 prompt 模板。
- `clients.py`：封装 OpenAI 兼容 API 访问，不写死密钥。
- `ranking/`：保存 CompSP pairwise 推理和 BTL 排序逻辑。
- `whitebox/`：保存白盒激活提取、方向构造和排序对齐工具。
