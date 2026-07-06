# whitebox

这里存放与后续“安全结构”方向相关的白盒分析工具。

- `activations.py`：加载 Llama 类因果语言模型，并提取每层 last-token 与 mean hidden state。
- `directions.py`：根据 harmful/benign 激活差构造方向，并将提示词激活投影到这些方向上。
- `ranking_io.py`：读取 `data/saverk` 中已有的 CompSP BTL 排名，用于和白盒方向分数对齐。

实现逻辑：先用少量 harmful 与 benign anchor 构造逐层方向，再对 Llama 相关 `q1` 数据打分，最后与同一数据集、同一问题、同一 `item_index` 的 CompSP 排名比较。注意：Qwen 结果不能作为 Llama 白盒实验的标签。
