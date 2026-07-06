# 源码审计

## 已纳入并重构

- `prompt.py`：核心提示词模板，已重构为 `src/compsp/prompts.py`。
- `make_qa_multimodel.py`：最有代表性的在线 OFA 生成脚本，已重构为 `src/compsp/ofa_pipeline.py`。
- `make_data_tx*.py`：安全判定和 `txfilled` 生成脚本，已重构为 `src/compsp/ofa_pipeline.py` 的 `judge` 阶段。
- `make_data_a_llama*.py`：本地模型回答生成脚本。原脚本保留为溯源；在线回答生成已经重写，本地 vLLM 后续也可以挂到同一套 `answers` 接口后面。
- `q02rank/`：两两代理推理和排序，已重构为 `src/compsp/ranking`。

## 纳入但只做索引

- `/remote-home/zty/src/rubbish/train_code` 和 `/remote-home/zty/src/rubbish/train_logs`：大概率是 LoRA 训练和日志溯源。因为你已经排除了 `rubbish`，所以没有搬进清理后的主流程。
- `/remote-home/xzh/for_zty/reproduce/...`：`q02rank/config.py` 引用的外部复现数据，只保留为外部数据索引。
- `/home/xzh/model/...`：旧版 `q02rank/q02mt_all.py` 里的过时路径，已被 `/remote-home/model/...` 和 `/remote-home/zty/tidy_CompSP/data/saverk` 取代。

## 已过时或探索性

- `make_data_qa.py`、`make_data_qa6.py`、`make_data_qa_ds.py`：早期的一次性流水线，绑定 CSV 输入和硬编码问题编号。
- `make_data_q1_gpt.py`、`make_data_q1_ds.py`：早期的 q1-only 生成脚本。
- `llama运行示例.py`：本地 vLLM smoke test。
- `function.py`：只提供时间戳格式化，已被标准库时间格式取代。

## 重要清理决定

- 没有复制旧 API key。
- 原始代码和数据 `/remote-home/zty/src`、`/remote-home/zty/data`、`/remote-home/zty/save` 都没有被修改。
- 硬编码路径已经改成了 CLI 参数。
