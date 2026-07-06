# 实验映射

## 论文锚点

论文里有两个核心系统：

1. OFA 数据构建：对每个抽样的有害问题生成 75 条大纲填充提示词，每条提示词调用目标模型 20 次，并计算 `ASR=tx/totQA`、`ALR=txl/totQA`。
2. CompSP 排序：训练/评估一个两两比较代理，预测提示词 A 的目标模型 ASR/ALR 是否高于提示词 B，然后用 Bradley-Terry-Luce 聚合 pairwise 结果。

## OFA 数据构建

有效的旧脚本包括：

- `/remote-home/zty/src/prompt.py`：附录里使用的提示词模板。
- `/remote-home/zty/src/make_qa_multimodel.py`：HB/JBB 风格 JSON 问题文件的在线模型 q0/q1/回答生成主版本。
- `/remote-home/zty/src/make_qa_multimodel_grok.py`：Grok/API 版本。
- `/remote-home/zty/src/make_data_tx.py`、`make_data_tx_tp.py`、`make_data_tx_temp.py`：判定和 `tx/txl` 填充版本。
- `/remote-home/zty/src/make_data_a_llama.py`、`make_data_a_llama_temp.py`：本地 vLLM 目标回答生成。
- `/remote-home/zty/src/make_data_q1_gpt_new.py`：重试较重的 q1 生成版本。

清理后的替代实现：

- `src/compsp/ofa_pipeline.py`
- `scripts/run_ofa.py`

阶段映射：

- `init`：为某个问题和轮次区间创建记录。
- `q1`：生成 `q0`、`a0` 和 `q1`。
- `answers`：调用目标模型并填充 `Ans`。
- `judge`：计算 `txl`、判定长回答、计算 `tx`。
- `all`：把 `q1`、`answers`、`judge` 串在一个流程里执行。

## CompSP 两两排序

有效的旧脚本包括：

- `/remote-home/zty/src/q02rank/evaluator.py`：LoRA 分类器对 q1 pair 的推理。
- `/remote-home/zty/src/q02rank/ranker.py`：Bradley-Terry 拟合。
- `/remote-home/zty/src/q02rank/main.py`：两阶段矩阵生成与排序。
- `/remote-home/zty/src/q02rank/config.py`：2026 年版本的 JBB DrAttack/pair 基线和 LoRA 路径映射。
- `/remote-home/zty/src/q02rank/q02mt_all.py`、`mt2rk_all.py`：更早的全流程版本，使用的是 `/home/xzh/...` 路径。
- `/remote-home/zty/src/q02rank/ana_tx.py`：按 top-k 提升 `tx` 的分析。

清理后的替代实现：

- `src/compsp/ranking/evaluator.py`
- `src/compsp/ranking/ranker.py`
- `scripts/run_pairwise_matrix.py`
- `scripts/run_bt_rank.py`

## 论文表格/图

- 表 4-1 的 QSR/IASR/NIR 基线：来自 OFA 输出里的 `txfilled` 文件。
- 表 4-2 的 PASR/PNASR/PALR/PNALR：来自两两比较数据和分类器预测。
- 表 4-3 的 IASR/FASC 提升：来自 `data/saverk` 里的 BTL 排名。
- 图 4-1 长度-有害性关系：`tidy_CompSP/data_ana/模型回答长度分布分析`
- 图 4-2 ASR/ALR 分布：`tidy_CompSP/data_ana/总体含毒比例分析` 及相关脚本。
- 图 4-3 按 ASR 分布的预测准确率：`tidy_CompSP/data_ana/预测成功率随ASR` 和 `预测成功率随ASR/ALR分布`。
- 跨模型相关性：`tidy_CompSP/data_ana/不同模型对同指令ASR相关性` 和 `ASR-ALR跨模型相关性`。
