# tidy_CompSP

这里是 CompSP 论文代码整理后的研究工作区。

## 主要流程

1. OFA 数据构建：`q0 -> a0/q1 -> target answers -> txfilled`
2. CompSP 排序：`q1 pairs -> pairwise matrix -> Bradley-Terry scores -> top-k prompts`
3. 分析与论文产物索引：见 `docs/`

`/remote-home/zty/src` 和 `/remote-home/zty/data` 下的原始文件都视为只读来源。旧 `config.json` 里的 API 密钥没有被复制；请改用环境变量。

## 目录结构

- `src/compsp/`：可导入的 Python 包。
- `scripts/`：轻量级命令行入口。
- `configs/`：示例配置和命令片段。
- `data/`：少量复制/索引后的输入与说明文件；大体量生成数据保留索引。
- `docs/`：源码审计、实验映射、模型/数据索引。
- `src/compsp/asr_dataset.py`：训练和评估代码使用的标准 ASR/ALR 数据接口。

## 环境

```bash
cd /remote-home/zty/tidy_CompSP
export PYTHONPATH=$PWD/src
export OPENAI_API_KEY=...
export OPENAI_BASE_URL=...
```

可选依赖会随阶段而变化：

- OFA 在线生成/判定：`aiohttp`
- CompSP 模型推理：`torch`、`transformers`、`peft`
- 仅做排序：`numpy`

## 最小示例

为一个 JBB 问题初始化 75 条 OFA 记录：

```bash
python scripts/run_ofa.py \
  --stage init \
  --questions /remote-home/zty/data/jailbreak_questions.json \
  --question-id 9 \
  --start 0 \
  --end 75 \
  --tot-qa 20 \
  --output data/work/Q_9_0_75_init.json
```

生成大纲和 q1 提示词：

```bash
python scripts/run_ofa.py \
  --stage q1 \
  --input data/work/Q_9_0_75_init.json \
  --output data/work/Q_9_0_75_q1.json \
  --outline-model gpt-3.5-turbo-0125
```

从已有的 pairwise matrix 运行 BTL 排序：

```bash
python scripts/run_bt_rank.py \
  --matrix data/saverk/save_jbb6100_pair_qwen/Q_9/Q_9_0_24pair.json \
  --q-id 9 \
  --start-id 0 \
  --output data/work/Q_9_0_24_ranking_scores.json
```
