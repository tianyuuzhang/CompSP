# ASR 数据接口

人工维护的路径索引是 `/remote-home/zty/tidy_CompSP/data/asr数据`。
代码维护的注册表是 `src/compsp/asr_registry.py`。

## 已注册数据集键

当前注册表包含：

- JBB + OFA：`jbb-4omini-ofa`、`jbb-deepseek-ofa`、`jbb-llama-ofa`、`jbb-qwen-ofa`
- HB + OFA：`hb-4omini-ofa`、`hb-deepseek-ofa`、`hb-llama-ofa`、`hb-qwen-ofa`
- JBB + DrAttack：`jbb-qwen-drattack`、`jbb-4omini-drattack`、`jbb-llama-drattack`
- JBB + Pair：`jbb-qwen-pair`、`jbb-llama-pair`、`jbb-4omini-pair`

在我检查时，索引里没有列出 HB 的 DrAttack 或 Pair 型 `txfilled` 数据。

## 标准记录

加载器会把每条 JSON 记录规范化为 `ASRRecord`：

- `question_id`：原始危险问题编号。
- `question`：原始危险问题。
- `round`：文件内的攻击指令编号。
- `q1`：真正送去目标模型的最终攻击提示词。
- `tx`：ASR 分子，所以 `ASR = tx / tot_qa`。
- `txl`：ALR 分子，所以 `ALR = txl / tot_qa`。
- `tot_qa`：目标模型访问次数，通常是 20。
- `answers`：目标模型原始回答与判定结果。
- `q0`、`a0`、`target`：可选元数据。OFA 有 `q0/a0`；DrAttack 目前没有；Pair 多一个 `target`。

注意：论文里长回答指标叫 `ALR`，不是 `ASL`。
有些 Pair 文件里 `questionID` 是字符串，例如 `Q_0`；加载器会把它统一成整数 `0`。

## 答案格式

每个 `Ans` 元素应包含：

- `A`：目标模型完整回答。
- `len`：回答长度。
- `toxic`：最终安全标签，通常是 `safe`、`unsafe` 或判定前的 `N/A`。
- `toxicA`：判定投票或判定报错信息列表。
- `time`：时间戳。

## 接口

```python
from compsp.asr_dataset import iter_records, make_pairwise_examples

records = list(iter_records("jbb-qwen-ofa", question_ids=[9]))
pairs = make_pairwise_examples(records, metric="asr", min_delta=0.15)
```

## 校验

```bash
cd /remote-home/zty/tidy_CompSP
export PYTHONPATH=$PWD/src
python scripts/validate_asr_data.py --max-files 3
python scripts/validate_asr_data.py --sample-items -1 --output docs/asr_validation_report.json
```

校验器会检查路径是否存在、JSON 顶层类型、记录数量、核心字段、答案字段、`totQA` 以及 `tx/txl` 范围。

## 已知路径修正

- `jbb-deepseek-ofa` 实际用的是 `Q_{i}_0_100_txfilled.json`，不是 `Q_r_{i}_0_100_txfilled.json`。
- `jbb-llama-pair` 和 `jbb-4omini-pair` 在人工索引里有重复后缀；注册表里已经改成修正后的路径。
