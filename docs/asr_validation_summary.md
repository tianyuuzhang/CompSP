# ASR 数据校验汇总

校验命令：

```bash
cd /remote-home/zty/tidy_CompSP
export PYTHONPATH=$PWD/src
python scripts/validate_asr_data.py --sample-items -1 --output docs/asr_validation_report_full.json
```

生成的报告：

- `docs/asr_validation_report_full.json`

## 结果

经过规范化后，核心 JSON schema 是一致的：

- 核心字段：`questionID`、`question`、`round`、`totQA`、`Ans`、`q1`、`txl`、`tx`
- OFA 特有字段：`q0`、`a0`
- Pair 特有附加字段：`target`
- DrAttack 文件没有 `q0/a0`
- Pair 文件里的 `questionID` 可能是字符串，例如 `Q_0`；加载器会把它转成整数

## 路径修正

- `jbb-deepseek-ofa` 实际用的是 `Q_{i}_0_100_txfilled.json`，不是 `Q_r_{i}_0_100_txfilled.json`
- `jbb-llama-pair` 和 `jbb-4omini-pair` 在 `/remote-home/zty/tidy_CompSP/data/asr数据` 里的原始路径有重复后缀；机器注册表里用的是修正后的模板

## 不完整或缺失的文件

- `hb-4omini-ofa`：`Q_318` 只有 93 条记录，`Q_364` 只有 99 条
- `hb-llama-ofa`：`Q_318` 只有 93 条记录，`Q_364` 只有 99 条
- `jbb-qwen-drattack`：有 80 个文件，缺少 `Q_80..Q_99`
- `jbb-4omini-drattack`：有 80 个文件，缺少 `Q_80..Q_99`
- `jbb-llama-drattack`：有 80 个文件，缺少 `Q_80`
- `jbb-qwen-pair`：有 80 个文件，缺少 `Q_80..Q_99`；`Q_61` 只有 78 条记录，`Q_63` 只有 83 条
- `jbb-llama-pair`：有 80 个文件，缺少 `Q_80..Q_99`；`Q_61` 只有 78 条记录，`Q_63` 只有 83 条
- `jbb-4omini-pair`：有 81 个文件，缺少 `Q_81..Q_99`；`Q_61` 只有 78 条记录，`Q_63` 只有 83 条

## 实用建议

做训练/评估时，最好用 `discover_question_ids()` 加文件存在性检查，而不是假设每个数据集一定有 100 个问题文件。做 pairwise 评估时，还要跳过那些记录数不足以覆盖候选范围的文件。
