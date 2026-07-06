# 归档数据说明

本目录保存不适合以原始 JSON 形态直接提交的大型高价值数据。归档策略是：保留原始数据内容，使用 `tar.gz` 压缩，并用 `split` 切成小于 GitHub 100 MB 单文件限制的分片。这样可以避免把约 8 GiB 原始 JSON 直接写入 Git 历史，同时仍然让数据随仓库可恢复。

## 内容

- `大纲拆分/`：`data/大纲拆分` 的压缩快照，包含 OFA 从 `q0/a0/q1` 到 `txfilled` 相关的整理数据。
- `asr真实数据/`：14 组真实 ASR/ALR 结果数据的压缩快照。每组对应 `data/asr_datasets.json` 中的一个 key，例如 `jbb-llama-ofa`、`jbb-qwen-pair`。
- `archive_build.log`：生成归档时的日志。

## 分片格式

每个数据集目录包含：

- `*.tar.gz.part000`、`*.tar.gz.part001` 等：按顺序切分的压缩包分片。
- `*.tar.gz.sha256`：完整压缩包的 SHA256 校验值。
- `*.parts.sha256`：每个分片的 SHA256 校验值。

## 恢复方式

在仓库根目录运行：

```bash
bash scripts/解包归档数据.sh
```

默认会把数据解包到 `data/恢复数据/`，不会覆盖现有原始目录。如需指定输出目录：

```bash
bash scripts/解包归档数据.sh /path/to/output
```

恢复后的 ASR 数据每个目录中包含 `数据清单.json`，记录来源路径模板、题号覆盖范围、数据集、攻击方法和目标模型。

## 注意

这些归档是为了版本保存和迁移，不建议训练或评估脚本直接读取分片。实验代码应优先通过 `data/asr_datasets.json` 或恢复后的 JSON 文件读取数据。
