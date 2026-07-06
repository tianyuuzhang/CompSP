# data

这个清洗后的目录按用途组织数据。大体量生成产物保留在原位，并由 `docs/data_index.md` 引用。

## 记录格式

OFA 流水线读写 JSON 列表。每个条目的结构如下：

```json
{
  "id": 0,
  "questionID": 9,
  "question": "...",
  "round": 0,
  "totQA": 20,
  "Ans": [
    {
      "A": "model response",
      "len": 1234,
      "toxic": "safe|unsafe|N/A",
      "toxicA": ["safe", "unsafe"],
      "time": "YYYY-MM-DD HH:MM:SS"
    }
  ],
  "q0": "outline-generation prompt",
  "a0": "generated outline",
  "q1": "fill-outline attack prompt",
  "txl": 0,
  "tx": 0
}
```

- `txl`：长回复数量，对应 ALR 分子。
- `tx`：被判定为 unsafe 的回复数量，对应 ASR 分子。
- `totQA`：ASR/ALR 的分母，论文实验中通常为 20。

## 已知来源输入

- `/remote-home/zty/data/jailbreak_questions.json`
- `/remote-home/zty/data/harmbench_questions.json`

## ASR/ALR 数据集注册表

- `asr数据`：人工维护的路径索引。
- `asr_datasets.json`：机器可读的修正路径注册表。
- Python 接口：`src/compsp/asr_registry.py` 和 `src/compsp/asr_dataset.py`。
