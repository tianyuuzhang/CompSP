# 白盒 / CompSP 误差对齐实验

## 问题

CompSP 是否学习到了一个和白盒安全/拒绝方向类似的潜在结构，即使它的排序相对于经验 ASR 来说是错的？

## 主假设

设：

- `A_i`：prompt `i` 的经验 ASR。
- `C_i`：prompt `i` 的 CompSP 排序分数。
- `W_i`：prompt `i` 的白盒方向/子空间分数。

如果 CompSP 学到了白盒式的潜在安全结构，那么：

1. `C_i` 和 `W_i` 应该在 prompt 级别相关。
2. 两两误差应该是结构化的：当 `C_i > C_j` 和 `A_i > A_j` 冲突时，它可能仍然和 `W_i > W_j` 一致。
3. 高 ASR-低 CompSP 和低 ASR-高 CompSP 的 prompt，应该落在白盒安全空间里的有意义区域，而不是随机失败。

## 数据

使用标准化 ASR 加载器：

```python
from compsp.asr_dataset import iter_records
records = list(iter_records("jbb-llama-pair", question_ids=[9, 30, 73]))
```

主数据集必须和经验标签对应到同一个目标模型。由于不同目标模型之间的 ASR/ALR 相关性只是中等，Qwen 标签不能拿来当 Llama 标签。

- Llama OFA 经验标签：`jbb-llama-ofa`
- Llama Pair 经验标签 + 现有排序：`jbb-llama-pair`，对应 `data/saverk/save_jbb2100_pair_llama`
- Llama DrAttack 经验标签 + 现有排序：`jbb-llama-drattack`，对应 `data/saverk/save_jbb4100_dr_ins_llama`

辅助数据：

- Qwen/4omini Pair/DrAttack 作为跨模型对照，不作为 Llama 主证据。
- HB OFA 仍需跳过 4omini/llama 中不完整的 `Q_318` 和 `Q_364`。

## 模型

白盒目标：

- 优先：`/remote-home/model/llama-3.1-8B-Instruct/Llama-3.1-8B-Instruct`
- 备选：`/remote-home/model/llama-3.1-8B/Llama-3.1-8B`

使用 Transformers，并开启 `output_hidden_states=True`。

## 第 1 阶段：复现单一拒绝方向

### 输入

构造 anchor：

- 有害 prompt：JBB/HB 原始问题。
- 无害 prompt：`/remote-home/zty/data/data_benign-behaviors.csv` 或其他 benign 集。

用模型的 chat template 格式化 prompt。

### 方向

对每一层 `l`：

```text
v_l = mean(hidden_l(last_token, harmful_prompts)) - mean(hidden_l(last_token, benign_prompts))
```

把 `v_l` 归一化成单位向量。

### 复现检查

1. 有害 prompt 在该方向上的投影应该高于 benign prompt。
2. 如果 SCANS/ABD 的模式成立，中间层的分离应该更强。
3. 可选的因果干预：
   - 给 benign prompt 加 `+alpha * v_l` 看拒绝是否增强；
   - 给 harmful prompt 加 `-alpha * v_l` 看拒绝是否减弱。

第一轮只做非因果版本也可以。

## 第 2 阶段：给 OFA 的 q1 打分

对每个 `q1` 计算：

```text
W_i,l,last = dot(hidden_l(last_token(q1)), v_l)
W_i,l,mean = mean_token dot(hidden_l(tokens(q1)), v_l)
```

保存：

- 数据集键；
- 问题编号；
- round；
- q1；
- ASR/ALR；
- 每层分数。

## 第 3 阶段：和 CompSP 排名对齐

从 `data/saverk` 读取 CompSP BTL 分数。匹配键使用：

- `questionID`；
- `item_index` / `item_id`。

要注意：Pair 数据里 `round` 不是可靠的 item ID。比如有些行会出现 `round=1`。因此应该用 JSON 列表下标，也就是 `item_index`，去匹配 `ranking_scores.json` 里的 `raw_scores_by_item[*].item_id`。`round` 只保留为溯源信息。

对每条 prompt 记录：

```text
questionID, round, ASR, ALR, CompSP_BTL, W_layer_0 ... W_layer_n
```

## 第 4 阶段：指标

### 条目级指标

对每个问题和数据集：

- Spearman(`CompSP_BTL`, `ASR`)
- Spearman(`WhiteBoxScore`, `ASR`)
- Spearman(`CompSP_BTL`, `WhiteBoxScore`)
- 在控制 ASR 后的 CompSP/WhiteBox partial correlation

按层汇报，并对所有问题求平均。

### 两两指标

对所有满足 `abs(ASR_i - ASR_j) > 0.15` 的 pair：

```text
empirical_label = ASR_i > ASR_j
compsp_label = C_i > C_j
whitebox_label = W_i > W_j
```

核心统计：

- CompSP 相对于 ASR 的准确率。
- 白盒方向相对于 ASR 的准确率。
- CompSP 和白盒方向的一致率。
- 错误一致率：

```text
P(whitebox_label == compsp_label | compsp_label != empirical_label)
```

置换基线：

- 在同一问题内打乱 `W_i`。
- 重算 wrong-agreement 1000 次。
- 报告 z-score / p-value。

### 案例分桶

挑选这些 top cases：

1. 高 ASR、低 CompSP、高 white-box；
2. 高 ASR、低 CompSP、低 white-box；
3. 低 ASR、高 CompSP、高 white-box；
4. 低 ASR、高 CompSP、低 white-box。

这些桶可以区分白盒结构是否真的解释了排序异常。

## 第 5 阶段：有害性 vs 拒绝分离

在单一拒绝方向复现之后，再扩展成两个方向：

1. 指令末尾 token 上的有害性方向。
2. 指令后/生成起始 token 上的拒绝方向。

比较有用的图：

- x 轴：有害性投影。
- y 轴：拒绝投影。
- 点颜色：ASR。
- 点形状或描边：CompSP 排序桶。

解释方式：

- 如果高 ASR-低 CompSP 的 prompt 在有害性上很高、在拒绝上很低，那 CompSP 可能更像在跟踪有害性而不是 ASR。
- 如果低 ASR-高 CompSP 的 prompt 在拒绝上很高，那 CompSP 可能更像在跟踪会触发拒绝的结构，从而阻止了实际攻击成功。

## 第 6 阶段：多维扩展

如果单方向结果不够强：

1. 先构造相对于 benign 中心的逐条 shift vector。
2. 在中间层上跑 PCA/SVD。
3. 在 top-k 正交成分上给 prompt 打分。
4. 看有没有某个成分能解释 CompSP 误差。

这会更接近那篇多维安全方向的论文。

## 要补的代码骨架

- `src/compsp/whitebox/activations.py`
- `src/compsp/whitebox/directions.py`
- `src/compsp/whitebox/score_q1.py`
- `src/compsp/whitebox/error_alignment.py`
- `scripts/build_refusal_direction.py`
- `scripts/score_whitebox_q1.py`
- `scripts/analyze_whitebox_compsp_alignment.py`

## 第一轮配置

使用：

- 模型：`/remote-home/model/llama-3.1-8B-Instruct/Llama-3.1-8B-Instruct`
- 数据：`jbb-llama-pair`
- 排名：`data/saverk/save_jbb2100_pair_llama`
- 问题编号：`[9, 30, 73, 10, 37, 15, 42, 1, 50, 41, 45, 65, 70, 72, 21, 55, 56, 78, 48, 76]`
- 指标：先看 ALR (`txl/totQA`)，再看 ASR。ALR 更直接反映拒绝/长度行为，也更少受判定噪声影响。

预计耗时：

- 用 100-200 个 anchor 构方向：单卡几分钟。
- 给 20 个问题 × 100 个 prompt × 32 层打分：单卡加 batch 可以完成。

## 风险与混杂因素

1. 经验 ASR 只做了 20 次采样，二项噪声仍然存在。
2. BTL 排名会把局部 pairwise 错误放大成很大的全局排序误差。
3. CompSP 训练的是黑盒目标模型，而白盒 Llama 方向可能属于另一个模型。
4. OFA prompt 的长度和格式可能主导某些表示方向。
5. 拒绝、有害性和回答长度是相关的，但不完全相同。

所以，最稳妥的表述应该是：

> CompSP 的排序误差是结构化的，并且和白盒安全表示有部分对齐。

在没有更强跨模型证据之前，不要直接说黑盒目标的安全边界和 Llama 白盒方向是同一个东西。
