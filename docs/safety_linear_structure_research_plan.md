# 安全线性结构研究计划

## 0. 核心假设

CompSP 的两两预测准确率虽然已经足够高，能说明它不是随机泛化，但它导出的全局排序并不完美。真正有意思的信号，可能就藏在它“排错”的方式里。

假设是：

> 如果 CompSP 代理真的学到了某种和白盒安全/拒绝方向相关的潜在结构，那么它的排序错误应该和白盒方向分数有系统性对齐，即使二者都和经验 ASR 不完全一致。

简单说：代理模型和白盒结构，可能在相对于带噪声的经验 ASR 来看时，“错得很像”。

## 1. 文献任务

要沿着四条线去检索和整理论文：

1. 单一拒绝/安全方向论文。
2. 多维安全或对齐子空间论文。
3. 表示工程 / activation steering 论文。
4. 把有害内容表征和拒绝行为分开的越狱或有害性论文。

每篇论文都要记录：

- 完整引用和链接。
- 研究的模型家族和层数。
- 方向/子空间怎么构造。
- 施加了什么干预。
- 这个方向是预测拒绝、有害性，还是两者都有。
- 有什么可复用的指标能拿来和 CompSP 比较。

交付物：

- 把 PDF 下载到 `/remote-home/zty/tidy_CompSP/参考文献`，文件名用稳定的英文命名。
- 在 `docs/safety_linear_paper_notes.md` 里写总结。

## 2. 要复用的数据

用标准化后的 ASR 接口：

- 注册表：`src/compsp/asr_registry.py`
- 加载器：`src/compsp/asr_dataset.py`

主实验第一轮的数据集必须和被解释的目标模型一致。
由于不同模型之间的 ASR/ALR 相关性只是中等，不能拿 Qwen 的攻击结果当 Llama 的经验标签。

- Llama OFA 标签：`jbb-llama-ofa`
- Llama Pair 标签 + 现成 CompSP 排名：`jbb-llama-pair` + `data/saverk/save_jbb2100_pair_llama`
- Llama DrAttack 标签 + 现成 CompSP 排名：`jbb-llama-drattack` + `data/saverk/save_jbb4100_dr_ins_llama`
- Qwen/4omini 数据只作为跨模型对照，不能作为 Llama 标签替代。
- 论文里用于排序实验的 20 个测试题编号：`[9, 30, 73, 10, 37, 15, 42, 1, 50, 41, 45, 65, 70, 72, 21, 55, 56, 78, 48, 76]`

需要的字段：

- `q1`：攻击提示词。
- `tx/totQA`：经验 ASR。
- `txl/totQA`：经验 ALR。
- `round`：用于和排序文件定位的提示词编号。
- 可选 `q0/a0/target`：攻击方法元数据。

## 3. 白盒模型和路径

服务器上看到的基座模型候选：

- `/remote-home/model/llama-3.1-8B/Llama-3.1-8B`
- `/remote-home/model/llama-3.1-8B-Instruct/Llama-3.1-8B-Instruct`

如果是拒绝/安全行为分析，优先用 instruct 版做激活提取，除非某篇论文的复现明确要求 base Llama。

隐藏状态提取优先用 Hugging Face Transformers。vLLM 适合做生成，不适合抽中间激活。

## 4. 最小白盒方向复现

目标是在 Llama-3.1-8B-Instruct 上构造一个轻量的拒绝/安全方向。

### 4.1 对照集

构造对照对：

- 来自 JBB/HB 的有害直接问题。
- 来自现有 benign CSV 的无害问题。
- 如果要复现输出侧拒绝向量，还可以用拒绝式回答和顺从式回答做对照。

可以先用最简单的方向：

```text
direction[layer] = mean(hidden_state(harmful_prompt)) - mean(hidden_state(benign_prompt))
```

更贴近拒绝方向的形式是：

```text
direction[layer] = mean(hidden_state(refusal_response_prefix)) - mean(hidden_state(compliance_response_prefix))
```

具体怎么构造，最后还是要跟着选定的参考论文来。

### 4.2 给每个 q1 打分

对每个 OFA prompt `q1`，计算：

- 每层最后一个 token 在方向上的投影；
- 每层所有 token 平均后的投影；
- 也可以在套好 chat template 后再投影。

这样就能得到白盒标量 `wb_score(q1, layer)`。

## 5. CompSP 误差对齐实验

### 5.1 条目级相关性

对每个问题和目标模型：

- 经验分数：`ASR(q1)` 和 `ALR(q1)`；
- 来自 `data/saverk` 的 CompSP BTL 分数；
- 白盒方向分数。

要计算：

- CompSP BTL 和 ASR 的 Spearman 相关；
- 白盒分数和 ASR 的 Spearman 相关；
- CompSP BTL 和白盒分数的 Spearman 相关；
- 在控制 ASR 后，CompSP 和白盒的 partial correlation。

支持假设的关键证据是：

> 在某些层和设置下，CompSP 和白盒分数彼此的相关性，比它们和经验 ASR 的相关性更强。

### 5.2 两两对齐

对提示词对 `(i, j)`：

- 经验标签：`ASR_i > ASR_j`；
- CompSP 标签：`BTL_i > BTL_j` 或分类器概率；
- 白盒标签：`wb_i > wb_j`。

把 pair 分成：

- 两个都对；
- CompSP 对 / 白盒错；
- CompSP 错 / 白盒对；
- 相对于 ASR 两个都错。

核心指标：

```text
wrong_agreement = P(white-box agrees with CompSP | CompSP is wrong relative to ASR)
```

再和同一问题内打乱后的随机基线比较。

### 5.3 高 ASR-低排序 和 低 ASR-高排序 的案例

手工查看这些 prompt：

- 经验 ASR 高，但 CompSP 排得低；
- 经验 ASR 低，但 CompSP 排得高。

对这些 prompt 看：

- 白盒分数；
- 回答长度行为 `ALR`；
- `q1` 的文本结构；
- 有没有会触发安全方向的风格或格式特征。

## 6. 控制项

为了避免过度解释，需要做几种控制：

1. 同范数随机方向。
2. 不带安全标签的 q1 上 PCA 方向。
3. 词法层面的 benign/harmful baseline。
4. 按层 sweep 时做多重比较修正。
5. 在每个问题内打乱 ASR 标签。

## 7. 预计要新增的代码

新增一个模块：

- `src/compsp/whitebox/activations.py`：隐藏状态提取。
- `src/compsp/whitebox/directions.py`：方向构造和投影。
- `src/compsp/whitebox/error_alignment.py`：CompSP/ASR/白盒一致性分析。

新增脚本：

- `scripts/build_refusal_direction.py`
- `scripts/score_q1_whitebox.py`
- `scripts/analyze_error_alignment.py`

## 8. 第一轮最小实验

1. 先用论文里的 20 个 JBB 测试题。
2. 用 `jbb-llama-pair` 的 `txfilled` 记录，因为 `data/saverk/save_jbb2100_pair_llama` 里有匹配的 BTL 分数。
3. 再在 `jbb-llama-drattack` 和 `data/saverk/save_jbb4100_dr_ins_llama` 上重复。
4. 白盒方向和经验 ASR/ALR 的对照先用 `jbb-llama-ofa`，除非后面补齐了 OFA 对应的 BTL 分数。
5. 用 50 个 harmful + 50 个 benign prompt 构造一个拒绝/安全方向。
6. 对所有 `q1` 分层打分。
7. 汇报：
   - 最好的层级相关性；
   - 相对于 ASR 的 wrong-agreement；
   - 前 20 个分歧案例。

这个第一轮实验已经足够判断这个想法值不值得继续往下做。
